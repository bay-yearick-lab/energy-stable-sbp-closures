from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
from matplotlib.patches import Polygon


def _frac_to_log(lo: float, hi: float, frac: float) -> float:
    return float(lo + frac * (hi - lo))


def _axes_log_bounds(ax):
    """Return sorted log-10 bounds of the axes.

    The helper is agnostic to whether either axis has been inverted by the
    caller; the returned ``(lx_left, lx_right, ly_bottom, ly_top)`` are
    always in ascending order, so that clipping and fractional arithmetic
    behave identically in every orientation.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if min(xlim) <= 0.0 or min(ylim) <= 0.0:
        raise ValueError("theoretical reference guides require positive log-log axes")
    lx_left = min(np.log10(xlim[0]), np.log10(xlim[1]))
    lx_right = max(np.log10(xlim[0]), np.log10(xlim[1]))
    ly_bottom = min(np.log10(ylim[0]), np.log10(ylim[1]))
    ly_top = max(np.log10(ylim[0]), np.log10(ylim[1]))
    return lx_left, lx_right, ly_bottom, ly_top


def _pick_envelope_curve(
    ax, order: float
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Pick the plotted curve whose finite-grid slope is closest to ``order``.

    This selects, among the plotted lines, the curve whose log-log slope on
    the finest half of its data is nearest to the requested formal order.
    The returned ``(xs, ys)`` are already sorted by ascending ``x``.
    """
    best: Optional[tuple[float, np.ndarray, np.ndarray]] = None
    for line in ax.get_lines():
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
        if mask.sum() < 3:
            continue
        xs, ys = xs[mask], ys[mask]
        order_idx = np.argsort(xs)
        xs, ys = xs[order_idx], ys[order_idx]
        half = max(2, len(xs) // 2)
        lx = np.log(xs[-half:])
        ly = np.log(ys[-half:])
        slope, _ = np.polyfit(lx, ly, 1)
        dist = abs(abs(slope) - float(order))
        if best is None or dist < best[0]:
            best = (dist, xs, ys)
    if best is None:
        return None
    return best[1], best[2]


def _data_to_axes_fraction(ax, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lx_left, lx_right, ly_bottom, ly_top = _axes_log_bounds(ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inverted = xlim[0] > xlim[1]
    y_inverted = ylim[0] > ylim[1]
    x_span = max(lx_right - lx_left, 1.0e-12)
    y_span = max(ly_top - ly_bottom, 1.0e-12)

    lx = np.log10(xs)
    ly = np.log10(ys)
    if x_inverted:
        xfrac = (lx_right - lx) / x_span
    else:
        xfrac = (lx - lx_left) / x_span
    if y_inverted:
        yfrac = (ly_top - ly) / y_span
    else:
        yfrac = (ly - ly_bottom) / y_span
    return xfrac, yfrac


def _axes_fraction_to_log(ax, xfrac: np.ndarray, yfrac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lx_left, lx_right, ly_bottom, ly_top = _axes_log_bounds(ax)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inverted = xlim[0] > xlim[1]
    y_inverted = ylim[0] > ylim[1]
    x_span = lx_right - lx_left
    y_span = ly_top - ly_bottom

    if x_inverted:
        lx = lx_right - xfrac * x_span
    else:
        lx = lx_left + xfrac * x_span
    if y_inverted:
        ly = ly_top - yfrac * y_span
    else:
        ly = ly_bottom + yfrac * y_span
    return lx, ly


def _sample_curve_axes(ax, xs: np.ndarray, ys: np.ndarray, points_per_segment: int = 16) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
    if mask.sum() < 2:
        return np.empty((0, 2), dtype=float)
    xs = xs[mask]
    ys = ys[mask]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    lx = np.log10(xs)
    ly = np.log10(ys)
    samples = []
    for i in range(xs.size - 1):
        t = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)
        seg_lx = (1.0 - t) * lx[i] + t * lx[i + 1]
        seg_ly = (1.0 - t) * ly[i] + t * ly[i + 1]
        seg_x = 10.0 ** seg_lx
        seg_y = 10.0 ** seg_ly
        xfrac, yfrac = _data_to_axes_fraction(ax, seg_x, seg_y)
        samples.append(np.column_stack([xfrac, yfrac]))
    xfrac, yfrac = _data_to_axes_fraction(ax, xs[-1:], ys[-1:])
    samples.append(np.column_stack([xfrac, yfrac]))
    return np.vstack(samples)


def _collect_plotted_samples(ax) -> np.ndarray:
    samples = []
    for line in ax.get_lines():
        if not line.get_visible():
            continue
        pts = _sample_curve_axes(ax, line.get_xdata(), line.get_ydata())
        if pts.size:
            samples.append(pts)
    if not samples:
        return np.empty((0, 2), dtype=float)
    return np.vstack(samples)


def _textish_plain_length(text: str) -> int:
    plain = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", text)
    plain = re.sub(r"[^A-Za-z0-9.+-]", "", plain)
    return max(len(plain), 1)


def _collect_block_boxes(ax) -> list[tuple[float, float, float, float]]:
    fig = ax.figure
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return []

    boxes: list[tuple[float, float, float, float]] = []

    def _append_bbox(artist) -> None:
        if artist is None or not artist.get_visible():
            return
        try:
            bbox = artist.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
        except Exception:
            return
        if not np.all(np.isfinite([bbox.x0, bbox.y0, bbox.x1, bbox.y1])):
            return
        boxes.append((float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1)))

    _append_bbox(ax.get_legend())
    _append_bbox(ax.title)
    _append_bbox(ax.xaxis.label)
    _append_bbox(ax.yaxis.label)
    _append_bbox(ax.xaxis.get_offset_text())
    _append_bbox(ax.yaxis.get_offset_text())
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()) + list(ax.texts):
        _append_bbox(label)
    for legend in fig.legends:
        _append_bbox(legend)
    return boxes


def _point_set_clearance(points: np.ndarray, targets: np.ndarray) -> float:
    if points.size == 0 or targets.size == 0:
        return np.inf
    diff = points[:, None, :] - targets[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return float(np.min(dist))


def _points_to_box_clearance(points: np.ndarray, box: tuple[float, float, float, float]) -> float:
    if points.size == 0:
        return np.inf
    x0, y0, x1, y1 = box
    dx = np.maximum.reduce([x0 - points[:, 0], np.zeros(points.shape[0]), points[:, 0] - x1])
    dy = np.maximum.reduce([y0 - points[:, 1], np.zeros(points.shape[0]), points[:, 1] - y1])
    return float(np.min(np.hypot(dx, dy)))


def _points_to_boxes_clearance(points: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> float:
    if not boxes:
        return np.inf
    return min(_points_to_box_clearance(points, box) for box in boxes)


def _box_to_points_clearance(box: tuple[float, float, float, float], points: np.ndarray) -> float:
    return _points_to_box_clearance(points, box)


def _box_to_box_clearance(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    return float(np.hypot(dx, dy))


def _box_to_boxes_clearance(
    box: tuple[float, float, float, float],
    boxes: Sequence[tuple[float, float, float, float]],
) -> float:
    if not boxes:
        return np.inf
    return min(_box_to_box_clearance(box, other) for other in boxes)


def _candidate_box(
    x_anchor: float,
    y_anchor: float,
    width: float,
    height: float,
    mode: str,
    pad_x: float,
    pad_y: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float], str, str]:
    if mode == "right_above":
        box = (x_anchor + pad_x, y_anchor + pad_y, x_anchor + pad_x + width, y_anchor + pad_y + height)
        text_xy = (box[0], box[1])
        return box, text_xy, "left", "bottom"
    if mode == "right_below":
        box = (x_anchor + pad_x, y_anchor - pad_y - height, x_anchor + pad_x + width, y_anchor - pad_y)
        text_xy = (box[0], box[3])
        return box, text_xy, "left", "top"
    if mode == "left_above":
        box = (x_anchor - pad_x - width, y_anchor + pad_y, x_anchor - pad_x, y_anchor + pad_y + height)
        text_xy = (box[2], box[1])
        return box, text_xy, "right", "bottom"
    box = (x_anchor - pad_x - width, y_anchor - pad_y - height, x_anchor - pad_x, y_anchor - pad_y)
    text_xy = (box[2], box[3])
    return box, text_xy, "right", "top"


_VALID_CORNERS = ("lower_left", "lower_right", "upper_left", "upper_right")


def _corner_right_angle_vertex(
    corner: str, width_frac: float, height_frac: float, margin_frac: float
) -> tuple[float, float]:
    if corner == "lower_left":
        return (margin_frac, margin_frac)
    if corner == "lower_right":
        return (1.0 - margin_frac, margin_frac)
    if corner == "upper_left":
        return (margin_frac, 1.0 - margin_frac)
    if corner == "upper_right":
        return (1.0 - margin_frac, 1.0 - margin_frac)
    raise ValueError(f"unknown corner {corner!r}")


def _corner_leg_signs(corner: str) -> tuple[float, float]:
    sign_x = +1.0 if corner in ("lower_left", "upper_left") else -1.0
    sign_y = +1.0 if corner in ("lower_left", "lower_right") else -1.0
    return sign_x, sign_y


def _triangle_bbox_frac(
    corner: str, ra_xf: float, ra_yf: float, width_frac: float, height_frac: float
) -> tuple[float, float, float, float]:
    sign_x, sign_y = _corner_leg_signs(corner)
    x0 = ra_xf if sign_x > 0 else ra_xf - width_frac
    x1 = ra_xf + width_frac if sign_x > 0 else ra_xf
    y0 = ra_yf if sign_y > 0 else ra_yf - height_frac
    y1 = ra_yf + height_frac if sign_y > 0 else ra_yf
    return x0, y0, x1, y1


def _score_corner_clearance(
    box: tuple[float, float, float, float],
    data_pts: np.ndarray,
    block_boxes: Sequence[tuple[float, float, float, float]],
) -> tuple[int, float]:
    x0, y0, x1, y1 = box
    if data_pts.size:
        mask = (
            (data_pts[:, 0] >= x0)
            & (data_pts[:, 0] <= x1)
            & (data_pts[:, 1] >= y0)
            & (data_pts[:, 1] <= y1)
        )
        inside_count = int(mask.sum())
        clearance = _points_to_box_clearance(data_pts, box)
    else:
        inside_count = 0
        clearance = np.inf
    if block_boxes:
        clearance = min(clearance, _box_to_boxes_clearance(box, block_boxes))
    return inside_count, float(clearance)


def add_theoretical_reference_line(
    ax,
    *,
    slope: float,
    label: str,
    corner: Optional[str] = None,
    anchor_xy: Optional[tuple[float, float]] = None,
    size_frac: float = 0.16,
    margin_frac: float = 0.055,
    edge_color: str = "0.18",
    face_color: str = "0.93",
    face_alpha: float = 0.55,
    hyp_lw_boost: float = 0.25,
    text_color: str = "0.10",
    leg_fontsize: float = 10.3,
    fontsize: float = 10.6,
    show_leg_labels: bool = True,
    show_hyp_label: bool = False,
    lw: float = 1.15,
    zorder: float = 1.9,
    candidate_regions: Optional[Sequence[str]] = None,
) -> None:
    """Draw a classy JCP slope triangle in a clear region of a log-log plot.

    The triangle is placed at whichever of the two slope-compatible screen
    corners has the fewest plotted points inside its bounding box, so the
    guide reads as a dedicated slope marker rather than merging with the
    convergence curves. The horizontal leg is labelled ``1`` and the vertical
    leg is labelled with the slope magnitude, following the standard
    rise-over-run convention. The x-axis label identifies the variable, so the
    ``label`` argument (for example ``r"$\\propto h^{6}$"``) is kept off the
    triangle by default but can be shown adjacent to the hypotenuse via
    ``show_hyp_label=True``.

    Parameters
    ----------
    corner:
        If given, forces the right-angle vertex into that screen corner
        (``"lower_left"``, ``"lower_right"``, ``"upper_left"``,
        ``"upper_right"``). Otherwise chosen automatically among the two
        corners whose orientation produces a hypotenuse with the requested
        slope sign.
    anchor_xy:
        Optional ``(x, y)`` in data coordinates that pins the right-angle
        vertex; overrides ``corner`` placement entirely.
    size_frac:
        Horizontal leg as a fraction of the x-axis log span.
    show_hyp_label:
        If ``True``, places ``label`` just outside the hypotenuse.

    The ``candidate_regions`` keyword is accepted for backwards compatibility
    with the previous floating-guide signature and is otherwise ignored.
    """
    del candidate_regions

    _axes_log_bounds(ax)
    lx_left, lx_right, ly_bottom, ly_top = _axes_log_bounds(ax)
    x_span = max(lx_right - lx_left, 1.0e-12)
    y_span = max(ly_top - ly_bottom, 1.0e-12)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inverted = xlim[0] > xlim[1]
    y_inverted = ylim[0] > ylim[1]

    data_slope = float(slope)
    if data_slope == 0.0:
        return

    screen_slope_sign = 1.0 if data_slope > 0.0 else -1.0
    if x_inverted:
        screen_slope_sign *= -1.0
    if y_inverted:
        screen_slope_sign *= -1.0

    if screen_slope_sign > 0.0:
        natural_corners = ("lower_right", "upper_left")
    else:
        natural_corners = ("lower_left", "upper_right")

    width_frac = float(np.clip(size_frac, 0.08, 0.30))
    height_frac = abs(data_slope) * width_frac * x_span / y_span
    max_height = 1.0 - 2.5 * margin_frac
    if height_frac > max_height:
        scale = max_height / height_frac
        height_frac = max_height
        width_frac *= scale
    if width_frac < 0.04:
        width_frac = 0.04
        height_frac = abs(data_slope) * width_frac * x_span / y_span

    if corner is None and anchor_xy is None:
        data_pts = _collect_plotted_samples(ax)
        block_boxes = _collect_block_boxes(ax)
        scored = []
        for cand in natural_corners:
            ra_xf, ra_yf = _corner_right_angle_vertex(cand, width_frac, height_frac, margin_frac)
            box = _triangle_bbox_frac(cand, ra_xf, ra_yf, width_frac, height_frac)
            inside, clear = _score_corner_clearance(box, data_pts, block_boxes)
            scored.append((inside, -clear, cand))
        scored.sort()
        corner = scored[0][2]
    elif corner is None:
        corner = natural_corners[0]

    if corner not in _VALID_CORNERS:
        raise ValueError(f"corner must be one of {_VALID_CORNERS!r}, got {corner!r}")

    if anchor_xy is not None:
        anchor_xf_arr, anchor_yf_arr = _data_to_axes_fraction(
            ax,
            np.array([float(anchor_xy[0])]),
            np.array([float(anchor_xy[1])]),
        )
        ra_xf = float(anchor_xf_arr[0])
        ra_yf = float(anchor_yf_arr[0])
    else:
        ra_xf, ra_yf = _corner_right_angle_vertex(corner, width_frac, height_frac, margin_frac)

    sign_x, sign_y = _corner_leg_signs(corner)

    # Ensure the hypotenuse's data slope has the requested sign. If the
    # natural leg orientation from the chosen corner produces the wrong sign,
    # flip the vertical leg.
    def _triangle_vertices(sx: float, sy: float) -> tuple[np.ndarray, np.ndarray]:
        xfs = np.array([ra_xf, ra_xf + sx * width_frac, ra_xf])
        yfs = np.array([ra_yf, ra_yf, ra_yf + sy * height_frac])
        return _axes_fraction_to_log(ax, xfs, yfs)

    tri_lx, tri_ly = _triangle_vertices(sign_x, sign_y)
    hyp_dx = tri_lx[2] - tri_lx[1]
    hyp_dy = tri_ly[2] - tri_ly[1]
    if abs(hyp_dx) > 1.0e-12:
        actual = hyp_dy / hyp_dx
        if actual * data_slope < 0.0:
            sign_y *= -1.0
            tri_lx, tri_ly = _triangle_vertices(sign_x, sign_y)

    tri_x = 10.0 ** tri_lx
    tri_y = 10.0 ** tri_ly

    tri_polygon = Polygon(
        list(zip(tri_x, tri_y)),
        closed=True,
        facecolor=face_color,
        edgecolor="none",
        alpha=face_alpha,
        zorder=zorder,
    )
    ax.add_patch(tri_polygon)

    ax.plot([tri_x[0], tri_x[1]], [tri_y[0], tri_y[1]],
            color=edge_color, lw=lw, zorder=zorder + 0.08, solid_capstyle="round")
    ax.plot([tri_x[0], tri_x[2]], [tri_y[0], tri_y[2]],
            color=edge_color, lw=lw, zorder=zorder + 0.08, solid_capstyle="round")
    ax.plot([tri_x[1], tri_x[2]], [tri_y[1], tri_y[2]],
            color=edge_color, lw=lw + hyp_lw_boost,
            zorder=zorder + 0.12, solid_capstyle="round")

    if show_leg_labels:
        h_mid_xf = ra_xf + 0.5 * sign_x * width_frac
        h_mid_yf = ra_yf
        h_label_offset = 0.022
        h_label_yf = h_mid_yf - h_label_offset if sign_y > 0 else h_mid_yf + h_label_offset
        h_label_va = "top" if sign_y > 0 else "bottom"
        ax.text(
            h_mid_xf, h_label_yf, r"$1$",
            transform=ax.transAxes,
            ha="center", va=h_label_va,
            fontsize=leg_fontsize, color=text_color, zorder=zorder + 0.3,
        )

        v_mid_xf = ra_xf
        v_mid_yf = ra_yf + 0.5 * sign_y * height_frac
        v_label_offset = 0.016
        v_label_xf = v_mid_xf - v_label_offset if sign_x > 0 else v_mid_xf + v_label_offset
        v_label_ha = "right" if sign_x > 0 else "left"
        slope_mag = abs(data_slope)
        if float(slope_mag).is_integer():
            slope_text = f"{int(slope_mag)}"
        else:
            slope_text = f"{slope_mag:g}"
        ax.text(
            v_label_xf, v_mid_yf, rf"${slope_text}$",
            transform=ax.transAxes,
            ha=v_label_ha, va="center",
            fontsize=leg_fontsize, color=text_color, zorder=zorder + 0.3,
        )

    if show_hyp_label and label:
        hyp_mid_xf = ra_xf + 0.5 * sign_x * width_frac
        hyp_mid_yf = ra_yf + 0.5 * sign_y * height_frac
        # Push outward, perpendicular to the hypotenuse, in the direction
        # opposite the right-angle vertex.
        out_dx = hyp_mid_xf - ra_xf
        out_dy = hyp_mid_yf - ra_yf
        length = float(np.hypot(out_dx, out_dy))
        if length > 0.0:
            out_dx /= length
            out_dy /= length
        push_frac = 0.035
        ax.text(
            hyp_mid_xf + out_dx * push_frac,
            hyp_mid_yf + out_dy * push_frac,
            label,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=zorder + 0.3,
        )


def add_formal_order_triangle(
    ax,
    order: float,
    *,
    # Legacy fractional placement (kept as a fallback when anchoring fails).
    x_right_frac: float = 0.94,
    width_frac: float = 0.18,
    y_top_frac: float = 0.84,
    # JCP-style data anchoring. When an anchor is supplied (or can be
    # inferred from plotted curves) the triangle sits next to an actual
    # convergence curve with the requested slope, not floating in space.
    anchor_xy: Optional[tuple[float, float]] = None,
    anchor_curve: Optional[Sequence] = None,
    anchor_frac: float = 0.70,
    size_decades: Optional[float] = None,
    size_frac: float = 0.22,
    offset_decades: Optional[tuple[float, float]] = None,
    x_invert_aware: bool = True,
    facecolor: str = "0.93",
    edgecolor: str = "0.55",
    textcolor: str = "0.30",
    alpha: float = 0.42,
    lw: float = 1.0,
    label_pad_frac: float = 0.04,
) -> None:
    """Draw a JCP-style log-log reference triangle.

    The reference triangle has a horizontal leg labelled ``1`` and a vertical
    leg labelled by the formal ``order``. By default the triangle is anchored
    next to the plotted curve whose finest-grid slope most closely matches
    ``order``, so that readers can verify the slope visually instead of
    comparing against a floating box.

    Parameters
    ----------
    anchor_xy:
        If given, a point ``(x, y)`` in data coordinates that fixes the
        right-hand bottom vertex of the reference triangle (before the
        optional ``offset_decades`` shift).
    anchor_curve:
        An ``(xs, ys)`` pair whose ``anchor_frac`` point is used as the
        anchor. Ignored if ``anchor_xy`` is provided.
    anchor_frac:
        Fractional position along the (log-sorted) anchor curve that
        selects the anchor point; ``0.75`` places the triangle near the
        fine end of the sweep, which is the JCP convention.
    size_decades:
        Width of the horizontal leg in decades on the x-axis. When ``None``
        (the default) the width is set to ``size_frac`` of the visible
        x-range, which keeps the triangle legible both on wide MMS sweeps
        and on short refinement studies that span less than a decade in
        ``h``.
    size_frac:
        Fallback fractional width used when ``size_decades`` is not
        supplied.
    offset_decades:
        ``(dx, dy)`` offset in decades that shifts the triangle away from
        the anchor point. When ``None`` the helper centres the triangle
        around the anchor curve and pushes it above by a small amount
        scaled to the y-range. ``dx`` is applied in the *visual* direction
        (the helper flips its sign when the x-axis is inverted, so that a
        negative ``dx`` always means "to the left of the anchor" on
        screen).
    label_pad_frac:
        Distance in axes fractions between the triangle edges and the slope
        labels.
    """
    lx_left, lx_right, ly_bottom, ly_top = _axes_log_bounds(ax)

    anchor_log: Optional[tuple[float, float]] = None
    if anchor_xy is not None:
        ax_anchor_x = float(anchor_xy[0])
        ax_anchor_y = float(anchor_xy[1])
        if ax_anchor_x > 0.0 and ax_anchor_y > 0.0:
            anchor_log = (np.log10(ax_anchor_x), np.log10(ax_anchor_y))
    elif anchor_curve is not None:
        xs = np.asarray(anchor_curve[0], dtype=float)
        ys = np.asarray(anchor_curve[1], dtype=float)
        mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0) & (ys > 0.0)
        if mask.sum() >= 2:
            xs, ys = xs[mask], ys[mask]
            idx = np.argsort(xs)
            xs, ys = xs[idx], ys[idx]
            lx = np.log10(xs)
            ly = np.log10(ys)
            pos = float(np.clip(anchor_frac, 0.0, 1.0)) * (len(lx) - 1)
            i = int(np.clip(np.floor(pos), 0, len(lx) - 2))
            frac = pos - i
            anchor_log = (
                (1 - frac) * lx[i] + frac * lx[i + 1],
                (1 - frac) * ly[i] + frac * ly[i + 1],
            )
    if anchor_log is None:
        picked = _pick_envelope_curve(ax, order)
        if picked is not None:
            xs, ys = picked
            lx = np.log10(xs)
            ly = np.log10(ys)
            pos = float(np.clip(anchor_frac, 0.0, 1.0)) * (len(lx) - 1)
            i = int(np.clip(np.floor(pos), 0, len(lx) - 2))
            frac = pos - i
            anchor_log = (
                (1 - frac) * lx[i] + frac * lx[i + 1],
                (1 - frac) * ly[i] + frac * ly[i + 1],
            )

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_inverted = x_invert_aware and xlim[0] > xlim[1]
    y_inverted = ylim[0] > ylim[1]
    x_span = max(lx_right - lx_left, 1.0e-12)
    y_span = max(ly_top - ly_bottom, 1.0e-12)

    if size_decades is not None:
        width_dec = float(size_decades)
    else:
        width_dec = float(size_frac) * x_span
    width_dec = max(width_dec, 0.02)
    if float(order) * width_dec > 0.75 * y_span:
        width_dec = 0.75 * y_span / max(float(order), 1.0e-6)

    if offset_decades is None:
        offset_dec = (-0.25 * width_dec, 0.20 * y_span)
    else:
        offset_dec = (float(offset_decades[0]), float(offset_decades[1]))

    if anchor_log is not None:
        dx_dec, dy_dec = offset_dec
        sign_x = -1.0 if x_inverted else 1.0
        ax_anchor_log = (
            anchor_log[0] + sign_x * dx_dec,
            anchor_log[1] + dy_dec,
        )
        half = 0.5 * width_dec
        x_center_log = ax_anchor_log[0]
        y_high_log = ax_anchor_log[1] + float(order) * half
        x_left_log = x_center_log - half
        x_right_log = x_center_log + half
        y_low_log = y_high_log - float(order) * (x_right_log - x_left_log)
    else:
        x_left_frac_local = x_right_frac - width_frac
        x_left_log = _frac_to_log(lx_left, lx_right, x_left_frac_local)
        x_right_log = _frac_to_log(lx_left, lx_right, x_right_frac)
        y_high_log = _frac_to_log(ly_bottom, ly_top, y_top_frac)
        y_low_log = y_high_log + float(order) * (x_right_log - x_left_log)

    x_left_log = float(np.clip(x_left_log, lx_left, lx_right))
    x_right_log = float(np.clip(x_right_log, lx_left, lx_right))
    if x_right_log <= x_left_log:
        x_left_log, x_right_log = sorted((x_left_log, x_right_log))
        x_right_log = x_left_log + 1.0e-6
    y_low_log = float(np.clip(y_low_log, ly_bottom, ly_top))
    y_high_log = float(np.clip(y_high_log, ly_bottom, ly_top))
    if y_high_log <= y_low_log:
        y_high_log = y_low_log + 1.0e-6

    x_left = 10.0 ** x_left_log
    x_right = 10.0 ** x_right_log
    y_low = 10.0 ** y_low_log
    y_high = 10.0 ** y_high_log

    if x_inverted:
        tri_pts = [(x_left, y_low), (x_right, y_low), (x_right, y_high)]
        x_leg_log = x_right_log
        slope_x0, slope_x1 = x_right, x_left
        order_label_dx = +0.6 * label_pad_frac
        order_label_ha = "left"
    else:
        tri_pts = [(x_left, y_low), (x_right, y_low), (x_left, y_high)]
        x_leg_log = x_left_log
        slope_x0, slope_x1 = x_left, x_right
        order_label_dx = -0.6 * label_pad_frac
        order_label_ha = "right"

    def _x_axes_frac(x_log: float) -> float:
        if x_inverted:
            return (lx_right - x_log) / max(lx_right - lx_left, 1.0e-12)
        return (x_log - lx_left) / max(lx_right - lx_left, 1.0e-12)

    def _y_axes_frac(y_log: float) -> float:
        if y_inverted:
            return (ly_top - y_log) / max(ly_top - ly_bottom, 1.0e-12)
        return (y_log - ly_bottom) / max(ly_top - ly_bottom, 1.0e-12)

    tri = Polygon(
        tri_pts,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        joinstyle="miter",
        zorder=2.2,
    )
    ax.add_patch(tri)
    ax.plot([x_left, x_right], [y_low, y_low], color=edgecolor, lw=lw, zorder=2.4)
    ax.plot([10.0 ** x_leg_log, 10.0 ** x_leg_log], [y_low, y_high],
            color=edgecolor, lw=lw, zorder=2.4)
    ax.plot([slope_x0, slope_x1], [y_high, y_low], color="0.35", lw=lw + 0.15, zorder=2.5)

    q_text = f"{int(order)}" if float(order).is_integer() else f"{order:g}"
    x_left_frac_final = _x_axes_frac(x_left_log)
    x_right_frac_final = _x_axes_frac(x_right_log)
    x_leg_frac_final = _x_axes_frac(x_leg_log)
    y_low_frac_final = _y_axes_frac(y_low_log)
    y_high_frac_final = _y_axes_frac(y_high_log)
    y_mid_frac_final = 0.5 * (y_low_frac_final + y_high_frac_final)

    ax.text(
        0.5 * (x_left_frac_final + x_right_frac_final),
        y_low_frac_final - label_pad_frac,
        r"$1$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.1,
        color=textcolor,
    )
    ax.text(
        x_leg_frac_final + order_label_dx,
        y_mid_frac_final,
        rf"${q_text}$",
        transform=ax.transAxes,
        ha=order_label_ha,
        va="center",
        fontsize=9.3,
        color=textcolor,
    )
