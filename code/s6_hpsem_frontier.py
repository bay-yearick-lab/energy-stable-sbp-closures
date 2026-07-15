"""hp spectral-element frontier: the strongest variational competitor with
its free parameters optimized per error target.

For the heat-decay problem (convergence-ladder configuration) and the
closed-box standing-mode acoustics, the spectral element sweeps degree
P in {2,...,12} and element count K with total DOFs up to ~200 per
direction, each configuration at its own tier-B stable step, with measured
single-thread CPU. For each error target the CHEAPEST admissible hp
configuration is the frontier point; the spline routes sweep their own N
ladder under the identical protocol. The frontier table reports, per
target: the best hp-SEM configuration (P, K, DOFs, CPU) and the best spline
configuration (family, N, CPU), plus the winner by CPU and by DOFs.

Verification (--verify): every recorded value is finite; the frontier is
monotone (cost non-decreasing in tightening targets); heat uses the same
minimum DOFs on both sides, acoustics uses no more spline DOFs and strictly
fewer at the loose and tight targets, and optimized hp-SEM is cheaper in CPU
time at every tested target.

Output: code/results/hpsem_frontier_summary.json
"""
from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

import boundary_spline_analysis as ba
from heat_cavity_core import default_heat_modes, per_method_diffusion_dt, run_heat_cavity
from project_paths import RESULTSDIR, ensure_generated_dirs
from sem_operator import SEMOperator1D

HEAT_CFG = dict(m_center=8, n_center=8, half_band=1, viscosity=5.0e-5)
T_END = 0.5
HEAT_TARGETS = (1e-3, 1e-5, 1e-8)
WAVE_TARGETS = (1e-2, 1e-4, 1e-6)
WAVE_MODE = 6
WAVE_T = 6.0
SEM_P = (2, 3, 4, 5, 6, 8, 10, 12)
MAX_DOF = 220
SPLINE_N = (17, 25, 33, 49, 65, 97, 129)
RESCORE_PROTOCOL = "hpsem-frontier-spline-rescore-v2"
SEM_RESCORE_PROTOCOL = "hpsem-frontier-sem-rescore-v3"
THREAD_ENV_NAMES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def timing_stamp():
    return {
        "metric": "process_time_seconds",
        "cpu": platform.processor() or _cpu_model(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "thread_env": {name: os.environ.get(name, "unset") for name in THREAD_ENV_NAMES},
        "nice": os.nice(0),
    }


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def expected_config():
    return {
        "sem_P": list(SEM_P),
        "max_dof": MAX_DOF,
        "heat_targets": list(HEAT_TARGETS),
        "wave_targets": list(WAVE_TARGETS),
        "spline_N": list(SPLINE_N),
        "heat_norm": "relative L2 with actual-node trapezoid weights",
        "wave_norm": "full-domain coupled relative L2 with actual-node trapezoid weights",
    }


def _nonuniform_trap(x):
    nodes = np.asarray(x, float)
    weights = np.empty_like(nodes)
    weights[0] = 0.5 * (nodes[1] - nodes[0])
    weights[-1] = 0.5 * (nodes[-1] - nodes[-2])
    weights[1:-1] = 0.5 * (nodes[2:] - nodes[:-2])
    return weights


def _coupled_rel_l2(result):
    weights = _nonuniform_trap(result.grid)
    pressure_error = np.asarray(result.pressure) - np.asarray(result.pressure_exact)
    velocity_error = np.asarray(result.velocity) - np.asarray(result.velocity_exact)
    numerator = np.sum(weights * (pressure_error**2 + velocity_error**2))
    denominator = np.sum(
        weights
        * (np.asarray(result.pressure_exact) ** 2 + np.asarray(result.velocity_exact) ** 2)
    )
    return float(np.sqrt(numerator / denominator))


def sem_heat_case(P, K, modes):
    op = SEMOperator1D(P, K)
    npts = op.npts
    x, nu = op.x, float(modes.viscosity)
    lam = float(np.max(np.abs(np.linalg.eigvals(op.D2).real))) * nu * 2.0
    nsteps = max(1, int(np.ceil(T_END / (0.4 * 2.785 / max(lam, 1e-30)))))
    dt = T_END / nsteps
    U = modes.field_grid(x, x, 0.0)
    U[[0, -1], :] = 0.0
    U[:, [0, -1]] = 0.0

    def rhs(Uv):
        R = nu * (op.D2 @ Uv + Uv @ op.D2.T)
        R[[0, -1], :] = 0.0
        R[:, [0, -1]] = 0.0
        return R

    t0 = time.process_time()
    t = 0.0
    for _ in range(nsteps):
        k1 = rhs(U); k2 = rhs(U + 0.5 * dt * k1)
        k3 = rhs(U + 0.5 * dt * k2); k4 = rhs(U + dt * k3)
        U = U + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        U[[0, -1], :] = 0.0
        U[:, [0, -1]] = 0.0
        t += dt
    cpu = time.process_time() - t0
    ue = modes.field_grid(x, x, t)
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    w2 = np.outer(w, w)
    err = float(np.sqrt(np.sum(w2 * (U - ue) ** 2)) / np.sqrt(np.sum(w2 * ue ** 2)))
    return err, cpu, npts


def sem_wave_case(P, K):
    op = SEMOperator1D(P, K)
    x, npts = op.x, op.npts
    dt = 0.012 / (npts - 1)
    nsteps = int(np.ceil(WAVE_T / dt)); dt = WAVE_T / nsteps
    c = WAVE_MODE * np.pi
    Pf, U = np.cos(c * x), np.zeros_like(x)
    D1 = op.D1
    t0 = time.process_time()
    for _ in range(nsteps):
        def rhs(Pv, Uv):
            dP = -(D1 @ Uv); dU = -(D1 @ Pv)
            dU[0] = dU[-1] = 0.0
            return dP, dU
        k1p, k1u = rhs(Pf, U)
        k2p, k2u = rhs(Pf + 0.5 * dt * k1p, U + 0.5 * dt * k1u)
        k3p, k3u = rhs(Pf + 0.5 * dt * k2p, U + 0.5 * dt * k2u)
        k4p, k4u = rhs(Pf + dt * k3p, U + dt * k3u)
        Pf = Pf + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        U = U + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        U[0] = U[-1] = 0.0
    cpu = time.process_time() - t0
    pr = np.cos(c * x) * np.cos(c * WAVE_T)
    ur = np.sin(c * x) * np.sin(c * WAVE_T)
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2]); w[0] = 0.5 * (x[1] - x[0]); w[-1] = 0.5 * (x[-1] - x[-2])
    err = float(np.sqrt(np.sum(w * ((Pf - pr) ** 2 + (U - ur) ** 2)))
                / np.sqrt(np.sum(w * (pr ** 2 + ur ** 2))))
    return err, cpu, npts


def spline_heat_case(label, npts, modes):
    dt = per_method_diffusion_dt(label, npts=npts, viscosity=modes.viscosity, t_end=T_END)
    t0 = time.process_time()
    res = run_heat_cavity(label, modes, npts=npts, t_end=T_END, dt=dt,
                          record_count=2, snapshot_count=2, enforcement="horth")
    return float(res.u_rel_l2[-1]), time.process_time() - t0, npts


def spline_wave_case(label, npts):
    op = ba.build_operator(ba.scheme_kind(label), int(label[-1]), npts)
    t0 = time.process_time()
    res = ba.simulate_reflected_acoustics(
        op,
        lambda x, t: np.cos(WAVE_MODE * np.pi * np.asarray(x, float)) * np.cos(WAVE_MODE * np.pi * t),
        lambda x, t: np.sin(WAVE_MODE * np.pi * np.asarray(x, float)) * np.sin(WAVE_MODE * np.pi * t),
        WAVE_T, cfl=0.012, interior_strip=0, record_count=4)
    return _coupled_rel_l2(res), time.process_time() - t0, npts


def frontier(cases, targets):
    """cases: list of dicts with err, cpu, dofs, tag; per target the cheapest
    admissible configuration by cpu and by dofs."""
    out = {}
    for tgt in targets:
        ok = [c for c in cases if c["err"] <= tgt]
        out[f"{tgt:g}"] = {
            "by_cpu": min(ok, key=lambda c: c["cpu"]) if ok else None,
            "by_dofs": min(ok, key=lambda c: c["dofs"]) if ok else None,
            "n_admissible": len(ok),
        }
    return out


def run_all():
    modes = default_heat_modes(**HEAT_CFG)
    out = {"config": expected_config(), "timing": timing_stamp(),
           "sem_heat": [], "sem_wave": [], "spline_heat": [], "spline_wave": []}

    # crash-resume: one JSON line per completed case; a kill costs one case
    partial = RESULTSDIR / "hpsem_frontier_v2.partial.jsonl"
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            try:
                rec = json.loads(line)
                done[rec["k"]] = rec["v"]
            except (json.JSONDecodeError, KeyError):
                continue   # torn last line from a kill mid-write
        print(f"resume: {len(done)} completed cases", flush=True)

    def one(group, tag, fn):
        k = f"{group}:{tag}"
        if k in done:
            out[group].append(done[k])
            return
        err, cpu, dofs = fn()
        rec = {"tag": tag, "err": err, "cpu": cpu, "dofs": dofs}
        out[group].append(rec)
        with partial.open("a") as f:
            f.write(json.dumps({"k": k, "v": rec}) + "\n")

    for P in SEM_P:
        for K in range(2, MAX_DOF // P):
            npts = K * P + 1
            if npts > MAX_DOF or npts < 13:
                continue
            one("sem_heat", f"P{P}K{K}",
                lambda P=P, K=K: sem_heat_case(P, K, modes))
    for P in SEM_P:
        for K in range(2, MAX_DOF // P):
            npts = K * P + 1
            if npts > MAX_DOF or npts < 13:
                continue
            one("sem_wave", f"P{P}K{K}", lambda P=P, K=K: sem_wave_case(P, K))
        print(f"sem P={P} done", flush=True)
    # the spline routes get the same freedom the hp sweep gives the SEM:
    # degree is a free parameter, optimized per error target
    for label in ("BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8"):
        for npts in SPLINE_N:
            one("spline_heat", f"{label}N{npts}",
                lambda label=label, npts=npts: spline_heat_case(label, npts, modes))
            one("spline_wave", f"{label}N{npts}",
                lambda label=label, npts=npts: spline_wave_case(label, npts))
        print(f"{label} done", flush=True)
    out["frontier"] = {
        "heat_sem": frontier(out["sem_heat"], HEAT_TARGETS),
        "heat_spline": frontier(out["spline_heat"], HEAT_TARGETS),
        "wave_sem": frontier(out["sem_wave"], WAVE_TARGETS),
        "wave_spline": frontier(out["spline_wave"], WAVE_TARGETS),
    }
    return out


def _expected_sem_cases():
    expected = {}
    for P in SEM_P:
        for K in range(2, MAX_DOF // P):
            npts = K * P + 1
            if npts <= MAX_DOF and npts >= 13:
                expected[f"P{P}K{K}"] = npts
    return expected


def _expected_spline_cases():
    return {
        f"{label}N{npts}": npts
        for label in ("BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8")
        for npts in SPLINE_N
    }


def _validate_case_group(name, cases, expected):
    if not isinstance(cases, list):
        raise ValueError(f"{name} must be a list")
    indexed = {}
    for case in cases:
        if set(case) != {"tag", "err", "cpu", "dofs"}:
            raise ValueError(f"unexpected {name} case schema: {case}")
        tag = case["tag"]
        if tag in indexed:
            raise ValueError(f"duplicate {name} case: {tag}")
        if tag not in expected:
            raise ValueError(f"unexpected {name} case: {tag}")
        if case["dofs"] != expected[tag]:
            raise ValueError(f"wrong DOF count for {name} {tag}")
        if not np.isfinite(case["err"]) or case["err"] < 0.0:
            raise ValueError(f"invalid error for {name} {tag}")
        if not np.isfinite(case["cpu"]) or case["cpu"] <= 0.0:
            raise ValueError(f"invalid CPU time for {name} {tag}")
        indexed[tag] = case
    if set(indexed) != set(expected):
        missing = sorted(set(expected) - set(indexed))
        raise ValueError(f"incomplete {name} sweep; missing={missing}")


def _assemble_frontiers(out):
    return {
        "heat_sem": frontier(out["sem_heat"], HEAT_TARGETS),
        "heat_spline": frontier(out["spline_heat"], HEAT_TARGETS),
        "wave_sem": frontier(out["sem_wave"], WAVE_TARGETS),
        "wave_spline": frontier(out["spline_wave"], WAVE_TARGETS),
    }


def _validate_timing(timing):
    required = {"metric", "cpu", "platform", "python", "numpy", "thread_env", "nice"}
    if not isinstance(timing, dict) or set(timing) != required:
        raise ValueError("hp-SEM timing provenance is incomplete")
    if timing["metric"] != "process_time_seconds":
        raise ValueError("hp-SEM timing metric must be process CPU time")
    if not all(timing[key] for key in ("cpu", "platform", "python", "numpy")):
        raise ValueError("hp-SEM timing identity contains an empty field")
    if timing["thread_env"] != {name: "1" for name in THREAD_ENV_NAMES}:
        raise ValueError("hp-SEM timing was not pinned to one numerical thread")
    if timing["nice"] != 19:
        raise ValueError("hp-SEM timing was not collected at nice 19")


def replace_sem_sweeps(existing, sem_heat, sem_wave, timing):
    """Preserve corrected spline cases and replace both SEM sweeps exactly."""
    spline_expected = _expected_spline_cases()
    _validate_case_group("spline_heat", existing.get("spline_heat"), spline_expected)
    _validate_case_group("spline_wave", existing.get("spline_wave"), spline_expected)
    sem_expected = _expected_sem_cases()
    _validate_case_group("sem_heat", sem_heat, sem_expected)
    _validate_case_group("sem_wave", sem_wave, sem_expected)
    _validate_timing(timing)
    out = {
        "config": expected_config(),
        "timing": deepcopy(timing),
        "sem_heat": deepcopy(sem_heat),
        "sem_wave": deepcopy(sem_wave),
        "spline_heat": deepcopy(existing["spline_heat"]),
        "spline_wave": deepcopy(existing["spline_wave"]),
    }
    out["frontier"] = _assemble_frontiers(out)
    return out


def _sem_checkpoint_signature():
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_sha = hashlib.sha256(
        json.dumps(expected_config(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return source_sha, config_sha


def _load_sem_checkpoint(path):
    done = {}
    if not path.exists():
        return done
    source_sha, config_sha = _sem_checkpoint_signature()
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
            if (item.get("protocol") != SEM_RESCORE_PROTOCOL
                    or item.get("source_sha256") != source_sha
                    or item.get("config_sha256") != config_sha):
                continue
            group, rec = item["group"], item["record"]
            done[(group, rec["tag"])] = rec
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return done


def _sem_case_job(group, P, K):
    tag = f"P{P}K{K}"
    fn = (lambda: sem_heat_case(P, K, default_heat_modes(**HEAT_CFG))) \
        if group == "sem_heat" else (lambda: sem_wave_case(P, K))
    err, cpu, dofs = fn()
    return group, {"tag": tag, "err": float(err), "cpu": float(cpu), "dofs": int(dofs)}


def rescore_sem_cases(existing, checkpoint, workers):
    if workers < 1:
        raise ValueError("SEM rescore needs at least one worker")
    expected = _expected_sem_cases()
    jobs = []
    for tag in expected:
        p_text, k_text = tag[1:].split("K")
        for group in ("sem_heat", "sem_wave"):
            jobs.append((group, int(p_text), int(k_text), tag))
    done = _load_sem_checkpoint(checkpoint)
    records = {"sem_heat": [], "sem_wave": []}
    source_sha, config_sha = _sem_checkpoint_signature()
    pending = [(group, P, K, tag) for group, P, K, tag in jobs
               if (group, tag) not in done]
    print(f"SEM rescore: {len(done)} resumed, {len(pending)} pending, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_sem_case_job, group, P, K): (group, tag)
                   for group, P, K, tag in pending}
        for count, future in enumerate(as_completed(futures), 1):
            group, rec = future.result()
            done[(group, rec["tag"])] = rec
            with checkpoint.open("a") as stream:
                stream.write(json.dumps({
                    "protocol": SEM_RESCORE_PROTOCOL,
                    "source_sha256": source_sha,
                    "config_sha256": config_sha,
                    "group": group,
                    "record": rec,
                }) + "\n")
            if count % 50 == 0 or count == len(pending):
                print(f"SEM rescore completed {count}/{len(pending)} new cases", flush=True)
    for group in records:
        records[group] = [done[(group, tag)] for tag in expected]
    return replace_sem_sweeps(
        existing, records["sem_heat"], records["sem_wave"], timing_stamp()
    )


def _load_rescore_checkpoint(path):
    done = {}
    if path is None or not path.exists():
        return done
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
            if item.get("protocol") != RESCORE_PROTOCOL:
                continue
            group, tag, rec = item["group"], item["tag"], item["record"]
            if group not in ("spline_heat", "spline_wave") or rec.get("tag") != tag:
                continue
            done[(group, tag)] = rec
        except (json.JSONDecodeError, KeyError, AttributeError):
            continue
    return done


def rescore_splines(existing, checkpoint, heat_case=spline_heat_case,
                    wave_case=spline_wave_case):
    """Preserve the valid SEM sweep and recompute every spline case.

    A protocol-specific checkpoint prevents results from the retired
    trimmed/uniform score from being resumed under the corrected common norm.
    """
    source_config = existing.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("missing hp-SEM source configuration")
    base_keys = ("sem_P", "max_dof", "heat_targets", "wave_targets", "spline_N")
    if any(source_config.get(key) != expected_config()[key] for key in base_keys):
        raise ValueError("source hp-SEM sweep does not match the released grid")

    sem_expected = _expected_sem_cases()
    _validate_case_group("sem_heat", existing.get("sem_heat"), sem_expected)
    _validate_case_group("sem_wave", existing.get("sem_wave"), sem_expected)
    out = {
        "config": expected_config(),
        "timing": timing_stamp(),
        "sem_heat": deepcopy(existing["sem_heat"]),
        "sem_wave": deepcopy(existing["sem_wave"]),
        "spline_heat": [],
        "spline_wave": [],
    }
    done = _load_rescore_checkpoint(checkpoint)
    modes = default_heat_modes(**HEAT_CFG)

    def one(group, label, npts, fn):
        tag = f"{label}N{npts}"
        key = (group, tag)
        if key in done:
            rec = done[key]
        else:
            err, cpu, dofs = fn()
            rec = {"tag": tag, "err": float(err), "cpu": float(cpu), "dofs": int(dofs)}
            if checkpoint is not None:
                with checkpoint.open("a") as stream:
                    stream.write(json.dumps({
                        "protocol": RESCORE_PROTOCOL,
                        "group": group,
                        "tag": tag,
                        "record": rec,
                    }) + "\n")
        out[group].append(rec)

    for label in ("BMF4", "BMG4", "BMF6", "BMG6", "BMF8", "BMG8"):
        for npts in SPLINE_N:
            one("spline_heat", label, npts,
                lambda label=label, npts=npts: heat_case(label, npts, modes))
            one("spline_wave", label, npts,
                lambda label=label, npts=npts: wave_case(label, npts))
        print(f"rescored {label}", flush=True)
    spline_expected = _expected_spline_cases()
    _validate_case_group("spline_heat", out["spline_heat"], spline_expected)
    _validate_case_group("spline_wave", out["spline_wave"], spline_expected)
    out["frontier"] = _assemble_frontiers(out)
    return out


def verify_record(out):
    if out.get("config") != expected_config():
        raise ValueError("hp-SEM configuration differs from the released protocol")
    _validate_timing(out.get("timing"))
    sem_expected = _expected_sem_cases()
    spline_expected = _expected_spline_cases()
    for group in ("sem_heat", "sem_wave", "spline_heat", "spline_wave"):
        _validate_case_group(
            group, out.get(group),
            sem_expected if group.startswith("sem_") else spline_expected,
        )
    reconstructed = _assemble_frontiers(out)
    if out.get("frontier") != reconstructed:
        raise AssertionError("stored hp-SEM frontier is not the frontier of the raw cases")
    # Measured frontier contract (each side optimizes its free degree):
    # hp-SEM's diagonal mass wins CPU at every tested target; heat DOFs tie;
    # spline acoustic DOFs are no worse and are strictly fewer at the loose
    # and tight targets.
    fh = out["frontier"]
    target_map = {
        "heat": ("0.001", "1e-05", "1e-08"),
        "wave": ("0.01", "0.0001", "1e-06"),
    }
    for problem, targets in target_map.items():
        for target in targets:
            sem_cpu = fh[f"{problem}_sem"][target]["by_cpu"]
            spline_cpu = fh[f"{problem}_spline"][target]["by_cpu"]
            if sem_cpu and spline_cpu and not sem_cpu["cpu"] < spline_cpu["cpu"]:
                raise AssertionError(("optimized hp-SEM CPU lead", problem, target, sem_cpu, spline_cpu))
            sem_dofs = fh[f"{problem}_sem"][target]["by_dofs"]
            spline_dofs = fh[f"{problem}_spline"][target]["by_dofs"]
            if sem_dofs and spline_dofs:
                if problem == "heat" and spline_dofs["dofs"] != sem_dofs["dofs"]:
                    raise AssertionError(("heat DOF tie", target, spline_dofs, sem_dofs))
                if problem == "wave" and spline_dofs["dofs"] > sem_dofs["dofs"]:
                    raise AssertionError(("wave DOFs", target, spline_dofs, sem_dofs))
    for target in ("0.01", "1e-06"):
        sem_dofs = fh["wave_sem"][target]["by_dofs"]
        spline_dofs = fh["wave_spline"][target]["by_dofs"]
        if sem_dofs and spline_dofs and not spline_dofs["dofs"] < sem_dofs["dofs"]:
            raise AssertionError(("strict wave DOF lead", target, spline_dofs, sem_dofs))


def verify(out):
    verify_record(out)
    print("hpsem_frontier verify: ALL PASS")


def main():
    ensure_generated_dirs()
    path = RESULTSDIR / "hpsem_frontier_summary.json"
    if "--verify" in sys.argv:
        verify(json.loads(path.read_text()))
        return
    if "--rescore-sem" in sys.argv:
        workers = 1
        if "--workers" in sys.argv:
            workers = int(sys.argv[sys.argv.index("--workers") + 1])
        checkpoint = RESULTSDIR / "hpsem_frontier.rescore_sem_v3.partial.jsonl"
        out = rescore_sem_cases(json.loads(path.read_text()), checkpoint, workers)
    elif "--rescore-splines" in sys.argv:
        checkpoint = RESULTSDIR / "hpsem_frontier.rescore_splines_v2.partial.jsonl"
        out = rescore_splines(json.loads(path.read_text()), checkpoint=checkpoint)
    else:
        out = run_all()
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    verify(out)


if __name__ == "__main__":
    main()
