#!/usr/bin/env python3
"""Mixed compatible GLL and histopolant spectral-element flow factors."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import BarycentricInterpolator
from scipy.sparse import coo_matrix, csr_matrix, identity

from sem_operator import gll_nodes_weights
from tensor_compatible_flow import CoefficientAxis, TensorCompatibleFlow2D


Array = np.ndarray


def _local_exact_sequence(
    degree: int,
    points: Array,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate the GLL nodal and histopolant exact sequence stably."""
    nodes, _weights = gll_nodes_weights(degree)
    interpolator = BarycentricInterpolator(
        nodes,
        np.eye(degree + 1),
        axis=0,
        rng=np.random.default_rng(20260820 + int(degree)),
    )
    coordinates = np.asarray(points, dtype=float)
    b0 = np.asarray(interpolator(coordinates), dtype=float)
    d0 = np.asarray(interpolator.derivative(coordinates, der=1), dtype=float)
    d2 = np.asarray(interpolator.derivative(coordinates, der=2), dtype=float)
    b1 = -np.cumsum(d0[:, :-1], axis=1)
    d1 = -np.cumsum(d2[:, :-1], axis=1)
    return b0, b1, d0, d1


def compatible_sem_axis(
    degree: int,
    nelem: int,
    length: float = 1.0,
    quadrature_points_per_element: int | None = None,
    penalty_gamma: float | None = None,
) -> CoefficientAxis:
    """Assemble a one-dimensional nodal and histopolant exact sequence."""
    p = int(degree)
    elements = int(nelem)
    domain_length = float(length)
    if p < 2 or elements < 1:
        raise ValueError("compatible SEM requires degree at least two and one element")
    if not np.isfinite(domain_length) or domain_length <= 0.0:
        raise ValueError("length must be finite and positive")
    order = (
        max((3 * p + 1) // 2, 3)
        if quadrature_points_per_element is None
        else int(quadrature_points_per_element)
    )
    gauss, gauss_weight = leggauss(order)
    gll, _gll_weight = gll_nodes_weights(p)
    b0_local, b1_reference, d0_reference, d1_reference = _local_exact_sequence(
        p, gauss
    )
    trace_b0, trace_b1_reference, trace_d0_reference, trace_d1_reference = (
        _local_exact_sequence(p, np.array([-1.0, 1.0]))
    )
    h = domain_length / elements
    jacobian = 0.5 * h
    gamma = 4.0 * p**2 if penalty_gamma is None else float(penalty_gamma)
    if not np.isfinite(gamma) or gamma < 4.0 * p**2:
        raise ValueError("penalty_gamma must be finite and at least 4 p^2")
    n0 = elements * p + 1
    n1 = elements * p
    qsize = elements * order

    rows0: list[int] = []
    columns0: list[int] = []
    values0: list[float] = []
    rows1: list[int] = []
    columns1: list[int] = []
    values1: list[float] = []
    derivative0_values: list[float] = []
    derivative1_values: list[float] = []
    incidence_rows: list[int] = []
    incidence_columns: list[int] = []
    incidence_values: list[float] = []
    q = np.empty(qsize)
    w = np.empty(qsize)
    x0 = np.empty(n0)
    x1 = np.empty(n1)
    for element in range(elements):
        qslice = slice(element * order, (element + 1) * order)
        q[qslice] = element * h + jacobian * (gauss + 1.0)
        w[qslice] = jacobian * gauss_weight
        node_columns = element * p + np.arange(p + 1)
        edge_columns = element * p + np.arange(p)
        for local_row, global_row in enumerate(range(qslice.start, qslice.stop)):
            for local_column, global_column in enumerate(node_columns):
                rows0.append(global_row)
                columns0.append(int(global_column))
                values0.append(float(b0_local[local_row, local_column]))
                derivative0_values.append(
                    float(d0_reference[local_row, local_column] / jacobian)
                )
            for local_column, global_column in enumerate(edge_columns):
                rows1.append(global_row)
                columns1.append(int(global_column))
                values1.append(
                    float(b1_reference[local_row, local_column] / jacobian)
                )
                derivative1_values.append(
                    float(d1_reference[local_row, local_column] / jacobian**2)
                )
        for local_edge, global_edge in enumerate(edge_columns):
            incidence_rows.extend([int(global_edge), int(global_edge)])
            incidence_columns.extend(
                [int(node_columns[local_edge]), int(node_columns[local_edge + 1])]
            )
            incidence_values.extend([-1.0, 1.0])
        mapped_nodes = element * h + jacobian * (gll + 1.0)
        if element == 0:
            x0[: p + 1] = mapped_nodes
        else:
            x0[element * p + 1 : (element + 1) * p + 1] = mapped_nodes[1:]
        x1[edge_columns] = 0.5 * (mapped_nodes[:-1] + mapped_nodes[1:])

    b0 = coo_matrix((values0, (rows0, columns0)), shape=(qsize, n0)).tocsr()
    d0 = coo_matrix(
        (derivative0_values, (rows0, columns0)), shape=(qsize, n0)
    ).tocsr()
    b1 = coo_matrix((values1, (rows1, columns1)), shape=(qsize, n1)).tocsr()
    d1 = coo_matrix(
        (derivative1_values, (rows1, columns1)), shape=(qsize, n1)
    ).tocsr()
    incidence = coo_matrix(
        (incidence_values, (incidence_rows, incidence_columns)),
        shape=(n1, n0),
    ).tocsr()
    mass0 = (b0.T @ b0.multiply(w[:, None])).tocsr()
    mass1 = (b1.T @ b1.multiply(w[:, None])).tocsr()
    stiffness0 = (d0.T @ d0.multiply(w[:, None])).tocsr()
    stiffness1 = (d1.T @ d1.multiply(w[:, None])).tocsr()
    for interface in range(1, elements):
        jump = np.zeros(n1)
        average_derivative = np.zeros(n1)
        left = slice((interface - 1) * p, interface * p)
        right = slice(interface * p, (interface + 1) * p)
        jump[left] = trace_b1_reference[1] / jacobian
        jump[right] = -trace_b1_reference[0] / jacobian
        average_derivative[left] = 0.5 * trace_d1_reference[1] / jacobian**2
        average_derivative[right] = 0.5 * trace_d1_reference[0] / jacobian**2
        interface_form = -np.outer(average_derivative, jump)
        interface_form -= np.outer(jump, average_derivative)
        interface_form += gamma / h * np.outer(jump, jump)
        stiffness1 += csr_matrix(interface_form)

    trace0 = np.zeros((2, n0))
    trace1 = np.zeros((2, n1))
    normal0 = np.zeros((2, n0))
    normal1 = np.zeros((2, n1))
    trace0[0, : p + 1] = trace_b0[0]
    trace0[1, -p - 1 :] = trace_b0[1]
    trace1[0, :p] = trace_b1_reference[0] / jacobian
    trace1[1, -p:] = trace_b1_reference[1] / jacobian
    normal0[0, : p + 1] = -trace_d0_reference[0] / jacobian
    normal0[1, -p - 1 :] = trace_d0_reference[1] / jacobian
    normal1[0, :p] = -trace_d1_reference[0] / jacobian**2
    normal1[1, -p:] = trace_d1_reference[1] / jacobian**2

    return CoefficientAxis(
        npts=n0,
        degree=p,
        length=domain_length,
        knots0=np.linspace(0.0, domain_length, elements + 1),
        knots1=np.linspace(0.0, domain_length, elements + 1),
        x0=x0,
        x1=x1,
        q=q,
        w=w,
        b0=csr_matrix(b0),
        b1=csr_matrix(b1),
        d0=csr_matrix(d0),
        d1=csr_matrix(d1),
        mass0=csr_matrix(0.5 * (mass0 + mass0.T)),
        mass1=csr_matrix(0.5 * (mass1 + mass1.T)),
        stiffness0=csr_matrix(0.5 * (stiffness0 + stiffness0.T)),
        stiffness1=csr_matrix(0.5 * (stiffness1 + stiffness1.T)),
        incidence=incidence,
        trace0=trace0,
        trace1=trace1,
        normal0=normal0,
        normal1=normal1,
        greville0=identity(n0, format="csr"),
        greville1=identity(n1, format="csr"),
    )


class CompatibleSEMFlow2D(TensorCompatibleFlow2D):
    """Mixed compatible spectral-element flow on a uniform tensor mesh."""

    def __init__(
        self,
        degree: int,
        nelem: int,
        length: float = 1.0,
        penalty_gamma: float | None = None,
    ) -> None:
        axis = compatible_sem_axis(
            degree,
            nelem,
            length,
            penalty_gamma=penalty_gamma,
        )
        self.nelem = int(nelem)
        super().__init__(
            npts=axis.npts,
            degree=degree,
            length=length,
            penalty_gamma=penalty_gamma,
            coefficient_factors=axis,
            mesh_size=length / nelem,
        )

    def evaluation_factors(self, points: Array) -> tuple[csr_matrix, csr_matrix]:
        """Return element-local nodal and histopolant evaluation factors."""
        coordinates = np.asarray(points, dtype=float)
        if coordinates.ndim != 1:
            raise ValueError("evaluation points must be one-dimensional")
        if np.any(coordinates < 0.0) or np.any(coordinates > self.length):
            raise ValueError("evaluation points must lie in the domain")
        p = self.degree
        h = self.length / self.nelem
        jacobian = 0.5 * h
        primal_rows: list[int] = []
        primal_columns: list[int] = []
        primal_values: list[float] = []
        derivative_rows: list[int] = []
        derivative_columns: list[int] = []
        derivative_values: list[float] = []
        for row, coordinate in enumerate(coordinates):
            element = min(int(coordinate / h), self.nelem - 1)
            reference = 2.0 * (coordinate - element * h) / h - 1.0
            b0, b1, _d0, _d1 = _local_exact_sequence(
                p, np.array([reference])
            )
            node_columns = element * p + np.arange(p + 1)
            edge_columns = element * p + np.arange(p)
            primal_rows.extend([row] * (p + 1))
            primal_columns.extend(int(column) for column in node_columns)
            primal_values.extend(float(value) for value in b0[0])
            derivative_rows.extend([row] * p)
            derivative_columns.extend(int(column) for column in edge_columns)
            derivative_values.extend(float(value / jacobian) for value in b1[0])
        primal = coo_matrix(
            (primal_values, (primal_rows, primal_columns)),
            shape=(coordinates.size, self.axis.mass0.shape[0]),
        ).tocsr()
        derivative = coo_matrix(
            (derivative_values, (derivative_rows, derivative_columns)),
            shape=(coordinates.size, self.axis.derivative_size),
        ).tocsr()
        return primal, derivative

    def exact_sequence_residual(self) -> float:
        """Return the relative derivative factorization residual."""
        difference = self.axis.d0 - self.axis.b1 @ self.axis.incidence
        return float(np.linalg.norm(difference.toarray())) / max(
            float(np.linalg.norm(self.axis.d0.toarray())), 1.0e-300
        )
