"""Runtime construction of local polynomial Wendland smoothers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.chebyshev import chebval, chebvander


@dataclass(frozen=True)
class WendlandPolynomialFit:
    coefficients: np.ndarray
    degree: int
    relative_l2: float
    max_abs: float
    meets_tolerance: bool


def _even_ceiling(value: float) -> int:
    result = int(np.ceil(value))
    return result if result % 2 == 0 else result + 1


def _initial_degree(order: int, sigma_grid: float) -> int:
    if order == 2:
        return _even_ceiling(sigma_grid + 1.0)
    if order == 4:
        return _even_ceiling(0.75 * sigma_grid + 2.0)
    raise ValueError(f"Unsupported Wendland order {order}")


def _sampled_symbol(n: int, order: int, sigma_grid: float) -> np.ndarray:
    coordinates = np.arange(n, dtype=np.float64)
    coordinates = np.where(coordinates <= n // 2, coordinates, coordinates - n)
    squared = coordinates * coordinates
    radius = np.sqrt(
        squared[:, None, None] + squared[None, :, None] + squared[None, None, :]
    )
    t = radius / sigma_grid
    if order == 2:
        kernel = np.where(t <= 1.0, (1.0 - t) ** 4 * (4.0 * t + 1.0), 0.0)
    elif order == 4:
        kernel = np.where(
            t <= 1.0,
            (1.0 - t) ** 6 * (35.0 * t * t + 18.0 * t + 3.0),
            0.0,
        )
    else:
        raise ValueError(f"Unsupported Wendland order {order}")
    kernel /= kernel.sum()
    return np.fft.fftn(kernel).real


def _mehrstellen_symbol(n: int) -> np.ndarray:
    theta = 2.0 * np.pi * np.fft.fftfreq(n)
    cosine = np.cos(theta)
    cx = cosine[:, None, None]
    cy = cosine[None, :, None]
    cz = cosine[None, None, :]
    laplacian = -4.0 + (2.0 / 3.0) * (
        cx + cy + cz + cx * cy + cx * cz + cy * cz
    )
    return 1.0 + 3.0 * laplacian / 8.0


def generate_wendland_polynomial_fit(
    n: int,
    order: int,
    sigma_grid: float,
    relative_tolerance: float = 1.0e-3,
    max_abs_tolerance: float = 5.0e-4,
    max_degree: int = 30,
) -> WendlandPolynomialFit:
    """Fit the discrete Wendland symbol as a polynomial of the 19-point symbol."""
    if n < 4:
        raise ValueError("Polynomial Wendland smoothing requires N >= 4")
    if sigma_grid <= 0.0 or sigma_grid >= n / 2.0:
        raise ValueError(
            f"Wendland support sigma/h={sigma_grid:.8g} must lie in (0, N/2)"
        )
    if relative_tolerance <= 0.0 or max_abs_tolerance <= 0.0:
        raise ValueError("Polynomial fit tolerances must be positive")
    if max_degree < 4:
        raise ValueError("Polynomial maximum degree must be at least 4")

    first_degree = max(4, _initial_degree(order, sigma_grid) - 4)
    candidate_max = max_degree - max_degree % 2
    if first_degree > candidate_max:
        raise ValueError(
            f"No candidate degree is available below maximum degree {max_degree}"
        )

    x = _mehrstellen_symbol(n).ravel()
    y = _sampled_symbol(n, order, sigma_grid).ravel()

    # P(1)=1 is imposed by using columns T_j(x)-1, j>=1. Build the
    # normal equations once; every lower-degree candidate is a leading block.
    basis = chebvander(x, candidate_max)[:, 1:]
    basis -= 1.0
    gram = basis.T @ basis
    rhs = basis.T @ (y - 1.0)
    del basis

    target_norm = np.linalg.norm(y)
    best_fit: Optional[WendlandPolynomialFit] = None
    best_score = np.inf
    for degree in range(first_degree, candidate_max + 1, 2):
        tail = np.linalg.solve(gram[:degree, :degree], rhs[:degree])
        coefficients = np.empty(degree + 1, dtype=np.float64)
        coefficients[0] = 1.0 - tail.sum()
        coefficients[1:] = tail
        delta = chebval(x, coefficients) - y
        relative_l2 = float(np.linalg.norm(delta) / target_norm)
        max_abs = float(np.max(np.abs(delta)))
        score = max(
            relative_l2 / relative_tolerance,
            max_abs / max_abs_tolerance,
        )
        if score < best_score:
            best_score = score
            best_fit = WendlandPolynomialFit(
                coefficients=coefficients,
                degree=degree,
                relative_l2=relative_l2,
                max_abs=max_abs,
                meets_tolerance=False,
            )
        if score <= 1.0:
            return WendlandPolynomialFit(
                coefficients=coefficients,
                degree=degree,
                relative_l2=relative_l2,
                max_abs=max_abs,
                meets_tolerance=True,
            )

    assert best_fit is not None
    return best_fit
