"""
utils.py

Common helper functions: Lévy flight, chaotic reinitialization, spiral position update, etc.
"""
import numpy as np
from scipy.special import gamma
from typing import Tuple


def levy_flight(beta: float, dim: int) -> np.ndarray:
    """
    Generates a Lévy flight step with the given beta parameter.

    Args:
        beta: Lévy exponential distribution parameter.
        dim: Number of dimensions.

    Returns:
        A Lévy flight step vector.
    """
    sigma = (
        (gamma(1 + beta) * np.sin(np.pi * beta / 2))
        / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def chaos_reinit(bounds: np.ndarray, prob: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a chaotic reinitialization (random reset) position and mask.

    Args:
        bounds: (dim, 2) shaped array for lower and upper bounds.
        prob: Probability of chaotic reinitialization for each dimension.

    Returns:
        chaos: New chaotic position vector.
        mask: Boolean vector indicating which dimensions are reset chaotically.
    """
    dim = bounds.shape[0]
    chaos = np.random.uniform(bounds[:, 0], bounds[:, 1], dim)
    mask = np.random.rand(dim) < prob
    return chaos, mask


def spiral_position(current: np.ndarray, best: np.ndarray, l: np.ndarray, factor: float = 0.5) -> np.ndarray:
    """
    Performs a spiral position update (based on Whale Optimization Algorithm).

    Args:
        current: Current position vector.
        best: Leader (alpha) position vector.
        l: Random vector in range [-1, 1].
        factor: Exponential coefficient (0.5 in WOA; can differ in other algorithms like MFO).

    Returns:
        Updated position vector.
    """
    return np.abs(best - current) * np.exp(factor * l) * np.cos(2 * np.pi * l) + best
