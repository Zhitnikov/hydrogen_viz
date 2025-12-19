import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial

def radial_wavefunction(n: int, l: int, r: np.ndarray, a0: float = 1.0) -> np.ndarray:
    rho = 2 * r / (n * a0)
    prefactor = np.sqrt(
        (2 / (n * a0))**3 * factorial(n - l - 1) / (2 * n * factorial(n + l))
    )
    laguerre_poly = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return prefactor * np.exp(-rho / 2) * (rho ** l) * laguerre_poly

def angular_wavefunction(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return sph_harm(m, l, phi, theta)

def probability_density(n: int, l: int, m: int, r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    R = radial_wavefunction(n, l, r)
    Y = angular_wavefunction(l, m, theta, phi)
    psi = R * Y
    return np.abs(psi)**2
