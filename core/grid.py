import numpy as np

def generate_grid(extent: float, resolution: int):
    lin = np.linspace(-extent, extent, resolution)
    X, Y, Z = np.meshgrid(lin, lin, lin)
    R = np.sqrt(X**2 + Y**2 + Z**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Theta = np.arccos(Z / R)
        Theta[np.isnan(Theta)] = 0
    Phi = np.arctan2(Y, X)
    return X, Y, Z, R, Theta, Phi
