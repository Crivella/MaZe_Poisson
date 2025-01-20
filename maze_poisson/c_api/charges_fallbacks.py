import numpy as np

# from .mympi import c_get_n_loc, c_get_n_start


def g(x, L, h):
    x = np.abs(x - L * np.rint(x / L))
    return np.where(x < h, 1 - x / h, 0)

def update_charges(
        n_grid: int, n_p: int, h: float,
        pos: np.ndarray, neighbors: np.ndarray, charges: np.ndarray, q: np.ndarray
    ) -> float:
    """Update the charges on the grid.

    Args:
        n_grid (int): The size of the grid.
        n_p (int): The number of particles.
        h (float): The grid spacing.
        pos (np.ndarray): Particle positions. shape (N_p, 3)
        neighbors (np.ndarray): Particle neighbor indexes. shape (N_p, 8, 3)
        charges (np.ndarray): Particle charges. shape (N_p,)
        q (np.ndarray): Output charge grid. shape (n_grid, n_grid, n_grid)

    Returns:
        float: Total charge contribution.
    """
    n_loc = n_grid
    n_start = 0

    L = n_grid * h
    q.fill(0)
    diff = pos[:, np.newaxis, :] - neighbors * h  # shape (N_p, 8, 3)

    indices = tuple(neighbors.reshape(-1, 3).T)

    updates = (charges[:, np.newaxis] * np.prod(g(diff, L, h), axis=2)).flatten()
    for i,j,k,upd in zip(*indices, updates):
        i -= n_start
        if 0 <= i < n_loc:
            q[i, j, k] += upd

    q_tot = np.sum(updates)

    return q_tot