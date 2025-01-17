""""Fallback implementations of the force computation functions."""
import numpy as np


def compute_force_fd(
        N: int, N_p: int, h: float,
        phi: np.ndarray, q: np.ndarray, neighbors: np.ndarray,
        forces: np.ndarray,
    ) -> float:
    """Compute the forces from the field using finite differences.

    Args:
        N (int): Grid size. (Not used only to match the signature)
        N_p (int): Number of particles. (Not used only to match the signature)
        h (float): Grid spacing.
        phi_v (np.ndarray): Electrostatic potential. Shape (N, N, N).
        q (np.ndarray): 3D Charge. Shape (N, N, N).
        neighbors (np.ndarray): Indices of the neighbors. Shape (N_p, 8, 3).
        forces (np.ndarray): Output forces. Shape (N_p, 3).

    Returns:
        float: Total charge contribution.
    """
    h *= 2  # Double the grid spacing
    q_neighbors = q[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]
    for axis in range(3):
        E_ax = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / h
        E_neighbors = E_ax[neighbors[:, :, 0], neighbors[:, :, 1], neighbors[:, :, 2]]
        forces[:, axis] = -np.sum(q_neighbors * E_neighbors, axis=1)

    q_tot = np.sum(q_neighbors)
    return q_tot


def compute_tf_forces(
        N_p: int, L: float, pos: np.ndarray, B: float, params: np.ndarray, r_cut: float,
        forces: np.ndarray
    ) -> float:
    """Compute the forces from the field using finite differences.

    Args:
        N_p (int): Number of particles.
        L (float): Grid size.
        pos (np.ndarray): Particle positions. Shape (N_p, 3).
        B (float): B parameter of TF forces.
        params (np.ndarray): TF parameters per particle pair [A,C,D,sigma,alpha,beta]. Shape (6, N_p, N_p).
        r_cut (float): Cutoff radius.
        forces (np.ndarray): Output forces. Shape (N_p, 3).

    Returns:
        float: TF potential
    """
    r_diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # Shape: (N_p, N_p, 3)

    # Compute pairwise distances and unit vectors
    r_diff -= L * np.rint(r_diff / L)  # Apply periodic boundary conditions
    r_mag = np.linalg.norm(r_diff, axis=2)  # Shape: (N_p, N_p)
    np.fill_diagonal(r_mag, np.inf)  # Avoid self-interaction by setting diagonal to infinity

    r_cap = np.divide(r_diff, r_mag[:, :, np.newaxis], where=r_mag[:, :, np.newaxis] != 0)  # Avoid division by zero

    # Unpack parameters
    A, C, D, sigma, alpha, beta = params

    # Apply cutoff mask
    within_cutoff = r_mag <= r_cut

    # Compute force magnitudes and potentials only for particles within the cutoff
    f_mag = np.where(within_cutoff, B * A * np.exp(B * (sigma - r_mag)) - 6 * C / r_mag**7 - 8 * D / r_mag**9 - alpha, 0)
    V_mag = np.where(within_cutoff, A * np.exp(B * (sigma - r_mag)) - C / r_mag**6 - D / r_mag**8 + alpha * r_mag + beta, 0) #- V_shift

    # Compute pairwise forces, now with the shift applied to the force magnitudes
    pairwise_forces = f_mag[:, :, np.newaxis] * r_cap  # Shape: (N_p, N_p, 3)

    # Sum forces to get the net force on each particle
    net_forces = np.sum(pairwise_forces, axis=1)  # Sum forces acting on each particle (ignore NaN values)

    # Update the instance variables
    forces[:] = net_forces  # Store net forces in the output array
    potential_energy = np.sum(V_mag) / 2  # Avoid double-counting for potential energy

    return potential_energy
