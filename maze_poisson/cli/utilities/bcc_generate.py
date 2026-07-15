import numpy as np
import csv


def generate_bcc_positions(box_size, num_particles):
    """
    Generate BCC lattice positions within a 3D box, evenly distributed across the dimensions,
    ensuring that epsilon is dynamically computed from half the lattice spacing.

    Parameters:
        box_size (float): Size of the cubic box along one dimension (assuming a cube).
        num_particles (int): Number of particles to place in the box.

    Returns:
        positions (np.ndarray): Array of shape (N, 3) containing the positions of the particles.
    """
    num_cells = int(np.ceil((num_particles / 2) ** (1 / 3)))
    lattice_constant = box_size / num_cells
    epsilon = lattice_constant / 4

    positions = []
    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                # Corner atom
                positions.append([epsilon + x * lattice_constant,
                                  epsilon + y * lattice_constant,
                                  epsilon + z * lattice_constant])
                # Body-centered atom
                positions.append([epsilon + (x + 0.5) * lattice_constant,
                                  epsilon + (y + 0.5) * lattice_constant,
                                  epsilon + (z + 0.5) * lattice_constant])

    positions = np.array(positions)

    if len(positions) > num_particles:
        positions = positions[:num_particles]

    return positions


# Dimensioni delle scatole
# L = [16.51, 20.64, 24.77, 28.90, 33.02, 37.15, 41.28, 45.20, 49.54, 53.66]
# L = np.array(L)
# L = np.array([61.92, 70.18, 78.43, 86.69, 99.07])
# L = np.array([103.20])

# # Numero di particelle
# # num_particles = np.array([128, 250, 432, 686, 1024, 1458, 2000, 2626, 3456, 4394])
# # num_particles = np.array([6750, 9826, 13718, 18522, 27648])
# num_particles = np.array([31250])



# folder = 'input_files_scaling_density/'

# header = ["type", "x", "y", "z"]

# for i in range(len(L)):
#     positions = generate_bcc_positions(L[i], num_particles[i])

#     filename = folder + f'input_coord{num_particles[i]}.csv'

#     with open(filename, "w", newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)

#         idx = 0
#         for pos in positions:
#             charge = (-1) ** idx  # +1, -1, +1, ...
#             atom_type = "Na" if charge > 0 else "Cl"
#             writer.writerow([atom_type, pos[0], pos[1], pos[2]])
#             idx += 1

#     print(f"CSV file '{filename}' has been generated.")