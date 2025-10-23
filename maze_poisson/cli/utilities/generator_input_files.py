import numpy as np
import csv


def generate_bcc_positions(box_size, num_particles):
    """
    Generate BCC lattice positions within a 3D box, evenly distributed across the dimensions,
    ensuring that epsilon is dynamically computed from half the lattice spacing.
    """
    num_cells = int(np.ceil((num_particles / 2) ** (1 / 3)))
    lattice_constant = box_size / num_cells
    epsilon = lattice_constant / 4

    positions = []
    for x in range(num_cells):
        for y in range(num_cells):
            for z in range(num_cells):
                # Corner atom
                positions.append([
                    epsilon + x * lattice_constant,
                    epsilon + y * lattice_constant,
                    epsilon + z * lattice_constant
                ])
                # Body-centered atom
                positions.append([
                    epsilon + (x + 0.5) * lattice_constant,
                    epsilon + (y + 0.5) * lattice_constant,
                    epsilon + (z + 0.5) * lattice_constant
                ])

    positions = np.array(positions)

    if len(positions) > num_particles:
        positions = positions[:num_particles]

    return positions


# ---------------------------
# PARAMETRI DI GENERAZIONE
# ---------------------------

L = np.array([13.11])
num_particles = np.array([64])

folder = 'examples/input_files/'
old_input_files = True  # <--- change flag here to generate old/new format

m_Na = 22.99
m_Cl = 35.453
default_radius = 1.0  

# ---------------------------

for i in range(len(L)):
    positions = generate_bcc_positions(L[i], num_particles[i])
    filename = folder + f'input_coord{num_particles[i]}.csv'

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)

        if old_input_files:
            header = ["charge", "mass", "radius", "x", "y", "z"]
        else:
            header = ["type", "x", "y", "z"]
        writer.writerow(header)

        for idx, pos in enumerate(positions):
            charge = (-1) ** idx  # +1, -1, +1, ...
            if old_input_files:
                mass = m_Na if charge > 0 else m_Cl
                writer.writerow([charge, mass, default_radius, *pos])
            else:
                atom_type = "Na" if charge > 0 else "Cl"
                writer.writerow([atom_type, *pos])

    print(f"CSV file '{filename}' has been generated. (old_input_files={old_input_files})")
