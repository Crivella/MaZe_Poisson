import numpy as np
import logging

class LinkedCell:

    def __init__(self, r_cutoff, box_len, num_particles):
        self.side_cell_num = int(np.floor(box_len/r_cutoff)) # Number of cells per side
        #if self.side_cell_num<=2:
        #    logging.error(f"The linked cell method in your system has two or less cells per side. They should be al least three. Reduce the cut-off radius or increase the number of side copies.")
        #    exit()
        self.cell_len = box_len/self.side_cell_num
        self.relative_size = self.side_cell_num/box_len
        self.num_particles = num_particles
        self.header = None  # List with higher particle index in cell [X,Y,Z] (See book of J. M. Thijssen p. 204)
        self.link = None
        logging.info(f"Linked-cell created with {self.side_cell_num} cells per side with length {self.cell_len}")

    def update_lists(self, particles):  
        self.header = np.ones((self.side_cell_num, self.side_cell_num, self.side_cell_num ), dtype=int)*-1
        self.link = np.empty(self.num_particles)
        for i, p in enumerate(particles):
            IX = int(self.relative_size * p.pos[0])
            IY = int(self.relative_size * p.pos[1])
            IZ = int(self.relative_size * p.pos[2])
            self.link[i] = self.header[IX,IY,IZ]
            self.header[IX,IY,IZ] = i

    def interacting_particles(self ,IX, IY, IZ):
        central_cell_particles = np.array([], dtype=int)
        particle_index = self.header[IX, IY, IZ]
        # Interacting particles in the central cell
        while (particle_index >= 0):
            central_cell_particles = np.append(central_cell_particles,particle_index)
            particle_index = self.link[int(particle_index)]
        # Interacting particles in half the neighbors
        neighbor_cells_particles = np.array([], dtype=int)
        neighbor_indices = self.cell_neighbors(IX, IY, IZ)
        for cell_indices in neighbor_indices:
            particle_index = self.header[tuple(cell_indices)]
            while (particle_index >= 0):
                neighbor_cells_particles = np.append(neighbor_cells_particles,particle_index)
                particle_index = self.link[int(particle_index)]
        return central_cell_particles.astype(int), neighbor_cells_particles.astype(int)


    def cell_neighbors(self, IX, IY, IZ):  # Only half of the neighbors #Working for side_cells_num>=3 (?)
        num_neighbors = 13
        central_cell_index = np.array([IX,IY,IZ])
        displacements = np.array([[-1,0,0],[-1,0,1],[-1,-1,0],[-1,1,0],[-1,-1,1],[-1,1,1],
                         [1,0,1],[1,1,1],[1,-1,1],[0,-1,0],[0,-1,1],[0,1,1],[0,0,1]])   # Hardcoded neighbors
        cell_neighbors = np.zeros((num_neighbors,3),dtype=int)  # Hardcoded dimension
        for i in range(num_neighbors):
            cell_neighbors[i] = (central_cell_index+displacements[i])% self.side_cell_num
        return cell_neighbors