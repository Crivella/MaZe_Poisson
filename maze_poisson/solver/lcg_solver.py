import numpy as np

from .. import c_api
from ..grid import LCGGrid, LCGGrid_MPI
from ..mpi import MPIBase
from . import BaseSolver

mpi = MPIBase()


class LCGSolver(BaseSolver):
    def initialize_grid(self):
        """Initialize the grid."""
        if mpi and mpi.size > 1:
            self.grid = LCGGrid_MPI(self.gset.L, self.gset.h, self.gset.N, self.mdv.tol)
        else:
            self.grid = LCGGrid(self.gset.L, self.gset.h, self.gset.N, self.mdv.tol)

    
    def compute_forces_field(self) -> np.ndarray:
        """Compute the forces from the field."""
        h = self.grid.h  # h in angstrom
        N = self.grid.N
        N_loc = self.grid.N_loc
        N_loc_start = self.grid.N_loc_start
        N_p = self.particles.N_p
        phi = self.grid.phi
        q = self.grid.q

        neighbors = self.particles.neighbors

        out = self.particles.forces_elec
        self.q_tot = c_api.c_compute_force_fd(N, N_p, h, phi, q, neighbors, out, N_loc, N_loc_start)

        return out

    # def _compute_forces_field_mpi(self):
    #     """Compute the forces from the field."""
    #     h = 2 * self.grid.h  # h in angstrom
    #     N = self.grid.N
    #     N_p = self.particles.N_p
    #     phi = self.grid.phi
    #     q = self.grid.q

    #     N_loc = self.grid.N_loc
    #     N_start = self.grid.N_loc_start
    #     N_end = self.grid.N_loc_end

    #     neighbors = self.particles.neighbors
    #     # print(neighbors.shape)
    #     # for i,neigh in enumerate(neighbors):
    #     #     for j,nei in enumerate(neigh):
    #     #         if nei[0] < N_start or nei[0] >= N_end:
    #     #             continue
    #     #             neighbors[i,j] = -1
    #     # w = np.where((neighbors[:,:,0] >= N_start) & (neighbors[:,:,0] < N_end))
    #     # part = w[0]
    #     # neighbors = neighbors[w]
    #     # neighbors[:,0] -= N_start
    #     # print(neighbors.shape)
    #     # q_neighbors = q[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]

    #     # X
    #     # print(phi.shape)
    #     tmp_top = self.grid.tmp_top
    #     tmp_bot = self.grid.tmp_bot
    #     mpi.send_previous(phi[0, :, :])
    #     mpi.send_next(phi[-1, :, :])
    #     mpi.recv_previous(tmp_bot)
    #     mpi.recv_next(tmp_top)
    #     E_x = np.zeros_like(phi)
    #     E_x[:-1] += phi[1:]
    #     E_x[-1] += tmp_top
    #     E_x[1:] -= phi[:-1]
    #     E_x[0] -= tmp_bot
    #     E_x /= h
    #     E_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / h
    #     E_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / h

    #     forces = self.particles.forces_elec
    #     forces.fill(0)
    #     q_tot = 0
    #     for i,neigh in enumerate(neighbors):
    #         for x,y,z in neigh:
    #             if x < N_start or x >= N_end:
    #                 continue
    #             s = x - N_start
    #             qn = q[s, y, z]
    #             q_tot += qn
    #             forces[i][0] -= qn * E_x[s, y, z]
    #             forces[i][1] -= qn * E_y[s, y, z]
    #             forces[i][2] -= qn * E_z[s, y, z]
        
    #     q_tot = mpi.comm.allreduce(q_tot)
    #     mpi.all_reduce_inplace(forces)


    # compute_forces_field = _compute_forces_field if not mpi or mpi.size == 1 else _compute_forces_field_mpi
