import numpy as np

class MICPreconditionerPB:
    def __init__(self, eps_x, eps_y, eps_z, k2, alpha=0.95, delta=0.2):
        """
        MIC preconditioner per Poisson-Boltzmann.
        eps_x, eps_y, eps_z: array (N,N,N)
        k2: array (N,N,N)
        """
        self.eps_x = eps_x
        self.eps_y = eps_y
        self.eps_z = eps_z
        self.k2 = k2

        self.N = eps_x.shape[0]
        self.size = self.N**3

        self.alpha = alpha
        self.delta = delta

        # Precompute diagonale modificata
        self.d = np.empty(self.size)
        self.compute_modified_diagonal()

    def index(self, ix, iy, iz):
        return (iz * self.N + iy) * self.N + ix

    def compute_modified_diagonal(self):
        N = self.N
        d = self.d

        # Flattened eps and k2 for fast access
        ex = self.eps_x
        ey = self.eps_y
        ez = self.eps_z
        k2 = self.k2

        for iz in range(N):
            for iy in range(N):
                for ix in range(N):
                    i = self.index(ix, iy, iz)

                    # Diagonal term
                    diag = (
                        ex[ix, iy, iz] + ex[(ix-1)%N, iy, iz] +
                        ey[ix, iy, iz] + ey[ix, (iy-1)%N, iz] +
                        ez[ix, iy, iz] + ez[ix, iy, (iz-1)%N] +
                        k2[ix, iy, iz]
                    )
                    aii = -diag

                    sum_term = 0.0

                    # Loop over the 6 neighbors
                    shifts_coeffs = [
                        ((-1,0,0), ex[(ix-1)%N, iy, iz]),
                        ((+1,0,0), ex[ix, iy, iz]),
                        ((0,-1,0), ey[ix, (iy-1)%N, iz]),
                        ((0,+1,0), ey[ix, iy, iz]),
                        ((0,0,-1), ez[ix, iy, (iz-1)%N]),
                        ((0,0,+1), ez[ix, iy, iz]),
                    ]

                    for shift, coeff in shifts_coeffs:
                        dx, dy, dz = shift
                        jx = (ix + dx) % N
                        jy = (iy + dy) % N
                        jz = (iz + dz) % N
                        j = self.index(jx, jy, jz)
                        if j < i:
                            sum_term += (coeff **2)/d[j]

                    d[i] = aii - self.alpha * sum_term + self.delta
                    if d[i]<=0:
                        raise ValueError(f"d[{i}] <=0")

    def apply(self, r):
        N = self.N
        d = self.d
        size = self.size

        y = np.empty(size)
        z = np.empty(size)

        # Flattened eps
        ex = self.eps_x
        ey = self.eps_y
        ez = self.eps_z

        # Forward substitution
        for iz in range(N):
            for iy in range(N):
                for ix in range(N):
                    i = self.index(ix, iy, iz)
                    s = r[i]
                    shifts_coeffs = [
                        ((-1,0,0), ex[(ix-1)%N, iy, iz]),
                        ((+1,0,0), ex[ix, iy, iz]),
                        ((0,-1,0), ey[ix, (iy-1)%N, iz]),
                        ((0,+1,0), ey[ix, iy, iz]),
                        ((0,0,-1), ez[ix, iy, (iz-1)%N]),
                        ((0,0,+1), ez[ix, iy, iz]),
                    ]
                    for shift, coeff in shifts_coeffs:
                        dx, dy, dz = shift
                        jx = (ix + dx) % N
                        jy = (iy + dy) % N
                        jz = (iz + dz) % N
                        j = self.index(jx, jy, jz)
                        if j < i:
                            s -= coeff * y[j]
                    y[i] = s / d[i]

        # Backward substitution
        for iz in reversed(range(N)):
            for iy in reversed(range(N)):
                for ix in reversed(range(N)):
                    i = self.index(ix, iy, iz)
                    s = y[i] / d[i]
                    shifts_coeffs = [
                        ((-1,0,0), ex[(ix-1)%N, iy, iz]),
                        ((+1,0,0), ex[ix, iy, iz]),
                        ((0,-1,0), ey[ix, (iy-1)%N, iz]),
                        ((0,+1,0), ey[ix, iy, iz]),
                        ((0,0,-1), ez[ix, iy, (iz-1)%N]),
                        ((0,0,+1), ez[ix, iy, iz]),
                    ]
                    for shift, coeff in shifts_coeffs:
                        dx, dy, dz = shift
                        jx = (ix + dx) % N
                        jy = (iy + dy) % N
                        jz = (iz + dz) % N
                        j = self.index(jx, jy, jz)
                        if j > i:
                            s -= coeff * z[j]
                    z[i] = s
        return z