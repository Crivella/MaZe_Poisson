import os
from math import exp, tanh

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .c_api import c_conj_grad, c_laplace, rfft_3d
from .constants import kB
from .profiling import profile


def VerletSolutePart1(grid, thermostat=False):
    dt = grid.md_variables.dt
    N_p = grid.N_p
    L = grid.L

    if thermostat == True:
        mi_vi2 = grid.particles.masses * np.sum(grid.particles.vel**2, axis=1)
        T_exp = np.sum(mi_vi2) / (3 * N_p * kB)
        lambda_scaling = np.sqrt(grid.temperature / T_exp)

    particles = grid.particles

    if thermostat == True:
        particles.vel = particles.vel * lambda_scaling

    particles.vel = particles.vel + 0.5 * dt * ((particles.forces + particles.forces_notelec) / particles.masses[:, np.newaxis])
    particles.pos = particles.pos + particles.vel * dt 
    particles.pos = particles.pos % L  

    return particles


def VerletSolutePart2(grid, prev=False):
    not_elec = grid.not_elec
    elec = grid.elec
    particles = grid.particles
    dt = grid.md_variables.dt
    
    if not_elec:
        particles.ComputeForceNotElec()

    if elec:
        # particles.ComputeForce_FD_Q(prev=prev)
        particles.ComputeForce_FD(prev=prev)

    particles.vel = particles.vel + 0.5 * dt * (particles.forces + particles.forces_notelec) / particles.masses[:, np.newaxis]
    
    return grid

### OVRVO ###
     
### Solves the equations for the O-block ###
def O_block(N_p, v, m, gamma, dt, kBT):
    c1 = exp(-gamma*dt)
    rnd = np.random.multivariate_normal(
        mean=[0.0, 0.0, 0.0],
        cov=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        size=N_p
    )

    v_t_dt = np.sqrt(c1) * v + np.sqrt((1 - c1) * kBT / m[:, np.newaxis]) * rnd
    return v_t_dt

### Solves the equations for the V-block ###
def V_block(v, F, m, gamma, dt):
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2 /(gamma * dt) * tanh(0.5 * gamma * dt)) 
    v_t_dt = v + 0.5 * c2 * dt * F / m[:, np.newaxis] #F is the force
    return v_t_dt

### Solves the equations for the R-block ###
def R_block(x,v, gamma, dt, L):
    if gamma == 0:
        c2 = 1
    else:
        c2 = np.sqrt(2 /(gamma * dt) * tanh(0.5 * gamma * dt)) 

    x_t_dt = x + c2 * dt * v
    x_t_dt = x_t_dt % L  
    return x_t_dt

@profile
def OVRVO_part1(grid, thermostat=False):
    if thermostat:
        gamma_sim = grid.md_variables.gamma
    else:
        gamma_sim = 0

    dt = grid.md_variables.dt
    kBT = grid.md_variables.kBT
    L = grid.L
    N_p = grid.N_p

    grid.particles.vel = O_block(N_p, grid.particles.vel, grid.particles.masses, gamma_sim, dt, kBT)
    grid.particles.vel = V_block(grid.particles.vel, grid.particles.forces + grid.particles.forces_notelec, grid.particles.masses, gamma_sim, dt)
    grid.particles.pos = R_block(grid.particles.pos, grid.particles.vel, gamma_sim, dt, L)
    
    return grid.particles


def OVRVO_part2(grid, prev=False, thermostat=False):
    dt = grid.md_variables.dt
    kBT = grid.md_variables.kBT
    N_p = grid.N_p

    if thermostat:
        gamma_sim = grid.md_variables.gamma
    else:
        gamma_sim = 0
        
    if grid.not_elec:
        grid.particles.ComputeForceNotElec()

    if grid.elec:
        # grid.particles.ComputeForce_FD_Q(prev=prev)
        grid.particles.ComputeForce_FD(prev=prev)

    grid.particles.vel = V_block(grid.particles.vel, grid.particles.forces + grid.particles.forces_notelec, grid.particles.masses, gamma_sim, dt)
    grid.particles.vel = O_block(N_p, grid.particles.vel, grid.particles.masses, gamma_sim, dt, kBT)
    
    return grid.particles


@profile
def MatrixVectorProduct_C(v,):
    res = np.empty_like(v)
    c_laplace(v, res, v.shape[0])
    return res

def MatrixVectorProduct_manual(v):
    # N = int(v.shape[0]**(1/3) + 0.5)
    N = 100
    v = v.reshape((N,N,N))
    res = -6 * np.copy(v)

    res[1:,:,:] += v[:-1,:,:]
    res[:-1,:,:] += v[1:,:,:]
    res[-1,:,:] += v[0,:,:]
    res[0,:,:] += v[-1,:,:]

    res[:,1:,:] += v[:,:-1,:]
    res[:,:-1,:] += v[:,1:,:]
    res[:,-1,:] += v[:,0,:]
    res[:,0,:] += v[:,-1,:]

    res[:,:,1:] += v[:,:,:-1]
    res[:,:,:-1] += v[:,:,1:]
    res[:,:,-1] += v[:,:,0]
    res[:,:,0] += v[:,:,-1]

    return res

MatrixVectorProduct = MatrixVectorProduct_C

from . import c_api


def VerletPoisson_Q(grid):
    grid.calculate_phi_q()

# apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoisson(grid, y):
    ####################################################
    # Using Real FFT
    # c_api.init_rfft(grid.N)
    # c_api.rfft_solve(grid.N, grid.q, grid.ig2, grid.phi)
    # # grid.phi_prev = np.copy(grid.phi)
    # # grid.phi = np.fft.irfftn(np.fft.rfftn(grid.q) * grid.ig2, s=grid.q.shape)
    # # # if np.any(np.abs(grid.phi.imag) > 1E-8):
    # # #     raise ValueError('Imaginary part in phi')
    # # grid.phi = grid.phi.real
    # iter_conv = 1
    # y_new = 1
    ####################################################

    ####################################################
    # Using LCG only
    tol = grid.md_variables.tol
    h = grid.h
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = tmp

    # compute the constraint with the provisional value of the field phi
    matrixmult = MatrixVectorProduct(grid.phi)
    sigma_p = grid.q / h + matrixmult / (4 * np.pi) # M @ grid.phi for row-by-column product

    y_new, iter_conv = PrecondLinearConjGradPoisson(sigma_p, x0=y, tol=tol) #riduce di 1/3 il numero di iterazioni necessarie a convergere
    # scale the field with the constrained 'force' term
    grid.phi -= y_new * (4 * np.pi)

    # if grid.debug:
    #     matrixmult1 = MatrixVectorProduct(y_new)
    #     print('LCG precision     :',np.max(np.abs(matrixmult1 - sigma_p)))
        
    #     matrixmult2 = MatrixVectorProduct(grid.phi)
    #     sigma_p1 = grid.q / h + matrixmult2 / (4 * np.pi) # M @ grid.phi for row-by-column product
    
    #     print('max of constraint: ', np.max(np.abs(sigma_p1)),'\n')
    ####################################################

    return grid, y_new, iter_conv

def PrecondLinearConjGradPoisson_scipy(b, x0 = None, tol=1e-7):
    if x0 is not None:
        x0 = x0.flatten()
    func = lambda x: MatrixVectorProduct(x.reshape(b.shape)).flatten()
    x, i = cg(LinearOperator((b.size, b.size), matvec=func), b.flatten(), x0=x0, atol=tol)
    if i > 0:
        raise ValueError(f'Conjugate gradient did not converge {i} iterations')
    x = x.reshape(b.shape)
    return x, i

def PrecondLinearConjGradPoisson_C(b, x0 = None, tol=1e-7):
    x = np.empty_like(b)
    if x0 is None:
        x0 = np.zeros_like(b)
    i = c_conj_grad(b, x0, x, tol, b.shape[0])
    if i == -1:
        raise ValueError('Conjugate gradient did not converge')
    return x, i

def PrecondLinearConjGradPoisson_OLD(b, x0 = None, tol=1e-7):
    N_tot = b.size
    if x0 is None:
        x0 = np.zeros_like(b)
    P_inv = - 1 / 6
    x = np.copy(x0)
    r = MatrixVectorProduct(x) - b
    v = P_inv * r  
    p = -v
    
    r_new = np.ones_like(r)
    iter = 0

    while np.linalg.norm(r_new) > tol:
        iter = iter + 1
        if iter > N_tot:
            raise ValueError('Conjugate gradient did not converge')
        if iter % 100 == 0:
            print('iter=',iter, np.linalg.norm(r_new))
        Ap = MatrixVectorProduct(p) # A @ d for row-by-column product
        # r_dot_v = blas.ddot(r, v, N_tot)
        r_dot_v = np.sum(r * v)
        # r_dot_v = c_ddot(r, v, N_tot)

        # alpha = r_dot_v / blas.ddot(p, Ap, N_tot)
        alpha = r_dot_v / np.sum(p * Ap)
        # alpha = r_dot_v / c_ddot(p, Ap, N_tot)
        # x = blas.daxpy(p, x , N_tot, alpha)
        x = alpha * p + x
        # c_daxpy(p, x, x, alpha, N_tot)
        # c_daxpy2(p, x, alpha, N_tot)
 
        # r_new = blas.daxpy(Ap, r, N_tot, alpha)
        r_new = alpha * Ap + r
        # c_daxpy(Ap, r, r_new, alpha, N_tot)
        # c_daxpy2(Ap, r, alpha, N_tot)
        # r_new = r
        v_new =  P_inv * r_new

        # beta = blas.ddot(r_new, v_new, N_tot) / r_dot_v
        beta = np.sum(r_new * v_new) / r_dot_v
        # beta = c_ddot(r_new, v_new, N_tot) / r_dot_v

        # p = blas.daxpy(p, -v_new, N_tot, beta)
        p = beta * p - v_new
        # c_daxpy(p, -v_new, p, beta, N_tot)
        # v_new = -v_new
        # c_daxpy2(p, v_new, beta, N_tot)
        
        r = r_new
        v = v_new

    return x, iter

PrecondLinearConjGradPoisson = PrecondLinearConjGradPoisson_C
# PrecondLinearConjGradPoisson = PrecondLinearConjGradPoisson_scipy
# PrecondLinearConjGradPoisson = PrecondLinearConjGradPoisson_OLD

# def PrecondLinearConjGradPoisson_Q(b, grid):
#     """Solves \\nabla^2 x = b in reciprocal space"""
#     # qq = np.empty_like(grid.phi_q, dtype=np.complex128)
#     rfft_3d(grid.N, b, grid.qq)
#     # y = np.fft.rfftn(b)
#     return grid.qq * grid.mig2



# alternative function for matrix-vector product
def MatrixVectorProduct_7entries(M, v, index):
    v_matrix = v[index]
    result = np.einsum('ij,ij->i', M, v_matrix)
    return result

# apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE
def VerletPoissonBerendsen(grid,eta):
    raise
    h = grid.h
    tol = grid.md_variables.tol

    # compute provisional update for the field phi
    tmp = np.copy(grid.phi)
    grid.phi = 2 * grid.phi - grid.phi_prev
    grid.phi_prev = np.copy(tmp)

    # apply SHAKE
    const_inv = 1 / 42
    stop_iteration =  False
    iter = 0

    # compute the constraint with the provisional value of the field phi
    M_phi = MatrixVectorProduct(grid.phi)
    sigma_p = grid.q / h + M_phi / (4 * np.pi) # M @ grid.phi for row-by-column product

    while(stop_iteration == False):	
        iter = iter + 1
        delta_eta =  -(4 * np.pi)**2 * const_inv * sigma_p 
        eta = eta + delta_eta
        
        M_delta_eta = MatrixVectorProduct(delta_eta)
        grid.phi = grid.phi + M_delta_eta / (4 * np.pi) 
        
        M_phi = MatrixVectorProduct(grid.phi)
        sigma_p = grid.q / h + M_phi / (4 * np.pi) # M @ grid.phi for row-by-column product
                
        if grid.output_settings.print_iters:
            # from .output_md import OutputFiles
            grid.output_files.file_output_iters.write(str(iter) + ',' + str(np.max(np.abs(sigma_p))) + ',' + str(np.linalg.norm(np.abs(sigma_p))) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n")
             
        # if np.linalg.norm(sigma_p) < tol: # MAX OR NORM?
        if np.max(np.abs(sigma_p)) < tol :
            stop_iteration = True
    
    print('iter=',iter)

    if grid.debug:
        matrixmult2 = MatrixVectorProduct(grid.phi)
        sigma_p1 = grid.q / h + matrixmult2 / (4 * np.pi) # M @ grid.phi for row-by-column product
    
        print('max of constraint: ', np.max(np.abs(sigma_p1)))
        print('norm of constraint: ', np.linalg.norm(sigma_p),'\n')
    return grid, eta, iter
