import numpy as np


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
