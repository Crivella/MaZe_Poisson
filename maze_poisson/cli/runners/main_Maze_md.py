##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...constants import a0, t_au, density
from ...grid import *
from ...loggers import logger
from ...restart import generate_restart
from ...verlet import (OVRVO_part1, OVRVO_part2, PrecondLinearConjGradPoisson, PrecondLinearConjGradPoisson_PB, VerletPB,
                       VerletPoisson, VerletSolutePart1, VerletSolutePart2)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_H_field_with_regions(grid, H_field,
                                       solute_color='indigo',
                                       bulk_color='lightgray',
                                       cmap_name='Purples',
                                       solute_threshold=0.999,
                                       bulk_threshold=0.001,
                                       max_points_bulk=200_000):
    """
    Plot H field with:
    - fixed color for solute
    - grayed-out bulk
    - gradient colormap for transition region

    Parameters:
        grid             : object with attribute `h`
        H_field          : 3D numpy array in [0, 1]
        solute_color     : matplotlib color for full solute (H ~ 1)
        bulk_color       : color for solvent/bulk (H ~ 0)
        cmap_name        : colormap for transition region (starts from solute_color)
        solute_threshold : H > this → solute
        bulk_threshold   : H < this → bulk
        max_points_bulk  : cap on number of bulk points
    """
    N = H_field.shape[0]
    h = grid.h
    coords = np.linspace(0, (N - 1) * h, N)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    H = H_field.flatten()

    # Masks
    mask_solute = H > solute_threshold
    mask_bulk = H < bulk_threshold
    mask_transition = (~mask_solute) & (~mask_bulk)

    # Subsample bulk
    if np.sum(mask_bulk) > max_points_bulk:
        idx_bulk = np.random.choice(np.where(mask_bulk)[0], size=max_points_bulk, replace=False)
    else:
        idx_bulk = np.where(mask_bulk)[0]

    # Plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Bulk: gray + transparent
    ax.scatter(x[idx_bulk], y[idx_bulk], z[idx_bulk],
               color=bulk_color, alpha=0.03, s=1, label='bulk')

    # Solute: fixed color
    ax.scatter(x[mask_solute], y[mask_solute], z[mask_solute],
               color=solute_color, alpha=1.0, s=10, label='solute')

    # Transition: gradient colormap
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize

    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=bulk_threshold, vmax=solute_threshold)
    H_trans = H[mask_transition]

    sc = ax.scatter(x[mask_transition], y[mask_transition], z[mask_transition],
                    c=H_trans, cmap=cmap, norm=norm, s=6, alpha=0.6, label='transition')

    # Axes and colorbar
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    ax.set_title('Solute Accessibility H')

    cbar = fig.colorbar(sc, ax=ax, label='H value (transition only)')
    # ax.legend()
    plt.tight_layout()
    plt.savefig("H_field_plot.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def main(grid_setting, output_settings, md_variables):
    begin_time = time.time()
    start_initialization = time.time()
    logger.info('--------Initialization begins---------')

    # get variables from input
    h = grid_setting.h
    L = grid_setting.L
    L_ang = grid_setting.L_ang
    N = grid_setting.N
    N_p = grid_setting.N_p
    h_ang = L_ang/N


    method = md_variables.method
    T = md_variables.T
    not_elec = md_variables.not_elec
    N_steps = md_variables.N_steps
    stride = md_variables.stride
    initialization = md_variables.initialization
    thermostat = md_variables.thermostat
    dt = md_variables.dt
    dt_fs = md_variables.dt_fs
    preconditioning = md_variables.preconditioning
    rescale = md_variables.rescale
    elec = md_variables.elec
    V = 27.211386245988
    tol = md_variables.tol
    iter_restart = output_settings.iter_restart
    non_polar = md_variables.non_polar

    # initialize grid by inserting particles in the system
    grid = Grid(grid_setting, md_variables, output_settings)

    ofiles = grid.output_files

    # log all the relevant info 
    logger.info(f'Simulation with N_p = {N_p}, N = {N} with N_steps = {N_steps} and tol = {md_variables.tol}')
    logger.info(f'Method: {method} \tIntegrator: {md_variables.integrator}')
    logger.info(f'Initialization is done with CG and preconditioning: {preconditioning}')
    logger.info(f'Parameters: h = {h_ang} A \ndt = {dt_fs} fs \nstride = {stride} \nL = {L_ang} A \ngamma = {md_variables.gamma}')
    logger.info(f'Charge assignment scheme: {grid_setting.cas}\n')
    logger.info(f'Potential: {md_variables.potential}')
    logger.info(f'Elec: {elec} \tNotElec: {not_elec} \tNonPolar: {non_polar}')
    logger.info(f'Temperature: {T} K \tDensity: {density} g/cm3.')
    logger.info(f'Print solute: {output_settings.print_solute} \tPrint field: {output_settings.print_field} \tPrint tot_force: {output_settings.print_tot_force}')
    logger.info(f'Print energy: {output_settings.print_energy} \tPrint temperature: {output_settings.print_temperature}')
    logger.info(f'Print performance: {output_settings.print_performance} \tRestart: {output_settings.restart}')
    logger.info(f'Thermostat: {thermostat} \tRescaling of the velocities: {rescale}')
    logger.info(f'Rescaling of the forces: {grid_setting.rescale_force}')
    logger.info(f'Transition region of width: {grid_setting.w} A')

    if 2 * grid_setting.w < h_ang and grid_setting.w > 0:
        logger.warning(f'Transition region double width {2 * grid_setting.w} A is smaller than the grid spacing {h_ang} A. This may lead to inaccuracies in the field calculation.')

    ################################ STEP 0 Verlet ##########################################
    #########################################################################################

    q_tot = 0
    #compute 8 (CIC) or 64(B-Spline) nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    q_tot = np.sum(grid.particles.charges)
    logger.info('Total charge q = '+ str(q_tot))

    # set charges with the weight function
    grid.SetCharges()
    
    # update dielectric and screening vectors 
    # grid.UpdateEpsAndK2()
    if method == 'PB MaZe':
        grid.UpdateEpsAndK2_transition()
    # plot_H_field_with_regions(grid, grid.H['center'])
    
    # initialize the electrostatic field with CG                  
    if preconditioning == "Yes":
        grid.phi_prev, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol)
        if method == 'PB MaZe':
            grid.phi_s_prev, _ = PrecondLinearConjGradPoisson_PB(- 4 * np.pi * grid.q / h, grid, tol=tol)

    if not_elec:
        grid.particles.ComputeForceNotElec()

    if elec:
        if method == 'Poisson MaZe':
            grid.particles.ComputeForce_FD(prev=True) 
        elif method == 'PB MaZe':
            grid.particles.ComputeForce_PB(prev=True)
    
    if non_polar and method == 'PB MaZe':
        grid.particles.ComputeNonpolarEnergyAndForces()
        

    ################################ STEP 1 Verlet ##########################################
    #########################################################################################

    # Velocity Verlet for the solute
    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part1(grid, thermostat = thermostat)
        #logger.info('Thermostat being applied')
    elif md_variables.integrator == 'VV':
        grid.particles = VerletSolutePart1(grid, thermostat=thermostat)
    elif md_variables.integrator == 'manual':
        grid.particles.pos[1,:] -= md_variables.delta
        #logger.info('Thermostat is not applied')

    # compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    #logger.info('Nearest neighbours calculated')

    # set charges with the weight function
    grid.SetCharges()
    #logger.info("Charges set with weight function")
    
    # update dielectric and screening vectors 
    # grid.UpdateEpsAndK2()
    if method == 'PB MaZe':
        grid.UpdateEpsAndK2_transition()

    if preconditioning == "Yes":
        grid.phi, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol, x0=grid.phi_prev)
        if method == 'PB MaZe':
            grid.phi_s, _ = PrecondLinearConjGradPoisson_PB(- 4 * np.pi * grid.q / h, grid, tol=tol, x0=grid.phi_s_prev)

    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part2(grid, thermostat = thermostat)
        #logger.info('OVRVO part 2 being run')
    elif md_variables.integrator == 'VV':
        grid = VerletSolutePart2(grid)
    elif md_variables.integrator == 'manual':
        if not_elec:
            grid.particles.ComputeForceNotElec()

        if elec:
            if method == 'Poisson MaZe':
                grid.particles.ComputeForce_FD(prev=True) 
            elif method == 'PB MaZe':
                grid.particles.ComputeForce_PB(prev=True)
        
        if non_polar and method == 'PB MaZe':
            grid.particles.ComputeNonpolarEnergyAndForces()
        

    # rescaling of the velocities to get total momentum = 0
    if rescale:
        logger.info('Rescaling of the velocities in progress...')
        grid.RescaleVelocities()

    ################################ FINE INIZIALIZZAZIONE ##########################################
    #########################################################################################

    X = np.arange(0, L, h)
    j = int(grid.particles.pos[0,1] / h)
    k = int(grid.particles.pos[0,2] / h)

    end_initialization = time.time()
    # print("\nInitialization time: {:.2f} s \n".format(end_initialization - start_initialization))
    logger.info('Initialization ends')
    logger.info(f'Initialization time: {end_initialization - start_initialization} s')

    if output_settings.restart == True and thermostat == False:
        init_steps = 0
    else:
        init_steps = md_variables.init_steps
        
    # print('Number of initialization steps:', init_steps,'\n')
    logger.info('Number of initialization steps '+str(init_steps))

    y = np.zeros_like(grid.q) 
    if method == 'PB MaZe':
        y_s = np.zeros_like(grid.q)

    ######################################### Verlet ############################################
    #############################################################################################

    counter = 0 
    
    # iterate over the number of steps (i.e times I move the particle 1)
    for i in tqdm(range(N_steps)):
        #print('\nStep = ', i, ' t elapsed from init =', time.time() - end_initialization)
        if md_variables.integrator == 'OVRVO':
            grid.particles = OVRVO_part1(grid, thermostat = thermostat)
        elif md_variables.integrator == 'VV':
            grid.particles = VerletSolutePart1(grid, thermostat = thermostat)
        elif md_variables.integrator == 'manual':
            grid.particles.pos[1,:] -= md_variables.delta

        if elec:
            # compute 8 nearest neighbors for any particle
            grid.particles.NearestNeighbors()
        
            # set charges with the weight function
            grid.SetCharges()
            
            # apply Verlet algorithm
            start_Verlet = time.time()
            grid, y, iter_conv = VerletPoisson(grid, y=y)
            end_Verlet = time.time()

            if method == 'PB MaZe':
                start_update = time.time()
                # grid.UpdateEpsAndK2()
                grid.UpdateEpsAndK2_transition()
                end_update = time.time()
                
                start_VerletPB = time.time()
                grid, y_s, iter_conv_PB = VerletPB(grid, y=y_s, tol=tol)
                end_VerletPB = time.time()

        if md_variables.integrator == 'OVRVO':
            grid.particles = OVRVO_part2(grid, thermostat = thermostat)
        elif md_variables.integrator == 'VV':
            grid = VerletSolutePart2(grid)
        elif md_variables.integrator == 'manual':
            if not_elec:
                grid.particles.ComputeForceNotElec()

            if elec:
                if method == 'Poisson MaZe':
                    grid.particles.ComputeForce_FD(prev=True) 
                elif method == 'PB MaZe':
                    grid.particles.ComputeForce_PB(prev=True)
            
            if non_polar and method == 'PB MaZe':
                grid.particles.ComputeNonpolarEnergyAndForces()

        if output_settings.print_tot_force:
            tot_force = np.zeros(3)
            tot_force = np.sum(grid.particles.forces + grid.particles.forces_notelec + grid.particles.forces_np, axis=0)
            
            ofiles.file_output_tot_force.write(str(i) + ',' + str(tot_force[0]) + ',' + str(tot_force[1]) + ','+ str(tot_force[2]) + "\n") 
            ofiles.file_output_tot_force.flush()

        grid.Energy(print_energy=output_settings.print_energy, iter=i) # prints the energy out
        grid.Temperature(print_temperature=output_settings.print_temperature, iter=i) # prints the temperature
        
        if output_settings.print_solute:
            df = pd.DataFrame(grid.particles.pos, columns=['x', 'y', 'z'])
            df['vx'] = grid.particles.vel[:, 0]
            df['vy'] = grid.particles.vel[:, 1]
            df['vz'] = grid.particles.vel[:, 2]
            df['fx_elec'] = grid.particles.forces[:, 0]
            df['fy_elec'] = grid.particles.forces[:, 1]
            df['fz_elec'] = grid.particles.forces[:, 2]
            df['charge'] = grid.particles.charges
            df['iter'] = i - init_steps
            df['particle'] = np.arange(N_p)
            df.to_csv(
                ofiles.file_output_solute, index=False, header=False, mode='a',
                columns=['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec']
                )
                
        if i % stride == 0 and i >= init_steps:
            if counter == 0 and thermostat == True:
                if np.abs(grid.temperature - 1550) <= 100:    
                    #print('End of thermostatting')
                    logger.info('End of thermostatting')
                    thermostat = False
                    counter = counter + 1
            if output_settings.print_performance and elec:
                if method == 'Poisson MaZe':
                    ofiles.file_output_performance.write(str(i - init_steps) + ',' + str(end_Verlet - start_Verlet) + ',' + str(iter_conv) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n"
                elif method == 'PB MaZe':
                    ofiles.file_output_performance.write(str(i - init_steps) + ',' + str(end_Verlet - start_Verlet) + ',' + str(iter_conv) + ',' + str(end_VerletPB - start_VerletPB) + ',' + str(iter_conv_PB) + ',' + str(end_update - start_update) + "\n")
            if output_settings.print_field and elec:
                field_x_MaZe = np.array([grid.phi[l, j, k] for l in range(N)])
                if method == 'Poisson MaZe':
                    for n in range(N):
                        ofiles.file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')
                elif method == 'PB MaZe':
                    field_x_MaZe_s = np.array([grid.phi_s[l, j, k] for l in range(N)])
                
                    for n in range(N):
                        ofiles.file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + ',' + str(field_x_MaZe_s[n] * V) + '\n')

    if output_settings.generate_restart_file:
        ofiles.file_output_solute.flush()
        restart_file = generate_restart(md_variables, grid_setting, output_settings, iter_restart)
        # print('Restart file generated: ', restart_file)
        logger.info(f'Restart file generated: {restart_file}')
        
    end_time = time.time()
    # print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))
    logger.info(f'Total time taken: {end_time - begin_time}')
    logger.info('--------------END RUN---------------------')
