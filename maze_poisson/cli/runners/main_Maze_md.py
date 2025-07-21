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
from ...verlet import (OVRVO_part1, OVRVO_part2, PrecondLinearConjGradPoisson, PrecondLinearConjGradPoisson_PB, PrecondLinearConjGradPoisson_PB_Jacobi, VerletPB,
                       VerletPoisson, VerletSolutePart1, VerletSolutePart2)
# from ...miccg import MICPreconditionerPB

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
    # grid.SetCharges(uniform=True)
    grid.SetCharges(uniform=False)
    
    # update dielectric and screening vectors 
    # grid.UpdateEpsAndK2()
    if method == 'PB MaZe':
        grid.UpdateEpsAndK2_transition()
    
    # initialize the electrostatic field with CG                  
    if preconditioning == "Yes":
        if method == 'Poisson MaZe':
            grid.phi_prev, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol)
        elif method == 'PB MaZe':
            grid.phi_s_prev, _ = PrecondLinearConjGradPoisson_PB_Jacobi(- 4 * np.pi * grid.q / h, grid, tol=tol)

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
        # grid.particles.pos[0,:] += md_variables.delta
        grid.particles.pos[1,:] -= md_variables.delta
        # grid.particles.pos[1,:] += md_variables.delta

    # compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()

    # set charges with the weight function
    # grid.SetCharges(uniform=True)
    grid.SetCharges(uniform=False)
    
    # update dielectric and screening vectors 
    # grid.UpdateEpsAndK2()
    if method == 'PB MaZe':
        grid.UpdateEpsAndK2_transition()

    if preconditioning == "Yes":
        if method == 'Poisson MaZe':
            grid.phi, _ = PrecondLinearConjGradPoisson(- 4 * np.pi * grid.q / h, tol=tol, x0=grid.phi_prev)
        elif method == 'PB MaZe':
            grid.phi_s, _ = PrecondLinearConjGradPoisson_PB_Jacobi(- 4 * np.pi * grid.q / h, grid, tol=tol, x0=grid.phi_s_prev)

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
                grid.particles.ComputeForce_FD() 
            elif method == 'PB MaZe':
                grid.particles.ComputeForce_PB()
        
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

    if method == 'Poisson MaZe':
        y = np.zeros_like(grid.q) 
    elif method == 'PB MaZe':
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
            # grid.particles.pos[0,:] += md_variables.delta
            grid.particles.pos[1,:] -= md_variables.delta
            # grid.particles.pos[1,:] += md_variables.delta

        if elec:
            # compute 8 nearest neighbors for any particle
            grid.particles.NearestNeighbors()
        
            # set charges with the weight function
            # grid.SetCharges(uniform=True)
            grid.SetCharges(uniform=False)
            
            if method == 'Poisson MaZe':
                # apply Verlet algorithm
                start_Verlet = time.time()
                grid, y, iter_conv = VerletPoisson(grid, y=y)
                end_Verlet = time.time()

            elif method == 'PB MaZe':
                start_update = time.time()
                # grid.UpdateEpsAndK2()
                grid.UpdateEpsAndK2_transition()
                end_update = time.time()
                
                # precond = MICPreconditionerPB(grid.eps_x, grid.eps_y, grid.eps_z, grid.k2)
                start_VerletPB = time.time()
                # grid, y_s, iter_conv_PB = VerletPB(grid, precond, y=y_s, tol=tol)
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
                    grid.particles.ComputeForce_FD() 
                elif method == 'PB MaZe':
                    grid.particles.ComputeForce_PB()
            
            if non_polar and method == 'PB MaZe':
                grid.particles.ComputeNonpolarEnergyAndForces()

        if output_settings.print_tot_force:
            tot_force = np.zeros(3)
            tot_force = np.sum(grid.particles.forces + grid.particles.forces_notelec, axis=0)
            
            ofiles.file_output_tot_force.write(str(i) + ',' + str(tot_force[0]) + ',' + str(tot_force[1]) + ','+ str(tot_force[2]) + "\n") 
            ofiles.file_output_tot_force.flush()

        grid.Energy(print_energy=output_settings.print_energy, iter=i) # prints the energy out
        grid.Temperature(print_temperature=output_settings.print_temperature, iter=i) # prints the temperature
        
        if output_settings.print_solute:
            df = pd.DataFrame(grid.particles.pos, columns=['x', 'y', 'z'])
            df['vx'] = grid.particles.vel[:, 0]
            df['vy'] = grid.particles.vel[:, 1]
            df['vz'] = grid.particles.vel[:, 2]
            df['charge'] = grid.particles.charges
            df['iter'] = i - init_steps
            df['particle'] = np.arange(N_p)
            df.to_csv(
                ofiles.file_output_solute, index=False, header=False, mode='a',
                columns=['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz']
                )
        
        if output_settings.print_components_force and method == 'PB MaZe':
            df = pd.DataFrame(grid.particles.forces_rf, columns=['fx_RF', 'fy_RF', 'fz_RF'])
            df['fx_DB'] = grid.particles.forces_db[:, 0]
            df['fy_DB'] = grid.particles.forces_db[:, 1]
            df['fz_DB'] = grid.particles.forces_db[:, 2]
            df['fx_IB'] = grid.particles.forces_ib[:, 0]
            df['fy_IB'] = grid.particles.forces_ib[:, 1]
            df['fz_IB'] = grid.particles.forces_ib[:, 2]
            df['fx_NP'] = grid.particles.forces_np[:, 0]
            df['fy_NP'] = grid.particles.forces_np[:, 1]
            df['fz_NP'] = grid.particles.forces_np[:, 2]
            df['iter'] = i - init_steps
            df['particle'] = np.arange(N_p)
            df.to_csv(
                ofiles.file_output_force, index=False, header=False, mode='a',
                columns=['iter', 'particle',
                          'fx_RF', 'fy_RF', 'fz_RF', 
                          'fx_DB', 'fy_DB', 'fz_DB',
                          'fx_IB', 'fy_IB', 'fz_IB',
                          'fx_NP', 'fy_NP', 'fz_NP'
                          ]
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
                if method == 'Poisson MaZe':
                    field_x_MaZe = np.array([grid.phi[l, j, k] for l in range(N)])
                    
                    for n in range(N):
                        ofiles.file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')
                elif method == 'PB MaZe':
                    field_x_MaZe_s = np.array([grid.phi_s[l, j, k] for l in range(N)])
                
                    for n in range(N):
                        ofiles.file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe_s[n] * V) + ',' + str(field_x_MaZe_s[n] * V) + '\n')

    if output_settings.generate_restart_file:
        ofiles.file_output_solute.flush()
        restart_file = generate_restart(md_variables, grid_setting, output_settings, iter_restart)
        # print('Restart file generated: ', restart_file)
        logger.info(f'Restart file generated: {restart_file}')
        
    end_time = time.time()
    # print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))
    logger.info(f'Total time taken: {end_time - begin_time}')
    logger.info('--------------END RUN---------------------')
