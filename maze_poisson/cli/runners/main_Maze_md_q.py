##### Computation of the electrostatic field with the Poisson-Boltzmann equation + MaZe ####
#####                        Federica Troni - 09 / 2023                                 ####

import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...c_api import init_rfft
from ...constants import a0, density, t_au
from ...grid import *
from ...loggers import logger
from ...restart import generate_restart
from ...verlet import (OVRVO_part1, OVRVO_part2, VerletPoisson_Q,
                       VerletSolutePart1, VerletSolutePart2)


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

    init_rfft(N)

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

    # initialize grid by inserting particles in the system
    grid = Grid(grid_setting, md_variables, output_settings)

    ofiles = grid.output_files

    # log all the relevant info 
    logger.info(f'Simulation with N_p = {N_p}, N = {N} with N_steps = {N_steps} and tol = {md_variables.tol}')
    logger.info(f'Initialization is done with CG and preconditioning: {preconditioning}')
    logger.info(f'Parameters: h = {h}')
    logger.info(f'Parameters: h [A] = {h_ang} A')
    logger.info(f'dt = {dt_fs} fs')
    logger.info(f'stride = {stride}')
    logger.info(f'L = {L_ang} A')
    logger.info(f'gamma = {md_variables.gamma}')
    logger.info(f'Potential: {md_variables.potential}')
    logger.info(f'Elec: {elec} \tNotElec: {not_elec}')
    logger.info(f'Temperature: {T} K \tDensity: {density} g/cm3.')
    logger.info(f'Print solute: {output_settings.print_solute} \tPrint field: {output_settings.print_field} \tPrint tot_force: {output_settings.print_tot_force}')
    logger.info(f'Print energy: {output_settings.print_energy} \tPrint temperature: {output_settings.print_temperature}')
    logger.info(f'Print performance: {output_settings.print_performance} \tRestart: {output_settings.restart}')
    logger.info(f'Thermostat: {thermostat} \tRescaling of the velocities: {rescale}')

    ################################ STEP 0 Verlet ##########################################
    #########################################################################################

    q_tot = 0
    #compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    q_tot = np.sum(grid.particles.charges)
    logger.info('Total charge q = '+str(q_tot))

    # set charges with the weight function
    grid.SetCharges()
    #logger.info('Charges are set')

    # initialize the electrostatic field with CG                  
    if preconditioning == "Yes":
        #logger.info('Preconditioning being done for elec field')
        grid.calculate_phi_q()
        # grid.phi_prev_q = np.copy(grid.qq)
        # grid.phi_prev_q = PrecondLinearConjGradPoisson_Q(- 4 * np.pi * grid.q / h**3, grid)
        # grid.phi_prev_q = PrecondLinearConjGradPoisson_Q(- 4 * np.pi * grid.q, grid)

    if not_elec:
        grid.particles.ComputeForceNotElec()
        #logger.info('Non_elec force being computed')

    if elec:
        grid.particles.ComputeForce_FD(prev=True)
        #logger.info('Elec force being computed')
    
    else:
        logger.error("There is an error here")
        raise ValueError("There is an error here")

    ################################ STEP 1 Verlet ##########################################
    #########################################################################################

    # Velocity Verlet for the solute
    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part1(grid, thermostat = thermostat)
        #logger.info('Thermostat being applied')
    else:
        grid.particles = VerletSolutePart1(grid, thermostat=thermostat)
        #logger.info('Thermostat is not applied')

    # compute 8 nearest neighbors for any particle
    grid.particles.NearestNeighbors()
    #logger.info('Nearest neighbours calculated')

    # set charges with the weight function
    grid.SetCharges()
    #logger.info("Charges set with weight function")
        
    if preconditioning == "Yes":
        grid.calculate_phi_q()
        # grid.phi_q = np.copy(grid.qq)
        # grid.phi_q = PrecondLinearConjGradPoisson_Q(- 4 * np.pi * grid.q / h**3, grid)
        # grid.phi_q = PrecondLinearConjGradPoisson_Q(- 4 * np.pi * grid.q, grid)

    if md_variables.integrator == 'OVRVO':
        grid.particles = OVRVO_part2(grid, thermostat = thermostat)
        #logger.info('OVRVO part 2 being run')
    else:
        grid = VerletSolutePart2(grid)

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

    ######################################### Verlet ############################################
    #############################################################################################

    counter = 0 

    # tot_fft = 0
  
    # iterate over the number of steps (i.e times I move the particle 1)
    for i in tqdm(range(N_steps)):
        #print('\nStep = ', i, ' t elapsed from init =', time.time() - end_initialization)
        if md_variables.integrator == 'OVRVO':
            grid.particles = OVRVO_part1(grid, thermostat = thermostat)
        else:
            grid.particles = VerletSolutePart1(grid, thermostat = thermostat)

        if elec:
            # compute 8 nearest neighbors for any particle
            grid.particles.NearestNeighbors()
        
            # set charges with the weight function
            grid.SetCharges()

            # apply Verlet algorithm
            # start_Verlet = time.time()
            VerletPoisson_Q(grid)
            #grid, y, iter_conv = VerletPoissonBerendsen(grid, y)
            # end_Verlet = time.time()
            # if iter_conv == -1:
            #     tot_fft += 1

        if md_variables.integrator == 'OVRVO':
            grid.particles = OVRVO_part2(grid, thermostat = thermostat)
        else:
            grid = VerletSolutePart2(grid)

        if output_settings.print_tot_force:
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
            # if output_settings.print_performance and elec:
            #     ofiles.file_output_performance.write(str(i - init_steps) + ',' + str(end_Verlet - start_Verlet) + ',' + str(iter_conv) + "\n") #+ ',' + str(end_Matrix - start_Matrix) + "\n"
                        
            # if output_settings.print_field and elec:
            #     field_x_MaZe = np.array([grid.phi[l, j, k] for l in range(N)])
            #     for n in range(N):
            #         ofiles.file_output_field.write(str(i - init_steps) + ',' + str(X[n] * a0) + ',' + str(field_x_MaZe[n] * V) + '\n')

    # print('Number of FFTs = ', tot_fft)

    if output_settings.generate_restart_file:
        ofiles.file_output_solute.flush()
        restart_file = generate_restart(md_variables, grid_setting, output_settings, iter_restart)
        # print('Restart file generated: ', restart_file)
        logger.info(f'Restart file generated: {restart_file}')
        
    end_time = time.time()
    # print('\nTotal time: {:.2f} s\n'.format(end_time - begin_time))
    logger.info(f'Total time taken: {end_time - begin_time}')
    logger.info('--------------END RUN---------------------')
