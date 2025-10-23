import os

import numpy as np
import pandas as pd

from.constants import a0, density

def generate_restart(md_variables, grid_setting, output_settings, iter = None):
    thermostat = md_variables.thermostat
    N_p = grid_setting.N_p
    N = grid_setting.N
    path = output_settings.path
    restart_path = os.path.join('Outputs')
    if thermostat == True: 
        path = os.path.join(path, 'Thermostatted')
    filename = os.path.join(path, 'solute.csv')

    df = pd.read_csv(filename)
    m_Na = 22.99
    m_Cl = 35.453
    radius = np.ones(N_p)

    max = np.max(df['iter'])
    if iter == None:
        new_df = df[df['iter'] == max][['charge','x','y','z','vx','vy','vz']] 
    else:
        new_df = df[df['iter'] == iter][['charge','x','y','z','vx','vy','vz']] 

    col_mass_bool = new_df['charge'] == 1
    col_mass = [m_Na if bool == True else m_Cl for bool in col_mass_bool]

    new_df.insert(loc=1, column='mass',value=col_mass)
    print(np.shape(new_df))
    #new_df.insert(loc=2, column='radius',value=radius)
    new_df['x'] = new_df['x'] * a0
    new_df['y'] = new_df['y'] * a0
    new_df['z'] = new_df['z'] * a0

    print(new_df.head())

    if iter == None:
        filename_output = os.path.join(restart_path, 'restart.csv')
    else:
        filename_output = os.path.join(restart_path, 'restart.csv')

    new_df.to_csv(filename_output, index=False)

    return filename_output