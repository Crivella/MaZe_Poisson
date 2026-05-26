import re

rgx_N = re.compile(r'N(\d+)')
rgx_Np = re.compile(r'N_p_(\d+)')
rgx_input_Np = re.compile(r'coord(\d+)')

def get_N(filename):
    return int(rgx_N.search(filename).group(1))

def get_Np(filename):
    return int(rgx_Np.search(filename).group(1))

def get_Np_input(filename):
    return int(rgx_input_Np.search(filename).group(1))