import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ...constants import a0
from ...myio.loggers import logger

N_vector = [30,40,50,60,70,80,90,100, 110, 120] # will change from 10 to 100 once the CG works for 10,20
N_vector = np.array(N_vector)
N_p_vector = [128,250,432,686,1024,1458]

path = 'Outputs/'
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)

path_pdf = path + 'PDFs/'
isExist = os.path.exists(path_pdf)
if not isExist:
    os.makedirs(path_pdf)

def g(x,a,b):
    return a * x**b

def g1(x, a, b):
    return a * x + b

def f(x,a,b):
    return a * x**b * np.log(x**b)

def k(x, a, b):
    return a*x + b*x**2

# time-vs-n3:  plot of time/iter VS N_grid = N^3
def plot_time_iterNgrid(N_p):
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    filename_MaZe=path+'performance_N'
    data1 = "time" 
    data2 = 'n_iters'
    
    path_all_files = [(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]
    isExist = [os.path.exists(i) for i in path_all_files]
    if all(isExist) == False:
        logger.error(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
        raise FileNotFoundError(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
    elif all(isExist) == True:
        df_list_MaZe = [pd.read_csv(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]

    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][100:]/df[data2][100:]))
        sd1.append(np.std(df[data1][100:]/df[data2][100:]))
        
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_vector
    poptMaZe, _ = curve_fit(g1, x**3, avg1, sigma = sd1, absolute_sigma=True, p0=[0.1, 1.4])
    a_optMaZe, b_optMaZe = poptMaZe

    print(f'Optimized parameters MaZe t: a = {a_optMaZe}, b = {b_optMaZe}')

    plt.figure(figsize=(10, 8))
    plt.errorbar(x**3, avg1,sd1, label = 'MaZe', color='r',marker='o', linestyle='', linewidth=1.5, markersize=6,capsize=4)
    plt.plot(x**3, g1(x**3, a_optMaZe, b_optMaZe), label=f'fit $ax+b$, b = {b_optMaZe:.6f},  a = {a_optMaZe}')
    #plt.ylim(0, 0.02)
    #plt.xlim(20**3, 120**3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of grid points', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    title = 'time_per_iter_VS_N3_N_p_'
    plt.grid()
    name =  title +str(N_p)+ ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()


def f(x,a,b):
    return a * x + b

# iter-vs-n3: plot of n iterations VS N_grid = N^3
def plot_convNgrid(N_p):
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    filename_MaZe=path+'performance_N'
    data1 = "n_iters"

    path_all_files = [(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]
    isExist = [os.path.exists(i) for i in path_all_files]
    if all(isExist) == False:
        logger.error(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
        raise FileNotFoundError(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
    elif all(isExist) == True:
        df_list_MaZe = [pd.read_csv(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]

    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][100:]))
        sd1.append(np.std(df[data1][100:]))
        
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_vector**3
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(N_vector**3, avg1,sd1, label = 'MaZe', color='r',marker='o', linestyle='', markersize=6,capsize=4)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe),  label=f'fit $ax^b$, b = {b_optMaZe:.2f}  $\\approx 1/3$ a = {a_optMaZe:.2f}')
    plt.xlabel('Number of grid points', fontsize=18)
    plt.ylabel('# of iterations', fontsize=18)
    #plt.ylim(0,200)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    plt.grid()
    title='n_iterations_vs_N_grid_N_p_'
    name =  title +str(N_p)+ ".pdf"
    plt.savefig(path_pdf + name, format='pdf')
    plt.show()


#### PLOT FUNCTION OF Number of particles ####

# time-vs-np: plot time / n iterations VS number of particles
def plot_scaling_particles_time_iters():
    data1 = "time"
    data2 = "n_iters"
    filename_MaZe='performance_N'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    N_vector = [80, 100, 120, 140, 160, 180]
    N_p=N_p_vector
    N_p = np.array(N_p)
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) + '.csv') for i in N_vector]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]/df[data2][-50:]))
        sd1.append(np.std(df[data1][-50:]/df[data2][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_p
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True, p0=[1e-8,1])
    a_optMaZe, b_optMaZe = poptMaZe

    print(f'Optimized parameters MaZe: a = {a_optMaZe}, b = {b_optMaZe}')
    plt.figure(figsize=(10, 8))
   
    plt.errorbar(N_p, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe),  label=f'fit $ax^b$, b = {b_optMaZe:.2f}')
    plt.xlabel('Number of particles', fontsize=18)
    plt.ylabel('Time (s)', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    plt.grid()
    title='time_per_n_iterations_vs_N_p'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()


def f(x,a,b):
    return a * np.log(x)**b


# iter-vs-np: plot n iterations VS number of particles
def plot_scaling_particles_conv():
    filename_MaZe='performance_N'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'

    data1 = "n_iters"
    N_vector = [80, 100, 120, 140, 160, 180]
    N_p = np.array(N_p_vector)
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'.csv') for i in N_vector]
    avg1 = []
    sd1 = []


    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]))
        sd1.append(np.std(df[data1][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_p
    poptMaZe, _ = curve_fit(f, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    print(N_p_vector)
    print(sd1)
    plt.errorbar(N_p, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, f(x, a_optMaZe, b_optMaZe), label=f'fit $a\\log{{x}}^b$, b = {b_optMaZe:.2f}') 
    plt.xlabel('Number of particles', fontsize=18)
    plt.ylabel('# Iterations', fontsize=18)
    plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    plt.grid()
    title='iterations_vs_N_p'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()

# iter-vs-threads: plot number of iterations vs. number of threads
# SANITY CHECK GRAPH
def iter_vs_threads():
    filename_MaZe='performance_N100_N_p_250_'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'

    data1 = "n_iters"
    threads = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'.csv') for i in threads]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]))
        sd1.append(np.std(df[data1][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = threads
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(x, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe), label=f'fit $ax^b$, b = {b_optMaZe:.2f} a = {a_optMaZe:.2f}') 
    plt.xlabel('Number of threads', fontsize=18)
    plt.ylabel('# of iterations', fontsize=18)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    plt.grid()
    title='iterations_vs_threads'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()

# time-vs-threads: plot time vs. number of threads
def time_vs_threads():
    # plotting strong scaling and weak scaling as references
    filename_strong='performance_strong_'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'

    data1 = "time"
    threads = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    df_list_strong = [pd.read_csv(path+filename_strong + str(i) +'.csv') for i in threads]
    avg1 = []
    sd1 = []

    for df in df_list_strong:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = threads
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(x, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe), label=f'fit $ax^b$, b = {b_optMaZe:.2f} a = {a_optMaZe:.2f}') 
    plt.xlabel('Number of threads', fontsize=18)
    plt.ylabel('time (s)', fontsize=18)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    plt.grid()
    title='time_vs_threads'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()

def iter_vs_tol():
    filename_MaZe='performance_tol'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'

    data1 = "n_iters"
    tolerance = np.array([1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10])
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'.csv') for i in tolerance]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = tolerance
    poptMaZe, _ = curve_fit(g1, np.log10(x), avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(np.log10(x), avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(np.log10(x), g1(np.log10(x), a_optMaZe, b_optMaZe), label=f'fit $ax^b$, b = {b_optMaZe:.2f} a = {a_optMaZe:.2f}') 
    plt.xlabel(r'log$_{10}$(Tolerance)', fontsize=18)
    plt.ylabel('# of iterations', fontsize=18)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper right', fontsize=15)
    plt.grid()
    plt.title("N_p=250, N=100, dt=0.25")
    title='iterations_vs_tol'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()