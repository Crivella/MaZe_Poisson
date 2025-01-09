a0 = 0.529177210903 # in Angstrom units
t_au = 2.4188843265857 * 1e-2 # fs = 1 a.u. of time 

amu_to_kg = 1.66054 * 1e-27 # conversion 
m_e = 9.1093837 * 1e-31 # kg
conv_mass = amu_to_kg / m_e
kB = 3.1668 * 1e-6 #E_h/K

m_Na_amu = 22.99
m_Cl_amu = 35.453

Ha_to_eV = 27.21138602

m_Na = m_Na_amu * amu_to_kg
m_Cl = m_Cl_amu * amu_to_kg

# STUFF TO BE MADE INTO INPUTS FROM USER
density = 1.3793 # this is the new value, g/cm^3
ref_L=20.64
ref_N=100