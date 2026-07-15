#ifndef __MP_CONSTANTS_H
#define __MP_CONSTANTS_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define a0 0.529177210903
#define a0_6 0.021958708714088133  // a0^6
#define a0_8 0.006149064714154663  // a0^8


#define kcalmol_to_ha 0.001593601  // 1 kcal/mol in Hartree
#define deg_to_rad (M_PI / 180.0)
#define kB 3.1668e-6  // Boltzmann constant in Hartree/Kelvin

// SPC/fw water model constants
#define kb (1059.162 * kcalmol_to_ha * a0 * a0)  // Convert kcal/mol/Å^2 to au
#define r0 (1.012 / a0)  // a0
#define ka (75.90 * kcalmol_to_ha)  // Hartree / rad^2
#define theta0 (113.24 * deg_to_rad)  // rad


#define t_au 2.4188843265857e-2  // fs = 1 a.u. of time
#define amu_to_kg 1.66054e-27  // conversion
#define m_e 9.10938356e-31  // kg
#define conv_mass 1822.8895391907286  // amu_to_kg / m_e

#define Ha_to_eV 27.21138602  // Hartree to eV

#endif // __MP_CONSTANTS_H