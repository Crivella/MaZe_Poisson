import numpy as np
import pandas as pd

# === Parametri di input ===
Np = 8  # Numero totale di particelle (deve essere multiplo di 8)
L_ang = 31.6  # Lato del box in Ångström

# === Proprietà fisiche degli ioni ===
mass_Na = 22.989769  # amu
mass_Cl = 35.45      # amu
radius_Na = 0.95     # Å (raggio nudo tipico per Na⁺)
radius_Cl = 1.81     # Å (raggio nudo tipico per Cl⁻)
charge_Na = 1.0
charge_Cl = -1.0

# === Controllo e dimensionamento ===
n_cells = round((Np / 8) ** (1 / 3))
if 8 * n_cells**3 != Np:
    raise ValueError("Np deve essere un multiplo esatto di 8 per strutture FCC.")

# === Offset della cella unitaria FCC ===
fcc_offsets = np.array([
    [0, 0, 0],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5]
])

# === Generazione della struttura ===
positions = []
a = L_ang / n_cells  # lunghezza della cella unitaria in Å

for i in range(n_cells):
    for j in range(n_cells):
        for k in range(n_cells):
            origin = np.array([i, j, k]) * a
            for offset in fcc_offsets:
                pos_cl = origin + offset * a
                pos_na = origin + ((offset + 0.5) % 1.0) * a
                positions.append([charge_Cl, mass_Cl, radius_Cl, *pos_cl])
                positions.append([charge_Na, mass_Na, radius_Na, *pos_na])

# === Centra la struttura nel box ===
df = pd.DataFrame(positions, columns=["charge", "mass", "radius", "x", "y", "z"])
center = df[["x", "y", "z"]].mean()
shift = (L_ang / 2) - center
df[["x", "y", "z"]] += shift

# === Salvataggio ===
filename = f"../../../examples/input_files_pb/input_coord{Np}.csv"
df.to_csv(filename, index=False)
print(f"File salvato: {filename}")
