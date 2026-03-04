import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("probabilidades.csv", delimiter=",")

plt.figure(figsize=(8,6))
plt.imshow(data, cmap="inferno", origin="lower", vmin=0, vmax=1)
plt.colorbar(label="Probabilidad de Incendio")
plt.title("Mapa de Probabilidad de Incendio - Monte Carlo")
plt.xlabel("Columnas")
plt.ylabel("Filas")
plt.tight_layout()
plt.show()
