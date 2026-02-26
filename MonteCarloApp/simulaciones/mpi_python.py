import numpy as np
import time
import sys
import os

# ==============================
# PARAMETROS CLI
# ==============================

if len(sys.argv) < 6:
    print("Uso: python mpi_python.py filas columnas simulaciones T_MAX probBase")
    sys.exit(0)

filas = int(sys.argv[1])
columnas = int(sys.argv[2])
simulaciones = int(sys.argv[3])
T_MAX = int(sys.argv[4])
probBase = float(sys.argv[5])

# ==============================
# CONSTANTES
# ==============================

SANO = 0
FUEGO = 1
QUEMADO = 2

contador = np.zeros((filas, columnas))

# ==============================
# SIMULACION
# ==============================

inicio = time.time()

for s in range(simulaciones):

    estado = np.zeros((filas, columnas), dtype=int)

    fi = np.random.randint(filas)
    co = np.random.randint(columnas)
    estado[fi, co] = FUEGO

    factor = np.random.uniform(0.7, 1.3)

    for t in range(T_MAX):

        nuevo = estado.copy()

        for i in range(filas):
            for j in range(columnas):
                if estado[i, j] == FUEGO:

                    nuevo[i, j] = QUEMADO

                    vecinos = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]

                    for ni, nj in vecinos:
                        if 0 <= ni < filas and 0 <= nj < columnas:
                            if estado[ni, nj] == SANO:
                                p = probBase * factor
                                if p > 1:
                                    p = 1
                                if np.random.rand() < p:
                                    nuevo[ni, nj] = FUEGO

        estado = nuevo

    contador += (estado == QUEMADO)

# ==============================
# RESULTADOS
# ==============================

prob = contador / simulaciones

# crear carpeta resultados si no existe
os.makedirs("resultados", exist_ok=True)

# guardar csv
np.savetxt("resultados/mpi.csv", prob, delimiter=",")

# ==============================
# IMPRIMIR MATRIZ (IMPORTANTE)
# ==============================

for fila in prob:
    print(",".join(map(str, fila)))

# ==============================
# METRICAS
# ==============================

fin = time.time()
print("TIEMPO", fin - inicio)
print("MEMORIA", "NA")
print("HILOS", 1)