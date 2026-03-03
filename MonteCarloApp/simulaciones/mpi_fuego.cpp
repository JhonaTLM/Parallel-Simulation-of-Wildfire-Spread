#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <fstream>

using namespace std;

const int SANO = 0;
const int FUEGO = 1;
const int QUEMADO = 2;

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0)
            cout << "Uso: ./mpi filas columnas simulaciones T_MAX probBase\n";
        MPI_Finalize();
        return 0;
    }

    int filas = atoi(argv[1]);
    int columnas = atoi(argv[2]);
    int simulaciones = atoi(argv[3]);
    int T_MAX = atoi(argv[4]);
    double probBase = atof(argv[5]);

    int sim_local = simulaciones / size;
    if (rank == size - 1)
        sim_local += simulaciones % size;

    vector<double> contador_local(filas * columnas, 0.0);

    srand(time(NULL) + rank);

    auto inicio = chrono::high_resolution_clock::now();

    for (int s = 0; s < sim_local; s++) {

        vector<int> estado(filas * columnas, 0);

        int fi = rand() % filas;
        int co = rand() % columnas;
        estado[fi * columnas + co] = FUEGO;

        double factor = 0.7 + (double)rand() / RAND_MAX * (1.3 - 0.7);

        for (int t = 0; t < T_MAX; t++) {

            vector<int> nuevo = estado;

            for (int i = 0; i < filas; i++) {
                for (int j = 0; j < columnas; j++) {

                    int idx = i * columnas + j;

                    if (estado[idx] == FUEGO) {

                        nuevo[idx] = QUEMADO;

                        int vecinos[4][2] = {
                            {i-1,j},{i+1,j},{i,j-1},{i,j+1}
                        };

                        for (int v = 0; v < 4; v++) {

                            int ni = vecinos[v][0];
                            int nj = vecinos[v][1];

                            if (ni >= 0 && ni < filas && nj >= 0 && nj < columnas) {

                                int nidx = ni * columnas + nj;

                                if (estado[nidx] == SANO) {

                                    double p = probBase * factor;
                                    if (p > 1.0) p = 1.0;

                                    double r = (double)rand() / RAND_MAX;

                                    if (r < p)
                                        nuevo[nidx] = FUEGO;
                                }
                            }
                        }
                    }
                }
            }

            estado = nuevo;
        }

        for (int i = 0; i < filas * columnas; i++) {
            if (estado[i] == QUEMADO)
                contador_local[i] += 1.0;
        }
    }

    vector<double> contador_global(filas * columnas, 0.0);

    MPI_Reduce(
        contador_local.data(),
        contador_global.data(),
        filas * columnas,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    auto fin = chrono::high_resolution_clock::now();
    double tiempo = chrono::duration<double>(fin - inicio).count();

    if (rank == 0) {

        vector<double> prob(filas * columnas);

        for (int i = 0; i < filas * columnas; i++)
            prob[i] = contador_global[i] / simulaciones;

        // Crear carpeta resultados
        system("mkdir -p resultados");

        ofstream archivo("resultados/mpi.csv");

        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {

                double val = prob[i * columnas + j];

                cout << val;
                archivo << val;

                if (j < columnas - 1) {
                    cout << ",";
                    archivo << ",";
                }
            }
            cout << "\n";
            archivo << "\n";
        }

        archivo.close();

        cout << "TIEMPO " << tiempo << endl;
        cout << "MEMORIA NA" << endl;
        cout << "HILOS " << size << endl;
    }

    MPI_Finalize();
    return 0;
}