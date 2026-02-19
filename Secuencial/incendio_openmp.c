#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <sys/resource.h>
#include <omp.h>

using namespace std;
using namespace chrono;

#define NUM_HILOS 4   // ← CAMBIA AQUI EL NUMERO DE HILOS

enum Estado {
    SANO = 0,
    FUEGO = 1,
    QUEMADO = 2
};

double obtenerMemoriaMB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss / 1024.0;
}

int main() {

    omp_set_num_threads(NUM_HILOS);

    int filas, columnas, simulaciones, T_MAX;
    double probBase;

    cout << "Ingrese numero de filas del terreno: ";
    cin >> filas;

    cout << "Ingrese numero de columnas del terreno: ";
    cin >> columnas;

    cout << "Ingrese numero de simulaciones Monte Carlo: ";
    cin >> simulaciones;

    cout << "Ingrese tiempo maximo de simulacion (T_MAX): ";
    cin >> T_MAX;

    cout << "Ingrese probabilidad base de propagacion (0-1): ";
    cin >> probBase;

    vector<vector<int>> contador(filas, vector<int>(columnas, 0));

    auto inicio = high_resolution_clock::now();

    #pragma omp parallel
    {
        random_device rd;
        mt19937 gen(rd() + omp_get_thread_num());

        uniform_real_distribution<> dist01(0.0, 1.0);
        uniform_real_distribution<> viento(0.9, 1.2);
        uniform_real_distribution<> vegetacion(0.8, 1.3);
        uniform_real_distribution<> humedad(0.7, 1.0);
        uniform_real_distribution<> temperatura(0.9, 1.2);
        uniform_real_distribution<> pendiente(0.9, 1.1);

        uniform_int_distribution<> distFila(0, filas - 1);
        uniform_int_distribution<> distCol(0, columnas - 1);

        vector<vector<int>> contador_local(filas, vector<int>(columnas, 0));

        #pragma omp for
        for (int s = 0; s < simulaciones; s++) {

            vector<vector<int>> estado(filas, vector<int>(columnas, SANO));

            int inicioFila = distFila(gen);
            int inicioCol = distCol(gen);
            estado[inicioFila][inicioCol] = FUEGO;

            double factorGlobal =
                viento(gen) *
                vegetacion(gen) *
                humedad(gen) *
                temperatura(gen) *
                pendiente(gen);

            for (int t = 0; t < T_MAX; t++) {

                vector<vector<int>> nuevoEstado = estado;

                for (int i = 0; i < filas; i++) {
                    for (int j = 0; j < columnas; j++) {

                        if (estado[i][j] == FUEGO) {

                            nuevoEstado[i][j] = QUEMADO;

                            auto intentar = [&](int ni, int nj) {
                                if (ni >= 0 && ni < filas &&
                                    nj >= 0 && nj < columnas &&
                                    estado[ni][nj] == SANO) {

                                    double p = probBase * factorGlobal;
                                    if (p > 1.0) p = 1.0;

                                    if (dist01(gen) < p)
                                        nuevoEstado[ni][nj] = FUEGO;
                                }
                            };

                            intentar(i-1, j);
                            intentar(i+1, j);
                            intentar(i, j-1);
                            intentar(i, j+1);
                        }
                    }
                }

                estado = nuevoEstado;
            }

            for (int i = 0; i < filas; i++)
                for (int j = 0; j < columnas; j++)
                    if (estado[i][j] == QUEMADO)
                        contador_local[i][j]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < filas; i++)
                for (int j = 0; j < columnas; j++)
                    contador[i][j] += contador_local[i][j];
        }
    }

    auto fin = high_resolution_clock::now();
    duration<double> tiempo = fin - inicio;

    // ===============================
    // CALCULAR PROBABILIDADES
    // ===============================

    vector<vector<double>> probabilidades(filas, vector<double>(columnas));

    for (int i = 0; i < filas; i++)
        for (int j = 0; j < columnas; j++)
            probabilidades[i][j] = (double)contador[i][j] / simulaciones;

    // ===============================
    // GUARDAR CSV
    // ===============================

    ofstream archivo("probabilidades.csv");

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            archivo << probabilidades[i][j];
            if (j < columnas - 1) archivo << ",";
        }
        archivo << "\n";
    }

    archivo.close();

    // ===============================

    // RESULTADOS
    // ===============================

    cout << "\n===== RESULTADOS =====\n";
    cout << "Terreno: " << filas << " x " << columnas << endl;
    cout << "Simulaciones: " << simulaciones << endl;
    cout << "T_MAX: " << T_MAX << endl;
    cout << "Hilos OpenMP: " << NUM_HILOS << endl;
    cout << "Tiempo de ejecucion: " << tiempo.count() << " segundos\n";
    cout << "Memoria maxima usada: " << obtenerMemoriaMB() << " MB\n";
    cout << "Archivo probabilidades.csv generado correctamente.\n";

    return 0;
}
//g++ -fopenmp incendio_openmp.cpp -o incendio
//./incendio