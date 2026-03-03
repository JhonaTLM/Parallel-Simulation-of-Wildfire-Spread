// =============================================================================
// SIMULACIÓN MONTE CARLO - PROPAGACIÓN DE INCENDIOS FORESTALES (OpenMP)
// =============================================================================

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>
#include <omp.h>

using namespace std;
using namespace chrono;

#define NUM_HILOS 4  // Número de hilos OpenMP

enum Estado {
    SANO = 0,
    FUEGO = 1,
    QUEMADO = 2
};

int main(int argc, char* argv[]) {

    // ===================== VALIDAR ARGUMENTOS =====================
    if (argc < 13) {
        cerr << "Uso: openmp.exe filas columnas simulaciones T_MAX probBase inicio_x inicio_y viento vegetacion humedad temperatura pendiente" << endl;
        return 1;
    }

    int filas = atoi(argv[1]);
    int columnas = atoi(argv[2]);
    int simulaciones = atoi(argv[3]);
    int T_MAX = atoi(argv[4]);
    double probBase = atof(argv[5]);
    int inicio_x = atoi(argv[6]);
    int inicio_y = atoi(argv[7]);
    double f_viento = atof(argv[8]);
    double f_vegetacion = atof(argv[9]);
    double f_humedad = atof(argv[10]);
    double f_temperatura = atof(argv[11]);
    double f_pendiente = atof(argv[12]);

    // Validar que el punto de inicio esté dentro del rango
    if (inicio_x < 0 || inicio_x >= filas || inicio_y < 0 || inicio_y >= columnas) {
        cerr << "Error: Punto de inicio fuera del rango. Debe ser 0 <= x < filas y 0 <= y < columnas" << endl;
        return 1;
    }

    omp_set_num_threads(NUM_HILOS);

    // ===================== INICIALIZAR CONTADOR =====================
    vector<vector<int>> contador(filas, vector<int>(columnas, 0));

    // ===================== SIMULACIÓN MONTE CARLO CON OpenMP =====================
    auto inicio = high_resolution_clock::now();

    #pragma omp parallel
    {
        // Cada hilo tiene su propio generador aleatorio
        random_device rd;
        mt19937 gen(rd() + omp_get_thread_num());

        uniform_real_distribution<> dist01(0.0, 1.0);

        // Contador local para evitar condiciones de carrera
        vector<vector<int>> contador_local(filas, vector<int>(columnas, 0));

        #pragma omp for
        for (int s = 0; s < simulaciones; s++) {

            // Inicializar terreno
            vector<vector<int>> estado(filas, vector<int>(columnas, SANO));

            // Punto de inicio fijo del fuego (definido por el usuario)
            estado[inicio_x][inicio_y] = FUEGO;

            // Factor global basado en parámetros del usuario
            double factorGlobal = f_viento * f_vegetacion * f_humedad * f_temperatura * f_pendiente;

            // Propagación temporal
            for (int t = 0; t < T_MAX; t++) {

                vector<vector<int>> nuevoEstado = estado;

                for (int i = 0; i < filas; i++) {
                    for (int j = 0; j < columnas; j++) {

                        if (estado[i][j] == FUEGO) {

                            nuevoEstado[i][j] = QUEMADO;

                            // Intentar propagar a 8 vecinos (Moore)
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

                            // Vecindad de Moore (8 vecinos)
                            intentar(i - 1, j);      // arriba
                            intentar(i + 1, j);      // abajo
                            intentar(i, j - 1);      // izquierda
                            intentar(i, j + 1);      // derecha
                            intentar(i - 1, j - 1);  // diagonal superior izquierda
                            intentar(i - 1, j + 1);  // diagonal superior derecha
                            intentar(i + 1, j - 1);  // diagonal inferior izquierda
                            intentar(i + 1, j + 1);  // diagonal inferior derecha
                        }
                    }
                }

                estado = nuevoEstado;
            }

            // Contar celdas quemadas (incluye FUEGO del último paso)
            for (int i = 0; i < filas; i++)
                for (int j = 0; j < columnas; j++)
                    if (estado[i][j] == QUEMADO || estado[i][j] == FUEGO)
                        contador_local[i][j]++;
        }

        // Combinar contadores locales en el contador global
        #pragma omp critical
        {
            for (int i = 0; i < filas; i++)
                for (int j = 0; j < columnas; j++)
                    contador[i][j] += contador_local[i][j];
        }
    }

    auto fin = high_resolution_clock::now();
    duration<double> tiempo = fin - inicio;

    // ===================== CALCULAR PROBABILIDADES =====================
    vector<vector<double>> probabilidades(filas, vector<double>(columnas));

    for (int i = 0; i < filas; i++)
        for (int j = 0; j < columnas; j++)
            probabilidades[i][j] = (double)contador[i][j] / simulaciones;

    // ===================== IMPRIMIR MATRIZ CSV A STDOUT =====================
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            printf("%.6f", probabilidades[i][j]);
            if (j < columnas - 1) printf(",");
        }
        printf("\n");
    }

    // ===================== IMPRIMIR TIEMPO =====================
    printf("TIEMPO %.6f\n", tiempo.count());

    return 0;
}
