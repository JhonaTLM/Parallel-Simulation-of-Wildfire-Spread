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
    if (argc < 14) {
        cerr << "Uso: openmp.exe filas columnas simulaciones T_MAX probBase inicio_x inicio_y viento vegetacion humedad temperatura pendiente viento_dir" << endl;
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
    int viento_dir = atoi(argv[13]);  // 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW

    // Validar que el punto de inicio esté dentro del rango
    if (inicio_x < 0 || inicio_x >= filas || inicio_y < 0 || inicio_y >= columnas) {
        cerr << "Error: Punto de inicio fuera del rango. Debe ser 0 <= x < filas y 0 <= y < columnas" << endl;
        return 1;
    }

    omp_set_num_threads(NUM_HILOS);

    // ===================== PRECALCULAR PROBABILIDADES POR DIRECCIÓN =====================
    // Direcciones: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    double probPorDir[8];
    double factorBase = f_vegetacion * f_humedad * f_temperatura * f_pendiente;
    
    for (int d = 0; d < 8; d++) {
        int diff = abs(viento_dir - d);
        if (diff > 4) diff = 8 - diff;  // distancia angular mínima
        
        double factorViento;
        switch (diff) {
            case 0: factorViento = f_viento * 1.3; break;  // a favor del viento
            case 1: factorViento = f_viento * 1.1; break;  // casi a favor
            case 2: factorViento = f_viento * 1.0; break;  // perpendicular
            case 3: factorViento = f_viento * 0.9; break;  // casi en contra
            case 4: factorViento = f_viento * 0.7; break;  // en contra del viento
            default: factorViento = f_viento;
        }
        
        probPorDir[d] = probBase * factorBase * factorViento;
        if (probPorDir[d] > 1.0) probPorDir[d] = 1.0;
    }

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

            // Propagación temporal
            for (int t = 0; t < T_MAX; t++) {

                vector<vector<int>> nuevoEstado = estado;

                for (int i = 0; i < filas; i++) {
                    for (int j = 0; j < columnas; j++) {

                        if (estado[i][j] == FUEGO) {

                            nuevoEstado[i][j] = QUEMADO;

                            // Intentar propagar a 8 vecinos (Moore) con probabilidad direccional
                            auto intentar = [&](int ni, int nj, int dir) {
                                if (ni >= 0 && ni < filas &&
                                    nj >= 0 && nj < columnas &&
                                    estado[ni][nj] == SANO) {

                                    if (dist01(gen) < probPorDir[dir])
                                        nuevoEstado[ni][nj] = FUEGO;
                                }
                            };

                            // Vecindad de Moore (8 vecinos) - dir indica hacia dónde se propaga
                            intentar(i - 1, j,     0);  // hacia N
                            intentar(i - 1, j + 1, 1);  // hacia NE
                            intentar(i,     j + 1, 2);  // hacia E
                            intentar(i + 1, j + 1, 3);  // hacia SE
                            intentar(i + 1, j,     4);  // hacia S
                            intentar(i + 1, j - 1, 5);  // hacia SW
                            intentar(i,     j - 1, 6);  // hacia W
                            intentar(i - 1, j - 1, 7);  // hacia NW
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
