// =============================================================================
// SIMULACIÓN MONTE CARLO - PROPAGACIÓN DE INCENDIOS FORESTALES (SECUENCIAL)
// Compatible con interfaz Streamlit - Windows
// =============================================================================

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdlib>

using namespace std;
using namespace chrono;

enum Estado {
    SANO = 0,
    FUEGO = 1,
    QUEMADO = 2
};

int main(int argc, char* argv[]) {

    // ===================== VALIDAR ARGUMENTOS =====================
    if (argc < 13) {
        cerr << "Uso: secuencial.exe filas columnas simulaciones T_MAX probBase inicio_x inicio_y viento vegetacion humedad temperatura pendiente" << endl;
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

    // ===================== INICIALIZAR GENERADOR ALEATORIO =====================
    vector<vector<int>> contador(filas, vector<int>(columnas, 0));

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<> dist01(0.0, 1.0);
    // Rango de variación aleatoria adicional (pequeña) sobre los factores del usuario
    uniform_real_distribution<> variacion(0.95, 1.05);

    uniform_int_distribution<> distFila(0, filas - 1);
    uniform_int_distribution<> distCol(0, columnas - 1);

    // ===================== SIMULACIÓN MONTE CARLO =====================
    auto inicio = high_resolution_clock::now();

    for (int s = 0; s < simulaciones; s++) {

        // Inicializar terreno
        vector<vector<int>> estado(filas, vector<int>(columnas, SANO));

        // Punto de inicio fijo del fuego (definido por el usuario)
        estado[inicio_x][inicio_y] = FUEGO;

        // Factor global basado en parámetros del usuario con pequeña variación aleatoria
        double factorGlobal = 
            f_viento * variacion(gen) *
            f_vegetacion * variacion(gen) *
            f_humedad * variacion(gen) *
            f_temperatura * variacion(gen) *
            f_pendiente * variacion(gen);

        // Propagación temporal
        for (int t = 0; t < T_MAX; t++) {

            vector<vector<int>> nuevoEstado = estado;

            for (int i = 0; i < filas; i++) {
                for (int j = 0; j < columnas; j++) {

                    if (estado[i][j] == FUEGO) {

                        nuevoEstado[i][j] = QUEMADO;

                        // Intentar propagar a 4 vecinos
                        auto intentarPropagar = [&](int ni, int nj) {
                            if (ni >= 0 && ni < filas &&
                                nj >= 0 && nj < columnas &&
                                estado[ni][nj] == SANO) {

                                double p_final = probBase * factorGlobal;
                                if (p_final > 1.0) p_final = 1.0;

                                if (dist01(gen) < p_final)
                                    nuevoEstado[ni][nj] = FUEGO;
                            }
                        };

                        intentarPropagar(i - 1, j);  // arriba
                        intentarPropagar(i + 1, j);  // abajo
                        intentarPropagar(i, j - 1);  // izquierda
                        intentarPropagar(i, j + 1);  // derecha
                    }
                }
            }

            estado = nuevoEstado;
        }

        // Contar celdas quemadas
        for (int i = 0; i < filas; i++)
            for (int j = 0; j < columnas; j++)
                if (estado[i][j] == QUEMADO)
                    contador[i][j]++;
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
