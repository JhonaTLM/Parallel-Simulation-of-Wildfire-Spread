#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SANO 0
#define FUEGO 1
#define QUEMADO 2

double rand01() {
    return rand() / (double) RAND_MAX;
}

int main(int argc, char *argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int filas, columnas, simulaciones, T_MAX;
    double probBase;

    if (rank == 0) {
        printf("Ingrese numero de filas: ");
        scanf("%d", &filas);
        printf("Ingrese numero de columnas: ");
        scanf("%d", &columnas);
        printf("Ingrese numero de simulaciones Monte Carlo: ");
        scanf("%d", &simulaciones);
        printf("Ingrese T_MAX: ");
        scanf("%d", &T_MAX);
        printf("Ingrese probabilidad base (0-1): ");
        scanf("%lf", &probBase);
    }

    MPI_Bcast(&filas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&simulaciones, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&T_MAX, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&probBase, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int sims_local = simulaciones / size;

    srand(time(NULL) + rank * 100);

    int **contador_local = malloc(filas * sizeof(int *));
    for (int i = 0; i < filas; i++) {
        contador_local[i] = calloc(columnas, sizeof(int));
    }

    double t_inicio = MPI_Wtime();

    for (int s = 0; s < sims_local; s++) {

        int **estado = malloc(filas * sizeof(int *));
        for (int i = 0; i < filas; i++) {
            estado[i] = malloc(columnas * sizeof(int));
            for (int j = 0; j < columnas; j++)
                estado[i][j] = SANO;
        }

        int fi = rand() % filas;
        int co = rand() % columnas;
        estado[fi][co] = FUEGO;

        double factorGlobal =
            (0.9 + rand01() * 0.3) *
            (0.8 + rand01() * 0.5) *
            (0.7 + rand01() * 0.3) *
            (0.9 + rand01() * 0.3) *
            (0.9 + rand01() * 0.2);

        for (int t = 0; t < T_MAX; t++) {

            int **nuevo = malloc(filas * sizeof(int *));
            for (int i = 0; i < filas; i++) {
                nuevo[i] = malloc(columnas * sizeof(int));
                for (int j = 0; j < columnas; j++)
                    nuevo[i][j] = estado[i][j];
            }

            for (int i = 0; i < filas; i++) {
                for (int j = 0; j < columnas; j++) {

                    if (estado[i][j] == FUEGO) {
                        nuevo[i][j] = QUEMADO;

                        int di[4] = {-1, 1, 0, 0};
                        int dj[4] = {0, 0, -1, 1};

                        for (int k = 0; k < 4; k++) {
                            int ni = i + di[k];
                            int nj = j + dj[k];

                            if (ni >= 0 && ni < filas &&
                                nj >= 0 && nj < columnas &&
                                estado[ni][nj] == SANO) {

                                double p = probBase * factorGlobal;
                                if (p > 1.0) p = 1.0;

                                if (rand01() < p)
                                    nuevo[ni][nj] = FUEGO;
                            }
                        }
                    }
                }
            }

            for (int i = 0; i < filas; i++) {
                free(estado[i]);
                estado[i] = nuevo[i];
            }
            free(nuevo);
        }

        for (int i = 0; i < filas; i++)
            for (int j = 0; j < columnas; j++)
                if (estado[i][j] == QUEMADO)
                    contador_local[i][j]++;

        for (int i = 0; i < filas; i++)
            free(estado[i]);
        free(estado);
    }

    int *local_flat = malloc(filas * columnas * sizeof(int));
    int *global_flat = NULL;

    for (int i = 0; i < filas; i++)
        for (int j = 0; j < columnas; j++)
            local_flat[i * columnas + j] = contador_local[i][j];

    if (rank == 0)
        global_flat = calloc(filas * columnas, sizeof(int));

    MPI_Reduce(local_flat, global_flat,
               filas * columnas, MPI_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    double t_fin = MPI_Wtime();

    if (rank == 0) {
        FILE *f = fopen("probabilidades.csv", "w");
        for (int i = 0; i < filas; i++) {
            for (int j = 0; j < columnas; j++) {
                double p = (double)global_flat[i * columnas + j] / simulaciones;
                fprintf(f, "%lf", p);
                if (j < columnas - 1) fprintf(f, ",");
            }
            fprintf(f, "\n");
        }
        fclose(f);

        printf("\n===== MPI RESULTADOS =====\n");
        printf("Procesos: %d\n", size);
        printf("Tiempo ejecucion: %lf segundos\n", t_fin - t_inicio);
        printf("Archivo probabilidades.csv generado\n");
    }

    MPI_Finalize();
    return 0;
}
