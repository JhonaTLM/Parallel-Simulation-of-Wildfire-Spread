// =============================================================================
// SIMULACIÓN MONTE CARLO - PROPAGACIÓN DE INCENDIOS FORESTALES (CUDA)
// Compatible con interfaz Streamlit - Windows
// =============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define SANO 0
#define FUEGO 1
#define QUEMADO 2

// =============================
// ESTRUCTURA TERRENO
// =============================

typedef struct {
    int filas;
    int columnas;
    int *estado;
} Terreno;

Terreno* crear_terreno(int filas, int columnas) {
    Terreno *t = (Terreno*)malloc(sizeof(Terreno));
    t->filas = filas;
    t->columnas = columnas;
    t->estado = (int*)malloc(filas * columnas * sizeof(int));
    for (int i = 0; i < filas * columnas; i++)
        t->estado[i] = SANO;
    return t;
}

void liberar_terreno(Terreno *t) {
    free(t->estado);
    free(t);
}

// =============================
// IMPRIMIR INFO GPU (stderr)
// =============================

void imprimir_info_gpu(int filas, int columnas, int threads_por_bloque) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        fprintf(stderr, "GPU_NOMBRE:Sin GPU CUDA detectada\n");
        fprintf(stderr, "GPU_MEMORIA_MB:0\n");
        fprintf(stderr, "GPU_MULTIPROCESADORES:0\n");
        fprintf(stderr, "CUDA_THREADS_POR_BLOQUE:%d\n", threads_por_bloque);
        fprintf(stderr, "CUDA_BLOQUES:0\n");
        fprintf(stderr, "CUDA_THREADS_TOTAL:0\n");
        return;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int total_celdas = filas * columnas;
    int bloques = (total_celdas + threads_por_bloque - 1) / threads_por_bloque;
    int threads_total = bloques * threads_por_bloque;

    // Info GPU
    fprintf(stderr, "GPU_NOMBRE:%s\n", prop.name);
    fprintf(stderr, "GPU_MEMORIA_MB:%d\n", (int)(prop.totalGlobalMem / (1024 * 1024)));
    fprintf(stderr, "GPU_MULTIPROCESADORES:%d\n", prop.multiProcessorCount);
    fprintf(stderr, "GPU_MAX_THREADS_SM:%d\n", prop.maxThreadsPerMultiProcessor);

    // Info de ejecución
    fprintf(stderr, "CUDA_THREADS_POR_BLOQUE:%d\n", threads_por_bloque);
    fprintf(stderr, "CUDA_BLOQUES:%d\n", bloques);
    fprintf(stderr, "CUDA_THREADS_TOTAL:%d\n", threads_total);
    fprintf(stderr, "CUDA_SIMULACIONES:%d\n", 0);  // Se actualiza después
}

// =============================
// KERNEL: INICIALIZAR CURAND
// =============================

__global__ void init_curand_kernel(curandState *states, unsigned long long seed, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    curand_init(seed, (unsigned long long)idx, 0, &states[idx]);
}

// =============================
// KERNEL: UN PASO DE PROPAGACIÓN
// =============================

__global__ void simular_kernel(
    int *estado, int *nuevo,
    int filas, int columnas,
    float prob,
    curandState *states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = filas * columnas;
    if (idx >= total) return;

    int i = idx / columnas;
    int j = idx % columnas;

    if (estado[idx] == FUEGO) {
        // Celda en fuego pasa a quemada
        nuevo[idx] = QUEMADO;

    } else if (estado[idx] == SANO) {
        // Revisar si algún vecino está en fuego (Vecindad de Moore - 8 vecinos)
        int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        bool vecino_fuego = false;

        for (int d = 0; d < 8; d++) {
            int ni = i + di[d];
            int nj = j + dj[d];
            if (ni >= 0 && ni < filas && nj >= 0 && nj < columnas) {
                if (estado[ni * columnas + nj] == FUEGO) {
                    vecino_fuego = true;
                    break;
                }
            }
        }

        if (vecino_fuego) {
            // Generar número aleatorio con CURAND
            float r = curand_uniform(&states[idx]);
            nuevo[idx] = (r < prob) ? FUEGO : SANO;
        } else {
            nuevo[idx] = SANO;
        }

    } else {
        // Celda quemada permanece quemada
        nuevo[idx] = QUEMADO;
    }
}

// =============================
// SIMULACIÓN MONTE CARLO CUDA
// =============================

float** montecarlo_incendios(
    Terreno *terreno_inicial,
    int N_simulaciones,
    int T_max,
    float probBase,
    int threads,
    int inicio_x,
    int inicio_y,
    float f_viento,
    float f_vegetacion,
    float f_humedad,
    float f_temperatura,
    float f_pendiente
) {
    int filas    = terreno_inicial->filas;
    int columnas = terreno_inicial->columnas;
    int total    = filas * columnas;

    // Matriz de contadores
    float **contador = (float**)malloc(filas * sizeof(float*));
    for (int i = 0; i < filas; i++)
        contador[i] = (float*)calloc(columnas, sizeof(float));

    // Memoria en GPU
    int *d_estado, *d_nuevo;
    cudaMalloc(&d_estado, total * sizeof(int));
    cudaMalloc(&d_nuevo,  total * sizeof(int));

    curandState *d_states;
    cudaMalloc(&d_states, total * sizeof(curandState));

    int blocks = (total + threads - 1) / threads;

    // Semilla base calculada una vez para evitar repeticiones con time(NULL)
    unsigned long long base_seed = (unsigned long long)time(NULL) * 1099511628211ULL;

    for (int sim = 0; sim < N_simulaciones; sim++) {

        // Reiniciar terreno
        for (int k = 0; k < total; k++)
            terreno_inicial->estado[k] = SANO;

        // Punto de inicio fijo del fuego (definido por el usuario)
        terreno_inicial->estado[inicio_x * columnas + inicio_y] = FUEGO;

        // Factor global basado en parámetros del usuario
        float factorGlobal = f_viento * f_vegetacion * f_humedad * f_temperatura * f_pendiente;

        float prob = probBase * factorGlobal;

        // Copiar estado inicial a GPU
        cudaMemcpy(d_estado, terreno_inicial->estado, total * sizeof(int), cudaMemcpyHostToDevice);

        // Inicializar CURAND con semilla única por simulación
        // Usamos base_seed (calculado una vez al inicio) combinado con sim
        unsigned long long seed = base_seed ^ ((unsigned long long)sim * 6364136223846793005ULL + sim);
        init_curand_kernel<<<blocks, threads>>>(d_states, seed, total);
        cudaDeviceSynchronize();

        // Propagación temporal
        for (int t = 0; t < T_max; t++) {
            cudaMemcpy(d_nuevo, d_estado, total * sizeof(int), cudaMemcpyDeviceToDevice);
            simular_kernel<<<blocks, threads>>>(d_estado, d_nuevo, filas, columnas, prob, d_states);
            cudaDeviceSynchronize();

            // Intercambiar punteros
            int *tmp = d_estado;
            d_estado = d_nuevo;
            d_nuevo  = tmp;
        }

        // Copiar resultado final a host
        int *h_final = (int*)malloc(total * sizeof(int));
        cudaMemcpy(h_final, d_estado, total * sizeof(int), cudaMemcpyDeviceToHost);

        // Contar celdas quemadas
        for (int i = 0; i < filas; i++)
            for (int j = 0; j < columnas; j++)
                if (h_final[i * columnas + j] == QUEMADO)
                    contador[i][j]++;

        free(h_final);
    }

    // Liberar memoria GPU
    cudaFree(d_estado);
    cudaFree(d_nuevo);
    cudaFree(d_states);

    // Calcular probabilidades
    for (int i = 0; i < filas; i++)
        for (int j = 0; j < columnas; j++)
            contador[i][j] /= N_simulaciones;

    return contador;
}

// =============================
// MAIN
// =============================

int main(int argc, char* argv[]) {

    if (argc < 13) {
        printf("Uso: cuda.exe filas columnas simulaciones T_MAX probBase inicio_x inicio_y viento vegetacion humedad temperatura pendiente\n");
        return 1;
    }

    srand(time(NULL));

    int   filas          = atoi(argv[1]);
    int   columnas       = atoi(argv[2]);
    int   N_simulaciones = atoi(argv[3]);
    int   T_max          = atoi(argv[4]);
    float probBase       = atof(argv[5]);
    int   inicio_x       = atoi(argv[6]);
    int   inicio_y       = atoi(argv[7]);
    float f_viento       = atof(argv[8]);
    float f_vegetacion   = atof(argv[9]);
    float f_humedad      = atof(argv[10]);
    float f_temperatura  = atof(argv[11]);
    float f_pendiente    = atof(argv[12]);

    // Validar que el punto de inicio esté dentro del rango
    if (inicio_x < 0 || inicio_x >= filas || inicio_y < 0 || inicio_y >= columnas) {
        fprintf(stderr, "Error: Punto de inicio fuera del rango. Debe ser 0 <= x < filas y 0 <= y < columnas\n");
        return 1;
    }

    int threads = 256;

    // Imprimir información de GPU por stderr
    imprimir_info_gpu(filas, columnas, threads);

    Terreno *terreno = crear_terreno(filas, columnas);

    // Usar cudaEvent para medir tiempo con precisión en GPU
    cudaEvent_t cuda_inicio, cuda_fin;
    cudaEventCreate(&cuda_inicio);
    cudaEventCreate(&cuda_fin);
    cudaEventRecord(cuda_inicio);

    float **probabilidades = montecarlo_incendios(
        terreno, N_simulaciones, T_max, probBase, threads, inicio_x, inicio_y,
        f_viento, f_vegetacion, f_humedad, f_temperatura, f_pendiente
    );

    cudaEventRecord(cuda_fin);
    cudaEventSynchronize(cuda_fin);
    float tiempo_ms = 0.0f;
    cudaEventElapsedTime(&tiempo_ms, cuda_inicio, cuda_fin);
    double tiempo_total = tiempo_ms / 1000.0;  // Convertir a segundos
    cudaEventDestroy(cuda_inicio);
    cudaEventDestroy(cuda_fin);

    // ===================== IMPRIMIR MATRIZ CSV A STDOUT =====================
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            printf("%.6f", probabilidades[i][j]);
            if (j < columnas - 1) printf(",");
        }
        printf("\n");
    }

    // ===================== IMPRIMIR TIEMPO =====================
    printf("TIEMPO %.6f\n", tiempo_total);

    // Liberar memoria
    liberar_terreno(terreno);
    for (int i = 0; i < filas; i++)
        free(probabilidades[i]);
    free(probabilidades);

    return 0;
}
