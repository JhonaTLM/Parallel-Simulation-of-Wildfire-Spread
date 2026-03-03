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
// KERNEL: REINICIAR TERRENO EN GPU
// =============================

__global__ void reiniciar_terreno_kernel(int *estado, int total, int inicio_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    estado[idx] = (idx == inicio_idx) ? FUEGO : SANO;
}

// =============================
// KERNEL: CONTAR CELDAS QUEMADAS (incluye FUEGO del último paso)
// =============================

__global__ void contar_quemados_kernel(int *estado, int *contador, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // Contar QUEMADO y FUEGO (las celdas en fuego al final también se quemaron)
    if (estado[idx] == QUEMADO || estado[idx] == FUEGO) {
        atomicAdd(&contador[idx], 1);
    }
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
        // Revisar CADA vecino en fuego (Vecindad de Moore - 8 vecinos)
        // Cada vecino en fuego tiene una oportunidad INDEPENDIENTE de propagar
        int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        
        bool se_enciende = false;

        for (int d = 0; d < 8; d++) {
            int ni = i + di[d];
            int nj = j + dj[d];
            if (ni >= 0 && ni < filas && nj >= 0 && nj < columnas) {
                if (estado[ni * columnas + nj] == FUEGO) {
                    // Cada vecino en fuego tiene su propia oportunidad
                    float r = curand_uniform(&states[idx]);
                    if (r < prob) {
                        se_enciende = true;
                        break;  // Ya se encendió, no necesita más intentos
                    }
                }
            }
        }

        nuevo[idx] = se_enciende ? FUEGO : SANO;

    } else {
        // Celda quemada permanece quemada
        nuevo[idx] = QUEMADO;
    }
}

// =============================
// SIMULACIÓN MONTE CARLO CUDA (OPTIMIZADA)
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
    int inicio_idx = inicio_x * columnas + inicio_y;

    // Memoria en GPU para estados
    int *d_estado, *d_nuevo;
    cudaMalloc(&d_estado, total * sizeof(int));
    cudaMalloc(&d_nuevo,  total * sizeof(int));

    // Guardar punteros originales para liberar correctamente
    int *d_estado_original = d_estado;
    int *d_nuevo_original = d_nuevo;

    // Contador en GPU (evita transferencias constantes)
    int *d_contador;
    cudaMalloc(&d_contador, total * sizeof(int));
    cudaMemset(d_contador, 0, total * sizeof(int));

    // Estados CURAND
    curandState *d_states;
    cudaMalloc(&d_states, total * sizeof(curandState));

    int blocks = (total + threads - 1) / threads;

    // Factor global (calculado UNA sola vez)
    float factorGlobal = f_viento * f_vegetacion * f_humedad * f_temperatura * f_pendiente;
    float prob = probBase * factorGlobal;
    if (prob > 1.0f) prob = 1.0f;  // Clamping estándar

    // ✅ OPTIMIZACIÓN: Inicializar CURAND UNA SOLA VEZ
    unsigned long long base_seed = (unsigned long long)time(NULL) * 1099511628211ULL;
    init_curand_kernel<<<blocks, threads>>>(d_states, base_seed, total);
    cudaDeviceSynchronize();

    for (int sim = 0; sim < N_simulaciones; sim++) {

        // ✅ OPTIMIZACIÓN: Reiniciar terreno en GPU (no en CPU)
        reiniciar_terreno_kernel<<<blocks, threads>>>(d_estado, total, inicio_idx);
        cudaDeviceSynchronize();

        // Propagación temporal
        for (int t = 0; t < T_max; t++) {
            // Ejecutar kernel de propagación (lee d_estado, escribe d_nuevo)
            simular_kernel<<<blocks, threads>>>(d_estado, d_nuevo, filas, columnas, prob, d_states);
            cudaDeviceSynchronize();  // ✅ Sincronizar antes de intercambiar

            // Intercambiar punteros
            int *tmp = d_estado;
            d_estado = d_nuevo;
            d_nuevo  = tmp;
        }

        // ✅ Contar en GPU con atomicAdd
        contar_quemados_kernel<<<blocks, threads>>>(d_estado, d_contador, total);
    }

    // ✅ Sincronizar solo al final de todas las simulaciones
    cudaDeviceSynchronize();

    // Copiar contadores a CPU (UNA sola vez)
    int *h_contador = (int*)malloc(total * sizeof(int));
    cudaMemcpy(h_contador, d_contador, total * sizeof(int), cudaMemcpyDeviceToHost);

    // Calcular probabilidades
    float **resultado = (float**)malloc(filas * sizeof(float*));
    for (int i = 0; i < filas; i++) {
        resultado[i] = (float*)malloc(columnas * sizeof(float));
        for (int j = 0; j < columnas; j++) {
            resultado[i][j] = (float)h_contador[i * columnas + j] / N_simulaciones;
        }
    }

    // Liberar memoria (usar punteros originales)
    free(h_contador);
    cudaFree(d_estado_original);
    cudaFree(d_nuevo_original);
    cudaFree(d_states);
    cudaFree(d_contador);

    return resultado;
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
