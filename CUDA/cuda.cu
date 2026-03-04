#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define SANO 0
#define FUEGO 1
#define QUEMADO 2

// =============================
// GENERADOR ALEATORIO INLINE
// =============================

__device__ __forceinline__ float rand_float(unsigned int seed) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (float)(seed & 0x00FFFFFF) / (float)0x01000000;
}

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

void imprimir_info_gpu(int filas, int columnas, int N_simulaciones, int threads_por_bloque) {
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

    int total_trabajo = filas * columnas * N_simulaciones;
    int bloques = (total_trabajo + threads_por_bloque - 1) / threads_por_bloque;
    int threads_total = bloques * threads_por_bloque;

    fprintf(stderr, "GPU_NOMBRE:%s\n", prop.name);
    fprintf(stderr, "GPU_MEMORIA_MB:%d\n", (int)(prop.totalGlobalMem / (1024 * 1024)));
    fprintf(stderr, "GPU_MULTIPROCESADORES:%d\n", prop.multiProcessorCount);
    fprintf(stderr, "GPU_MAX_THREADS_SM:%d\n", prop.maxThreadsPerMultiProcessor);
    fprintf(stderr, "CUDA_THREADS_POR_BLOQUE:%d\n", threads_por_bloque);
    fprintf(stderr, "CUDA_BLOQUES:%d\n", bloques);
    fprintf(stderr, "CUDA_THREADS_TOTAL:%d\n", threads_total);
    fprintf(stderr, "CUDA_SIMULACIONES:%d\n", N_simulaciones);
}

// =============================
// KERNEL: REINICIAR TODOS LOS TERRENOS (todas las simulaciones en paralelo)
// =============================

__global__ void reiniciar_todos_kernel(int *estado, int total_celdas, int N_simulaciones, int inicio_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_celdas * N_simulaciones;
    if (idx >= total) return;
    
    int celda = idx % total_celdas;
    estado[idx] = (celda == inicio_idx) ? FUEGO : SANO;
}

// =============================
// KERNEL: UN PASO DE PROPAGACIÓN (todas las simulaciones en paralelo)
// =============================

__global__ void simular_kernel(
    int *estado, int *nuevo,
    int filas, int columnas,
    int total_celdas, int N_simulaciones,
    float *probPorDir,
    unsigned int base_seed,
    int t
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_celdas * N_simulaciones;
    if (idx >= total) return;

    int sim = idx / total_celdas;    // Qué simulación
    int celda = idx % total_celdas;  // Qué celda
    int i = celda / columnas;
    int j = celda % columnas;
    int offset = sim * total_celdas;

    if (estado[idx] == FUEGO) {
        nuevo[idx] = QUEMADO;

    } else if (estado[idx] == SANO) {
        int di[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int dj[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
        int dirMap[8] = {3, 4, 5, 2, 6, 1, 0, 7};
        
        bool se_enciende = false;
        unsigned int seed = base_seed + idx * 15485863u + t * 1299709u;

        for (int d = 0; d < 8; d++) {
            int ni = i + di[d];
            int nj = j + dj[d];
            if (ni >= 0 && ni < filas && nj >= 0 && nj < columnas) {
                int vecino_idx = offset + ni * columnas + nj;
                if (estado[vecino_idx] == FUEGO) {
                    seed = seed * 1103515245u + 12345u;
                    float r = rand_float(seed);
                    if (r < probPorDir[dirMap[d]]) {
                        se_enciende = true;
                        break;
                    }
                }
            }
        }
        nuevo[idx] = se_enciende ? FUEGO : SANO;

    } else {
        nuevo[idx] = QUEMADO;
    }
}

// =============================
// KERNEL: CONTAR QUEMADOS (acumular de todas las simulaciones)
// =============================

__global__ void contar_todos_kernel(int *estado, int *contador, int total_celdas, int N_simulaciones) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_celdas * N_simulaciones;
    if (idx >= total) return;

    int celda = idx % total_celdas;
    if (estado[idx] == QUEMADO || estado[idx] == FUEGO) {
        atomicAdd(&contador[celda], 1);
    }
}

// =============================
// SIMULACIÓN MONTE CARLO CUDA (TODAS LAS SIMULACIONES EN PARALELO)
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
    float f_pendiente,
    int viento_dir
) {
    int filas = terreno_inicial->filas;
    int columnas = terreno_inicial->columnas;
    int total_celdas = filas * columnas;
    int total = total_celdas * N_simulaciones;  // Total de trabajo
    int inicio_idx = inicio_x * columnas + inicio_y;

    // Memoria para TODAS las simulaciones en paralelo
    int *d_estado, *d_nuevo;
    cudaMalloc(&d_estado, total * sizeof(int));
    cudaMalloc(&d_nuevo, total * sizeof(int));

    int *d_estado_original = d_estado;
    int *d_nuevo_original = d_nuevo;

    // Contador por celda (acumula de todas las simulaciones)
    int *d_contador;
    cudaMalloc(&d_contador, total_celdas * sizeof(int));
    cudaMemset(d_contador, 0, total_celdas * sizeof(int));

    int blocks = (total + threads - 1) / threads;

    // Precalcular probabilidades por dirección
    float h_probPorDir[8];
    float factorBase = f_vegetacion * f_humedad * f_temperatura * f_pendiente;
    
    for (int d = 0; d < 8; d++) {
        int diff = abs(viento_dir - d);
        if (diff > 4) diff = 8 - diff;
        
        float factorViento;
        switch (diff) {
            case 0: factorViento = f_viento * 1.3f; break;
            case 1: factorViento = f_viento * 1.1f; break;
            case 2: factorViento = f_viento * 1.0f; break;
            case 3: factorViento = f_viento * 0.9f; break;
            case 4: factorViento = f_viento * 0.7f; break;
            default: factorViento = f_viento;
        }
        
        h_probPorDir[d] = probBase * factorBase * factorViento;
        if (h_probPorDir[d] > 1.0f) h_probPorDir[d] = 1.0f;
    }
    
    float *d_probPorDir;
    cudaMalloc(&d_probPorDir, 8 * sizeof(float));
    cudaMemcpy(d_probPorDir, h_probPorDir, 8 * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int base_seed = (unsigned int)time(NULL);

    // Reiniciar TODOS los terrenos en paralelo (1 kernel)
    reiniciar_todos_kernel<<<blocks, threads>>>(d_estado, total_celdas, N_simulaciones, inicio_idx);

    // Solo el loop de tiempo está en CPU (T_max kernels en vez de N_sim * T_max)
    for (int t = 0; t < T_max; t++) {
        simular_kernel<<<blocks, threads>>>(d_estado, d_nuevo, filas, columnas, total_celdas, N_simulaciones, d_probPorDir, base_seed, t);
        
        int *tmp = d_estado;
        d_estado = d_nuevo;
        d_nuevo = tmp;
    }

    // Contar quemados de TODAS las simulaciones (1 kernel)
    contar_todos_kernel<<<blocks, threads>>>(d_estado, d_contador, total_celdas, N_simulaciones);

    cudaDeviceSynchronize();

    // Copiar contadores a CPU
    int *h_contador = (int*)malloc(total_celdas * sizeof(int));
    cudaMemcpy(h_contador, d_contador, total_celdas * sizeof(int), cudaMemcpyDeviceToHost);

    // Calcular probabilidades
    float **resultado = (float**)malloc(filas * sizeof(float*));
    for (int i = 0; i < filas; i++) {
        resultado[i] = (float*)malloc(columnas * sizeof(float));
        for (int j = 0; j < columnas; j++) {
            resultado[i][j] = (float)h_contador[i * columnas + j] / N_simulaciones;
        }
    }

    free(h_contador);
    cudaFree(d_estado_original);
    cudaFree(d_nuevo_original);
    cudaFree(d_contador);
    cudaFree(d_probPorDir);

    return resultado;
}

// =============================
// MAIN
// =============================

int main(int argc, char* argv[]) {

    if (argc < 14) {
        printf("Uso: cuda.exe filas columnas simulaciones T_MAX probBase inicio_x inicio_y viento vegetacion humedad temperatura pendiente viento_dir\n");
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
    int   viento_dir     = atoi(argv[13]);  // 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW

    // Validar que el punto de inicio esté dentro del rango
    if (inicio_x < 0 || inicio_x >= filas || inicio_y < 0 || inicio_y >= columnas) {
        fprintf(stderr, "Error: Punto de inicio fuera del rango. Debe ser 0 <= x < filas y 0 <= y < columnas\n");
        return 1;
    }

    int threads = 256;

    // Imprimir información de GPU por stderr
    imprimir_info_gpu(filas, columnas, N_simulaciones, threads);

    Terreno *terreno = crear_terreno(filas, columnas);

    // Usar cudaEvent para medir tiempo con precisión en GPU
    cudaEvent_t cuda_inicio, cuda_fin;
    cudaEventCreate(&cuda_inicio);
    cudaEventCreate(&cuda_fin);
    cudaEventRecord(cuda_inicio);

    float **probabilidades = montecarlo_incendios(
        terreno, N_simulaciones, T_max, probBase, threads, inicio_x, inicio_y,
        f_viento, f_vegetacion, f_humedad, f_temperatura, f_pendiente, viento_dir
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
