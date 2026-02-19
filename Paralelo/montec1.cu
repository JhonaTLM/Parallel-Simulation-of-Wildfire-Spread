#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Estados de las celdas
#define VEGETACION_SANA 0
#define EN_COMBUSTION 1
#define QUEMADA 2

// Direcciones para vecindad de Moore (8 vecinos)
__constant__ int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
__constant__ int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// Estructura para parámetros ambientales
typedef struct {
    float P_base;              // Probabilidad base de propagación
    float densidad_vegetacion; // Factor de densidad (0.5 - 1.5)
    float velocidad_viento;    // Velocidad del viento (0 - 20 m/s)
    int direccion_viento;      // Dirección del viento (0-7, según vecindad Moore)
    float humedad;             // Humedad relativa (0 - 100%)
    float temperatura;         // Temperatura ambiente (°C)
} ParametrosAmbientales;

// Estructura del terreno
typedef struct {
    int filas;
    int columnas;
    int *datos;               // Estados de las celdas
    float *densidad;          // Densidad de vegetación por celda
    float *pendiente;         // Pendiente del terreno por celda
} Terreno;

// Función para calcular el factor de viento según la dirección
__device__ float calcular_factor_viento(int dir_celda, int dir_viento, float velocidad) {
    int diff = abs(dir_celda - dir_viento);
    if (diff > 4) diff = 8 - diff;
    
    float factor = 1.0f;
    if (diff == 0) {
        factor = 1.0f + (velocidad * 0.05f);
    } else if (diff == 4) {
        factor = 1.0f - (velocidad * 0.03f);
    } else {
        factor = 1.0f + (velocidad * 0.02f * (4 - diff) / 4.0f);
    }
    
    return fmaxf(0.1f, fminf(factor, 3.0f));
}

// Función para calcular el factor de humedad
__device__ float calcular_factor_humedad(float humedad) {
    if (humedad > 70.0f) {
        return 0.3f;
    } else if (humedad > 50.0f) {
        return 0.6f;
    } else if (humedad > 30.0f) {
        return 0.9f;
    } else {
        return 1.2f;
    }
}

// Función para calcular el factor de temperatura
__device__ float calcular_factor_temperatura(float temperatura) {
    if (temperatura > 35.0f) {
        return 1.3f;
    } else if (temperatura > 25.0f) {
        return 1.1f;
    } else if (temperatura > 15.0f) {
        return 1.0f;
    } else {
        return 0.8f;
    }
}

// Función para calcular el factor de pendiente
__device__ float calcular_factor_pendiente(float pendiente_origen, float pendiente_destino) {
    float diferencia = pendiente_destino - pendiente_origen;
    
    if (diferencia > 0) {
        return 1.0f + (diferencia * 0.1f);
    } else {
        return 1.0f + (diferencia * 0.05f);
    }
}

// Kernel para simular un paso de tiempo del incendio
__global__ void paso_simulacion_kernel(int *terreno_actual, int *terreno_siguiente,
                                       float *densidad, float *pendiente,
                                       ParametrosAmbientales params,
                                       int filas, int columnas,
                                       unsigned long long seed, int tiempo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = filas * columnas;
    
    if (idx >= total) return;
    
    int i = idx / columnas;
    int j = idx % columnas;
    
    curandState state;
    curand_init(seed + idx + tiempo * total, 0, 0, &state);
    
    terreno_siguiente[idx] = terreno_actual[idx];
    
    if (terreno_actual[idx] == QUEMADA) {
        return;
    }
    
    if (terreno_actual[idx] == EN_COMBUSTION) {
        terreno_siguiente[idx] = QUEMADA;
        return;
    }
    
    if (terreno_actual[idx] == VEGETACION_SANA) {
        bool vecino_ardiendo = false;
        float prob_max = 0.0f;
        
        for (int dir = 0; dir < 8; dir++) {
            int ni = i + dx[dir];
            int nj = j + dy[dir];
            
            if (ni >= 0 && ni < filas && nj >= 0 && nj < columnas) {
                int n_idx = ni * columnas + nj;
                
                if (terreno_actual[n_idx] == EN_COMBUSTION) {
                    vecino_ardiendo = true;
                    
                    float P_final = params.P_base;
                    float F_vegetacion = densidad[idx];
                    float F_viento = calcular_factor_viento(dir, params.direccion_viento, 
                                                           params.velocidad_viento);
                    float F_humedad = calcular_factor_humedad(params.humedad);
                    float F_temperatura = calcular_factor_temperatura(params.temperatura);
                    float F_pendiente = calcular_factor_pendiente(pendiente[n_idx], 
                                                                  pendiente[idx]);
                    
                    P_final = P_final * F_vegetacion * F_viento * F_humedad * 
                              F_temperatura * F_pendiente;
                    
                    P_final = fminf(P_final, 0.95f);
                    P_final = fmaxf(P_final, 0.0f);
                    
                    if (P_final > prob_max) {
                        prob_max = P_final;
                    }
                }
            }
        }
        
        if (vecino_ardiendo) {
            float random = curand_uniform(&state);
            if (random < prob_max) {
                terreno_siguiente[idx] = EN_COMBUSTION;
            }
        }
    }
}

// Kernel para contar celdas quemadas
__global__ void contar_quemadas_kernel(int *terreno, int *contador, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total && (terreno[idx] == QUEMADA || terreno[idx] == EN_COMBUSTION)) {
        atomicAdd(&contador[idx], 1);
    }
}

// Generar parámetros aleatorios para una simulación
ParametrosAmbientales generar_parametros_aleatorios() {
    ParametrosAmbientales params;
    
    params.P_base = 0.3f + ((float)rand() / RAND_MAX) * 0.4f;
    params.densidad_vegetacion = 0.7f + ((float)rand() / RAND_MAX) * 0.6f;
    params.velocidad_viento = ((float)rand() / RAND_MAX) * 15.0f;
    params.direccion_viento = rand() % 8;
    params.humedad = 20.0f + ((float)rand() / RAND_MAX) * 60.0f;
    params.temperatura = 15.0f + ((float)rand() / RAND_MAX) * 25.0f;
    
    return params;
}

// Función para simular el incendio en GPU
void simular_incendio_gpu(Terreno *terreno_inicial, ParametrosAmbientales params,
                          int T_max, int *resultado_final) {
    int total_celdas = terreno_inicial->filas * terreno_inicial->columnas;
    
    int *d_terreno_actual, *d_terreno_siguiente;
    float *d_densidad, *d_pendiente;
    
    cudaMalloc(&d_terreno_actual, total_celdas * sizeof(int));
    cudaMalloc(&d_terreno_siguiente, total_celdas * sizeof(int));
    cudaMalloc(&d_densidad, total_celdas * sizeof(float));
    cudaMalloc(&d_pendiente, total_celdas * sizeof(float));
    
    cudaMemcpy(d_terreno_actual, terreno_inicial->datos, 
               total_celdas * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_densidad, terreno_inicial->densidad,
               total_celdas * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pendiente, terreno_inicial->pendiente,
               total_celdas * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_celdas + threadsPerBlock - 1) / threadsPerBlock;
    
    for (int t = 0; t < T_max; t++) {
        unsigned long long seed = (unsigned long long)time(NULL) + t;
        
        paso_simulacion_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_terreno_actual, d_terreno_siguiente,
            d_densidad, d_pendiente, params,
            terreno_inicial->filas, terreno_inicial->columnas,
            seed, t
        );
        cudaDeviceSynchronize();
        
        int *temp = d_terreno_actual;
        d_terreno_actual = d_terreno_siguiente;
        d_terreno_siguiente = temp;
    }
    
    cudaMemcpy(resultado_final, d_terreno_actual,
               total_celdas * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_terreno_actual);
    cudaFree(d_terreno_siguiente);
    cudaFree(d_densidad);
    cudaFree(d_pendiente);
}

// Algoritmo Monte Carlo principal
float** montecarlo_incendios(Terreno *terreno_inicial, int N_simulaciones, int T_max) {
    int filas = terreno_inicial->filas;
    int columnas = terreno_inicial->columnas;
    int total_celdas = filas * columnas;
    
    float **matriz_probabilidad = (float**)malloc(filas * sizeof(float*));
    for (int i = 0; i < filas; i++) {
        matriz_probabilidad[i] = (float*)calloc(columnas, sizeof(float));
    }
    
    int *d_contador;
    cudaMalloc(&d_contador, total_celdas * sizeof(int));
    cudaMemset(d_contador, 0, total_celdas * sizeof(int));
    
    int *h_resultado = (int*)malloc(total_celdas * sizeof(int));
    int *d_resultado;
    cudaMalloc(&d_resultado, total_celdas * sizeof(int));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_celdas + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Ejecutando %d simulaciones Monte Carlo...\n", N_simulaciones);
    
    for (int sim = 0; sim < N_simulaciones; sim++) {
        ParametrosAmbientales params = generar_parametros_aleatorios();
        
        simular_incendio_gpu(terreno_inicial, params, T_max, h_resultado);
        
        cudaMemcpy(d_resultado, h_resultado, total_celdas * sizeof(int),
                   cudaMemcpyHostToDevice);
        
        contar_quemadas_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_resultado, d_contador, total_celdas
        );
        cudaDeviceSynchronize();
        
        if ((sim + 1) % 10 == 0) {
            printf("Progreso: %d/%d simulaciones (%.1f%%)\n", 
                   sim + 1, N_simulaciones, 
                   (sim + 1) * 100.0f / N_simulaciones);
        }
    }
    
    int *h_contador = (int*)malloc(total_celdas * sizeof(int));
    cudaMemcpy(h_contador, d_contador, total_celdas * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            int idx = i * columnas + j;
            matriz_probabilidad[i][j] = (float)h_contador[idx] / N_simulaciones;
        }
    }
    
    cudaFree(d_contador);
    cudaFree(d_resultado);
    free(h_resultado);
    free(h_contador);
    
    return matriz_probabilidad;
}

// Función para inicializar terreno con valores realistas
Terreno* crear_terreno(int filas, int columnas) {
    Terreno *terreno = (Terreno*)malloc(sizeof(Terreno));
    terreno->filas = filas;
    terreno->columnas = columnas;
    
    int total = filas * columnas;
    terreno->datos = (int*)malloc(total * sizeof(int));
    terreno->densidad = (float*)malloc(total * sizeof(float));
    terreno->pendiente = (float*)malloc(total * sizeof(float));
    
    for (int i = 0; i < total; i++) {
        terreno->datos[i] = VEGETACION_SANA;
        terreno->densidad[i] = 0.7f + ((float)rand() / RAND_MAX) * 0.5f;
        terreno->pendiente[i] = -15.0f + ((float)rand() / RAND_MAX) * 30.0f;
    }
    
    int centro = (filas / 2) * columnas + (columnas / 2);
    terreno->datos[centro] = EN_COMBUSTION;
    
    return terreno;
}

void liberar_terreno(Terreno *t) {
    free(t->datos);
    free(t->densidad);
    free(t->pendiente);
    free(t);
}

void guardar_resultados(float **probabilidades, int filas, int columnas, 
                        const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        printf("Error al abrir archivo %s\n", filename);
        return;
    }
    
    fprintf(f, "i,j,probabilidad\n");
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            fprintf(f, "%d,%d,%.4f\n", i, j, probabilidades[i][j]);
        }
    }
    
    fclose(f);
    printf("Resultados guardados en %s\n", filename);
}

void imprimir_mapa_calor(float **probabilidades, int filas, int columnas) {
    printf("\n=== MAPA DE RIESGO DE INCENDIO ===\n");
    printf("Leyenda: . (0-20%%) - (20-40%%) + (40-60%%) * (60-80%%) # (80-100%%)\n\n");
    
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            float p = probabilidades[i][j];
            char c;
            if (p < 0.2f) c = '.';
            else if (p < 0.4f) c = '-';
            else if (p < 0.6f) c = '+';
            else if (p < 0.8f) c = '*';
            else c = '#';
            printf("%c ", c);
        }
        printf("\n");
    }
}

// Función para guardar imagen en formato PPM (no necesita librerías)
void guardar_imagen_ppm(float **probabilidades, int filas, int columnas, 
                        const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Error al crear archivo %s\n", filename);
        return;
    }
    
    int cell_size = 10;
    int width = columnas * cell_size;
    int height = filas * cell_size;
    
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    
    for (int i = 0; i < filas; i++) {
        for (int pi = 0; pi < cell_size; pi++) {
            for (int j = 0; j < columnas; j++) {
                unsigned char r, g, b;
                float prob = probabilidades[i][j];
                
                if (prob < 0.25f) {
                    float t = prob / 0.25f;
                    r = 0;
                    g = (unsigned char)(255 * t);
                    b = (unsigned char)(255 * (1 - t));
                } else if (prob < 0.5f) {
                    float t = (prob - 0.25f) / 0.25f;
                    r = (unsigned char)(255 * t);
                    g = 255;
                    b = 0;
                } else if (prob < 0.75f) {
                    float t = (prob - 0.5f) / 0.25f;
                    r = 255;
                    g = (unsigned char)(255 * (1 - t * 0.5f));
                    b = 0;
                } else {
                    float t = (prob - 0.75f) / 0.25f;
                    r = 255;
                    g = (unsigned char)(128 * (1 - t));
                    b = 0;
                }
                
                for (int pj = 0; pj < cell_size; pj++) {
                    fwrite(&r, 1, 1, f);
                    fwrite(&g, 1, 1, f);
                    fwrite(&b, 1, 1, f);
                }
            }
        }
    }
    
    fclose(f);
    printf("Mapa de calor guardado en: %s\n", filename);
}

// Función para guardar imagen con leyenda en formato PPM
void guardar_imagen_con_leyenda(float **probabilidades, int filas, int columnas,
                                 const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) {
        printf("Error al crear archivo %s\n", filename);
        return;
    }
    
    int cell_size = 10;
    int map_width = columnas * cell_size;
    int map_height = filas * cell_size;
    int legend_width = 80;
    int margin = 20;
    
    int total_width = map_width + legend_width + margin * 3;
    int total_height = map_height + margin * 2;
    
    fprintf(f, "P6\n%d %d\n255\n", total_width, total_height);
    
    unsigned char *line = (unsigned char*)malloc(total_width * 3);
    
    for (int y = 0; y < total_height; y++) {
        memset(line, 255, total_width * 3);
        
        if (y >= margin && y < margin + map_height) {
            int map_y = (y - margin) / cell_size;
            
            for (int x = margin; x < margin + map_width; x++) {
                int map_x = (x - margin) / cell_size;
                float prob = probabilidades[map_y][map_x];
                
                unsigned char r, g, b;
                if (prob < 0.25f) {
                    float t = prob / 0.25f;
                    r = 0;
                    g = (unsigned char)(255 * t);
                    b = (unsigned char)(255 * (1 - t));
                } else if (prob < 0.5f) {
                    float t = (prob - 0.25f) / 0.25f;
                    r = (unsigned char)(255 * t);
                    g = 255;
                    b = 0;
                } else if (prob < 0.75f) {
                    float t = (prob - 0.5f) / 0.25f;
                    r = 255;
                    g = (unsigned char)(255 * (1 - t * 0.5f));
                    b = 0;
                } else {
                    float t = (prob - 0.75f) / 0.25f;
                    r = 255;
                    g = (unsigned char)(128 * (1 - t));
                    b = 0;
                }
                
                line[x * 3 + 0] = r;
                line[x * 3 + 1] = g;
                line[x * 3 + 2] = b;
            }
            
            int legend_x = margin * 2 + map_width;
            int legend_bar_width = 30;
            
            if (y >= margin && y < margin + map_height) {
                float prob = 1.0f - (float)(y - margin) / map_height;
                
                unsigned char r, g, b;
                if (prob < 0.25f) {
                    float t = prob / 0.25f;
                    r = 0;
                    g = (unsigned char)(255 * t);
                    b = (unsigned char)(255 * (1 - t));
                } else if (prob < 0.5f) {
                    float t = (prob - 0.25f) / 0.25f;
                    r = (unsigned char)(255 * t);
                    g = 255;
                    b = 0;
                } else if (prob < 0.75f) {
                    float t = (prob - 0.5f) / 0.25f;
                    r = 255;
                    g = (unsigned char)(255 * (1 - t * 0.5f));
                    b = 0;
                } else {
                    float t = (prob - 0.75f) / 0.25f;
                    r = 255;
                    g = (unsigned char)(128 * (1 - t));
                    b = 0;
                }
                
                for (int x = legend_x; x < legend_x + legend_bar_width; x++) {
                    line[x * 3 + 0] = r;
                    line[x * 3 + 1] = g;
                    line[x * 3 + 2] = b;
                }
            }
        }
        
        fwrite(line, 1, total_width * 3, f);
    }
    
    free(line);
    fclose(f);
    printf("Mapa con leyenda guardado en: %s\n", filename);
}

int main() {
    srand(time(NULL));
    
    int filas = 50;
    int columnas = 50;
    int N_simulaciones = 100;
    int T_max = 50;
    
    printf("=== SIMULADOR DE PROPAGACION DE INCENDIOS FORESTALES ===\n");
    printf("Dimensiones del terreno: %dx%d\n", filas, columnas);
    printf("Numero de simulaciones: %d\n", N_simulaciones);
    printf("Pasos de tiempo: %d\n\n", T_max);
    
    Terreno *terreno = crear_terreno(filas, columnas);
    
    clock_t inicio = clock();
    float **probabilidades = montecarlo_incendios(terreno, N_simulaciones, T_max);
    clock_t fin = clock();
    
    double tiempo_total = (double)(fin - inicio) / CLOCKS_PER_SEC;
    printf("\nTiempo total de ejecucion: %.2f segundos\n", tiempo_total);
    
    imprimir_mapa_calor(probabilidades, filas, columnas);
    
    guardar_resultados(probabilidades, filas, columnas, "resultados_incendio.csv");
    
    printf("\nGenerando imagenes...\n");
    guardar_imagen_ppm(probabilidades, filas, columnas, "mapa_calor.ppm");
    guardar_imagen_con_leyenda(probabilidades, filas, columnas, "mapa_calor_leyenda.ppm");
    
    float max_prob = 0.0f, min_prob = 1.0f, sum_prob = 0.0f;
    int celdas_alto_riesgo = 0;
    
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < columnas; j++) {
            float p = probabilidades[i][j];
            if (p > max_prob) max_prob = p;
            if (p < min_prob) min_prob = p;
            sum_prob += p;
            if (p > 0.7f) celdas_alto_riesgo++;
        }
    }
    
    printf("\n=== ESTADISTICAS ===\n");
    printf("Probabilidad maxima: %.2f%%\n", max_prob * 100);
    printf("Probabilidad minima: %.2f%%\n", min_prob * 100);
    printf("Probabilidad promedio: %.2f%%\n", 
           (sum_prob / (filas * columnas)) * 100);
    printf("Celdas de alto riesgo (>70%%): %d (%.1f%%)\n", 
           celdas_alto_riesgo,
           celdas_alto_riesgo * 100.0f / (filas * columnas));
    
    printf("\nNOTA: Las imagenes estan en formato PPM.\n");
    printf("Para convertir a PNG usa: magick mapa_calor.ppm mapa_calor.png\n");
    printf("O abrelas directamente con GIMP, IrfanView, o cualquier visor de imagenes.\n");
    
    liberar_terreno(terreno);
    for (int i = 0; i < filas; i++) {
        free(probabilidades[i]);
    }
    free(probabilidades);
    
    return 0;
}