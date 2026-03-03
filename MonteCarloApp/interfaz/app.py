import streamlit as st
import subprocess
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import psutil

# ===================== CONFIGURACION =====================
st.set_page_config(
    page_title="Simulación Monte Carlo — Incendios",
    layout="wide"
)

st.title("Simulador Monte Carlo de propagación de incendios")

# ===================== SIDEBAR =====================
st.sidebar.header("Configuración del escenario")
filas = st.sidebar.number_input("Filas del terreno", 10, 500, 50)
columnas = st.sidebar.number_input("Columnas del terreno", 10, 500, 50)
simulaciones = st.sidebar.number_input("Número de simulaciones", 10, 100000, 500)
tmax = st.sidebar.number_input("Tiempo máximo de propagación (TMAX)", 1, 500, 50)
prob = st.sidebar.slider("Probabilidad base de propagación", 0.0, 1.0, 0.3)

st.sidebar.header("Punto de inicio del fuego")
inicio_x = st.sidebar.number_input("Fila de inicio (X)", 0, int(filas - 1), int(filas // 2))
inicio_y = st.sidebar.number_input("Columna de inicio (Y)", 0, int(columnas - 1), int(columnas // 2))

st.sidebar.header("Factores ambientales")
st.sidebar.caption("Valores > 1.0 aumentan propagación, < 1.0 la reducen")
f_viento = st.sidebar.slider("Factor Viento", 0.5, 2.0, 1.0, 0.1)
f_vegetacion = st.sidebar.slider("Factor Vegetación", 0.5, 2.0, 1.0, 0.1)
f_humedad = st.sidebar.slider("Factor Humedad", 0.5, 1.5, 1.0, 0.1)
f_temperatura = st.sidebar.slider("Factor Temperatura", 0.5, 2.0, 1.0, 0.1)
f_pendiente = st.sidebar.slider("Factor Pendiente", 0.5, 1.5, 1.0, 0.1)

metodo = st.sidebar.selectbox(
    "Método de cómputo",
    ["Secuencial C++", "OpenMP C++", "CUDA", "MPI"]
)

# ===================== RUTAS =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(BASE_DIR, "simulaciones")
RESULT_DIR = os.path.join(SIM_DIR, "resultados")
os.makedirs(RESULT_DIR, exist_ok=True)

# ===================== FUNCIONES =====================

def parse_line(line: str):
    line = line.strip()
    if not line:
        return None
    try:
        if ',' in line:
            return [float(x) for x in line.split(',')]
        else:
            return [float(x) for x in line.split()]
    except ValueError:
        return None


def parsear_info_cuda(stderr_text: str) -> dict:
    """Lee las líneas clave:valor que cuda.exe emite por stderr."""
    info = {}
    for linea in stderr_text.splitlines():
        linea = linea.strip()
        if ':' in linea:
            clave, _, valor = linea.partition(':')
            info[clave.strip()] = valor.strip()
    return info


def ejecutar_simulacion():
    if metodo in ["Secuencial C++", "OpenMP C++", "CUDA"]:
        exe_map = {
            "Secuencial C++": "secuencial.exe",
            "OpenMP C++":     "openmp.exe",
            "CUDA":           "cuda.exe"
        }
        exe = os.path.join(SIM_DIR, exe_map[metodo])
        if not os.path.exists(exe):
            st.error(f"Ejecutable {metodo} no encontrado en {exe}")
            return None, None, None, None, None
        comando = [exe, str(filas), str(columnas), str(simulaciones), str(tmax), str(prob), str(inicio_x), str(inicio_y), str(f_viento), str(f_vegetacion), str(f_humedad), str(f_temperatura), str(f_pendiente)]
    else:
        script = os.path.join(SIM_DIR, "mpi_python.py")
        if not os.path.exists(script):
            st.error(f"Script MPI Python no encontrado en {script}")
            return None, None, None, None, None
        comando = ["python", script, str(filas), str(columnas), str(simulaciones), str(tmax), str(prob)]

    inicio = time.time()
    proceso = subprocess.run(comando, capture_output=True, text=True)
    tiempo = time.time() - inicio
    mem_used = psutil.Process().memory_info().rss / 1024**2
    cpu_used = psutil.cpu_percent(interval=1)

    # Parsear CSV de stdout
    matriz = []
    for linea in proceso.stdout.splitlines():
        # Ignorar la línea TIEMPO que imprime cuda.exe
        if linea.startswith("TIEMPO"):
            continue
        fila_parsed = parse_line(linea)
        if fila_parsed:
            matriz.append(fila_parsed)

    if not matriz:
        st.error("No se generó salida numérica del ejecutable.")
        st.text("stdout: " + proceso.stdout[:500])
        st.text("stderr: " + proceso.stderr[:500])
        return None, None, None, None, None

    df = pd.DataFrame(matriz)

    # Info CUDA solo cuando corresponde
    cuda_info = None
    if metodo == "CUDA" and proceso.stderr:
        cuda_info = parsear_info_cuda(proceso.stderr)

    return df, tiempo, mem_used, cpu_used, cuda_info


# ===================== SIMULACION ACTUAL =====================
st.divider()
st.header("Simulación actual")
df_actual, tiempo, mem_used, cpu_used, cuda_info = None, None, None, None, None

if st.button("Ejecutar simulación"):
    with st.spinner("Ejecutando modelo Monte Carlo..."):
        df_actual, tiempo, mem_used, cpu_used, cuda_info = ejecutar_simulacion()

    if df_actual is not None:

        # ── Métricas generales ──────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo de ejecución", f"{tiempo:.2f} s")
        col2.metric("Memoria usada",       f"{mem_used:.2f} MB")
        col3.metric("Uso de CPU",          f"{cpu_used:.2f} %")

        # ── Info CUDA (solo si el método es CUDA) ──────────────────────
        if metodo == "CUDA" and cuda_info:
            st.subheader("Información de GPU (CUDA)")

            g1, g2, g3 = st.columns(3)
            g1.metric("GPU",              cuda_info.get("GPU_NOMBRE", "—"))
            g2.metric("Memoria GPU",      f"{cuda_info.get('GPU_MEMORIA_MB', '—')} MB")
            g3.metric("Multiprocesadores (SM)", cuda_info.get("GPU_MULTIPROCESADORES", "—"))

            p1, p2, p3 = st.columns(3)
            p1.metric("Threads por bloque", cuda_info.get("CUDA_THREADS_POR_BLOQUE", "—"))
            p2.metric("Bloques lanzados",   cuda_info.get("CUDA_BLOQUES", "—"))
            p3.metric("Threads totales GPU",cuda_info.get("CUDA_THREADS_TOTAL", "—"))

            max_threads_sm = cuda_info.get("GPU_MAX_THREADS_SM", None)
            if max_threads_sm:
                st.caption(f"Máx. threads por SM: {max_threads_sm}")

        # ── Mapa de calor ───────────────────────────────────────────────
        st.subheader("Mapa de probabilidad de incendio")
        data = df_actual.values.astype(float)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Probabilidad de incendio")
        ax.set_title(f"Distribución espacial ({metodo})")
        ax.set_xlabel("Columnas")
        ax.set_ylabel("Filas")
        plt.tight_layout()
        st.pyplot(fig)

        # ── Descargas ───────────────────────────────────────────────────
        archivo_prob = os.path.join(RESULT_DIR, f"{metodo.replace(' ','_')}_probabilidades.xlsx")
        df_actual.to_excel(archivo_prob, index=False, header=False)
        with open(archivo_prob, "rb") as f:
            st.download_button("⬇ Descargar Probabilidades XLSX", f,
                               file_name=os.path.basename(archivo_prob))

        archivo_metrics = os.path.join(RESULT_DIR, f"{metodo.replace(' ','_')}_metricas.xlsx")
        df_metrics = pd.DataFrame([{
            "Método":      metodo,
            "Tiempo(s)":   tiempo,
            "Memoria(MB)": mem_used,
            "CPU(%)":      cpu_used
        }])
        df_metrics.to_excel(archivo_metrics, index=False)
        with open(archivo_metrics, "rb") as f:
            st.download_button("⬇ Descargar Métricas XLSX", f,
                               file_name=os.path.basename(archivo_metrics))

# ===================== COMPARATIVA =====================
st.divider()
st.header("Comparativa de resultados")

tipo_comparacion = st.radio(
    "Selecciona tipo de comparativa:",
    ["Gráfico de líneas de métricas", "Mapas de calor lado a lado"]
)

# ── Gráfico de líneas de métricas ───────────────────────────────────────
if tipo_comparacion == "Gráfico de líneas de métricas":
    archivos_metricas = st.file_uploader(
        "Carga uno o varios archivos XLSX/CSV de métricas para comparar",
        accept_multiple_files=True,
        key="metricas_uploader"
    )

    if archivos_metricas:
        df_metricas_list = []
        for af in archivos_metricas:
            try:
                dfm = pd.read_excel(af) if af.name.endswith(".xlsx") else pd.read_csv(af)
                dfm.columns = [c.strip() for c in dfm.columns]
                df_metricas_list.append(dfm)
            except Exception as e:
                st.warning(f"No se pudo procesar {af.name}: {e}")

        if df_metricas_list:
            df_metricas_total = pd.concat(df_metricas_list, ignore_index=True)
            st.subheader("Métricas totales")
            st.dataframe(df_metricas_total, use_container_width=True)

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            for col in ["Tiempo(s)", "Memoria(MB)", "CPU(%)"]:
                if col in df_metricas_total.columns:
                    ax2.plot(df_metricas_total["Método"], df_metricas_total[col], marker="o", label=col)
            ax2.set_title("Comparativa de métricas por método")
            ax2.set_xlabel("Método")
            ax2.set_ylabel("Valor")
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            st.pyplot(fig2)
    else:
        st.info("Sube archivos de métricas para ver la comparativa.")

# ── Mapas de calor lado a lado ──────────────────────────────────────────
if tipo_comparacion == "Mapas de calor lado a lado":
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        archivo_prob_1 = st.file_uploader(
            "Archivo de probabilidades 1",
            type=["xlsx", "csv"],
            key="prob_1"
        )
    with col_up2:
        archivo_prob_2 = st.file_uploader(
            "Archivo de probabilidades 2",
            type=["xlsx", "csv"],
            key="prob_2"
        )

    if archivo_prob_1 and archivo_prob_2:
        try:
            df_1 = (
                pd.read_excel(archivo_prob_1, header=None)
                if archivo_prob_1.name.endswith(".xlsx")
                else pd.read_csv(archivo_prob_1, header=None)
            )
            df_2 = (
                pd.read_excel(archivo_prob_2, header=None)
                if archivo_prob_2.name.endswith(".xlsx")
                else pd.read_csv(archivo_prob_2, header=None)
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader(archivo_prob_1.name)
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                im1 = ax1.imshow(df_1.values.astype(float),
                                 cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=1)
                plt.colorbar(im1, ax=ax1)
                st.pyplot(fig1)

            with col2:
                st.subheader(archivo_prob_2.name)
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                im2 = ax2.imshow(df_2.values.astype(float),
                                 cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=1)
                plt.colorbar(im2, ax=ax2)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"No se pudo procesar los archivos: {e}")
    else:
        st.info("Sube dos archivos de probabilidades para comparar los mapas de calor.")