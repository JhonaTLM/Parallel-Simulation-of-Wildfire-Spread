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
st.sidebar.header("⚙ Configuración del escenario")
filas = st.sidebar.number_input("Filas del terreno", 10, 500, 50)
columnas = st.sidebar.number_input("Columnas del terreno", 10, 500, 50)
simulaciones = st.sidebar.number_input("Número de simulaciones", 10, 100000, 500)
tmax = st.sidebar.number_input("Tiempo máximo de propagación (TMAX)", 1, 500, 50)
prob = st.sidebar.slider("Probabilidad base de propagación", 0.0, 1.0, 0.3)

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

def ejecutar_simulacion():
    # Preparar comando según método
    if metodo in ["Secuencial C++", "OpenMP C++", "CUDA"]:
        exe_map = {
            "Secuencial C++": "secuencial.exe",
            "OpenMP C++": "openmp.exe",
            "CUDA": "cuda.exe"
        }
        exe = os.path.join(SIM_DIR, exe_map[metodo])
        if not os.path.exists(exe):
            st.error(f"Ejecutable {metodo} no encontrado en {exe}")
            return None, None, None, None
        comando = [exe, str(filas), str(columnas), str(simulaciones), str(tmax), str(prob)]
    else:
        script = os.path.join(SIM_DIR, "mpi_python.py")
        if not os.path.exists(script):
            st.error(f"Script MPI Python no encontrado en {script}")
            return None, None, None, None
        comando = ["python", script, str(filas), str(columnas), str(simulaciones), str(tmax), str(prob)]

    # Medir tiempo y recursos
    inicio = time.time()
    proceso = subprocess.run(comando, capture_output=True, text=True)
    tiempo = time.time() - inicio
    mem_used = psutil.Process().memory_info().rss / 1024**2
    cpu_used = psutil.cpu_percent(interval=1)

    # Parsear salida numérica
    matriz = []
    for linea in proceso.stdout.splitlines():
        fila = parse_line(linea)
        if fila:
            matriz.append(fila)

    if not matriz:
        st.error("No se generó salida numérica del ejecutable.")
        return None, None, None, None

    df = pd.DataFrame(matriz)
    return df, tiempo, mem_used, cpu_used

# ===================== SIMULACION ACTUAL =====================
st.divider()
st.header("Simulación actual")
df_actual, tiempo, mem_used, cpu_used = None, None, None, None

if st.button("Ejecutar simulación"):
    with st.spinner("Ejecutando modelo Monte Carlo..."):
        df_actual, tiempo, mem_used, cpu_used = ejecutar_simulacion()

    if df_actual is not None:
        # Métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo de ejecución", f"{tiempo:.2f} s")
        col2.metric("Memoria usada", f"{mem_used:.2f} MB")
        col3.metric("Uso de CPU", f"{cpu_used:.2f} %")

        # ================= MAPA DE CALOR =================
        st.subheader("Mapa de probabilidad de incendio")
        data = df_actual.values.astype(float)
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(data, cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Probabilidad de incendio")
        ax.set_title(f"Distribución espacial ({metodo})")
        ax.set_xlabel("Columnas")
        ax.set_ylabel("Filas")
        plt.tight_layout()
        st.pyplot(fig)

        # ================= DESCARGAS =================
        # Probabilidades
        archivo_prob = os.path.join(RESULT_DIR, f"{metodo.replace(' ','_')}_probabilidades.xlsx")
        df_actual.to_excel(archivo_prob, index=False, header=False)
        with open(archivo_prob, "rb") as f:
            st.download_button("⬇ Descargar Probabilidades XLSX", f, file_name=os.path.basename(archivo_prob))

        # Métricas
        archivo_metrics = os.path.join(RESULT_DIR, f"{metodo.replace(' ','_')}_metricas.xlsx")
        df_metrics = pd.DataFrame([{
            "Método": metodo,
            "Tiempo(s)": tiempo,
            "Memoria(MB)": mem_used,
            "CPU(%)": cpu_used
        }])
        df_metrics.to_excel(archivo_metrics, index=False)
        with open(archivo_metrics, "rb") as f:
            st.download_button("⬇ Descargar Métricas XLSX", f, file_name=os.path.basename(archivo_metrics))

# ===================== COMPARATIVA =====================
st.divider()
st.header("Comparativa de resultados")
tipo_comparacion = st.radio("Selecciona tipo de comparativa:", ["Gráfico de líneas de métricas", "Mapas de calor lado a lado"])

# -------------------- LINEAS --------------------
archivos_metricas = st.file_uploader("Carga uno o varios archivos XLSX/CSV de métricas", accept_multiple_files=True)

df_metricas_list = []

# Agregar la simulación actual si existe
if df_actual is not None:
    df_metricas_list.append(pd.DataFrame([{
        "Método": metodo,
        "Tiempo(s)": tiempo,
        "Memoria(MB)": mem_used,
        "CPU(%)": cpu_used
    }]))

# Procesar archivos subidos
if archivos_metricas:
    for af in archivos_metricas:
        try:
            if af.name.endswith(".xlsx"):
                dfm = pd.read_excel(af)
            else:
                dfm = pd.read_csv(af)
            dfm.columns = [c.strip().capitalize() for c in dfm.columns]
            if all(col in dfm.columns for col in ["Método","Tiempo(s)","Memoria(mb)","Cpu(%)"]):
                df_metricas_list.append(dfm)
        except Exception as e:
            st.warning(f"No se pudo procesar {af.name}: {e}")

# Graficar comparativa de lineas
if df_metricas_list and tipo_comparacion == "Gráfico de líneas de métricas":
    df_metricas_total = pd.concat(df_metricas_list, ignore_index=True)
    st.subheader("📊 Métricas totales")
    st.dataframe(df_metricas_total, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    for col in ["Tiempo(s)","Memoria(MB)","CPU(%)"]:
        if col in df_metricas_total.columns:
            ax2.plot(df_metricas_total["Método"], df_metricas_total[col], marker="o", label=col)
    ax2.set_title("Comparativa de métricas por método")
    ax2.set_xlabel("Método")
    ax2.set_ylabel("Valor")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)

# -------------------- MAPAS DE CALOR --------------------
if tipo_comparacion=="Mapas de calor lado a lado":
    archivo_prob_subido = st.file_uploader("Carga archivo XLSX/CSV de probabilidades para comparar", type=["xlsx","csv"])
    if df_actual is not None and archivo_prob_subido:
        try:
            if archivo_prob_subido.name.endswith(".xlsx"):
                df_externo = pd.read_excel(archivo_prob_subido, header=None)
            else:
                df_externo = pd.read_csv(archivo_prob_subido, header=None)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Simulación actual")
                fig1, ax1 = plt.subplots(figsize=(6,6))
                im1 = ax1.imshow(df_actual.values.astype(float), cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=1)
                plt.colorbar(im1, ax=ax1)
                st.pyplot(fig1)

            with col2:
                st.subheader("Archivo cargado")
                fig2, ax2 = plt.subplots(figsize=(6,6))
                im2 = ax2.imshow(df_externo.values.astype(float), cmap="inferno", origin="lower", aspect="auto", vmin=0, vmax=1)
                plt.colorbar(im2, ax=ax2)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"No se pudo procesar el archivo subido: {e}")