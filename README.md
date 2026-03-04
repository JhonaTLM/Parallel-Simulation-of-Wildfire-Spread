# Parallel Simulation of Wildfire Spread (Monte Carlo)

This project is a wildfire propagation simulator that utilizes the **Monte Carlo Method** to predict fire advancement under various environmental conditions. Developed as part of the **Concurrent and Parallel Programming** course at **UNMSM**.

The system evaluates and compares the performance of sequential implementations against parallel computing paradigms such as **OpenMP**, **CUDA** (NVIDIA GPU), and **MPI**.

## Project Structure

* **`MonteCarloApp/interfaz/app.py`**: Modern graphical user interface developed in Python with Streamlit.
* **`MonteCarloApp/simulaciones/`**: Directory containing the computation engines (executables) and results.
* **`Paralelo/`**: Source code for CUDA and MPI implementations.
* **`Secuencial/`**: Source code for the base Sequential and OpenMP implementations.

## Requirements

* **Python 3.x**: For the web interface.
* **GCC/G++**: C++ compiler with OpenMP support.
* **NVIDIA CUDA Toolkit**: Required for GPU acceleration.
* **MS-MPI / OpenMPI**: For cluster or multi-process execution.

## Installation & Setup

### 1. Install Python Dependencies

python -m pip install streamlit pandas matplotlib psutil 

### 2. Running the Application

cd MonteCarloApp/interfaz
streamlit run app.py
