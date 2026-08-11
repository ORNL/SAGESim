# Frontier Environment Setup — ROCm 7.2.0 + CuPy 14.0.1

Instructions for setting up a Python 3.11 conda environment on Frontier with ROCm 7.2.0 and CuPy 14.0.1 (HIP backend).

## 1. Load Modules

```bash
module reset
module load cpe/26.03
module load PrgEnv-gnu
module load miniforge3
module load rocm/7.2.0
module load craype-accel-amd-gfx90a
```

## 2. Create Conda Environment

```bash
conda create --name sage311_rocm720_cupy1401 python=3.11
source activate sage311_rocm720_cupy1401
```

## 3. Install mpi4py

Build mpi4py from source using the Cray compiler wrappers:

```bash
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
```

## 4. Install CuPy with HIP Support

Set the ROCm/HIP build variables and compile CuPy from source:

```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=${ROCM_PATH}
export HCC_AMDGPU_TARGET=gfx90a
export CUPY_GPU_ARCH=gfx90a
export CUPY_HIP_TARGET=gfx90a
CC=gcc CXX=g++ pip install --no-cache-dir --no-binary=cupy cupy==14.0.1
```

## 5. Install SAGESim and superneuroabm

Install each package from its respective repository root:

```bash
# Inside the SAGESim repo
cd /path/to/SAGESim
pip install .

# Inside the superneuroabm repo
cd /path/to/superneuroabm
pip install .
```
