# ==========================================
# HPC N-Body Project Makefile
# ==========================================

SRC_DIR = src
BIN_DIR = bin
REP_DIR = vtune_reports

CXX      = g++
MPICXX   = mpicxx
NVCC     = nvcc

# Architecture flags for Cascade Lake (using Skylake fallback)
ARCH_FLAGS = -march=skylake-avx512 -mtune=skylake-avx512

# Optimization flags for standard builds
COMMON_FLAGS = -O3 -std=c++17 -Wall
DEBUG_FLAGS  = -g -fno-omit-frame-pointer

# Aggressive flags for Ultra versions (AVX-512 + Loop Unrolling + Alignment)
ULTRA_FLAGS  = $(ARCH_FLAGS) -ffast-math -funroll-loops -finline-functions \
               -fno-trapping-math -fno-math-errno -falign-functions=32 \
               -falign-loops=32 -fno-semantic-interposition \
               -ftree-vectorize \
               -fopt-info-vec-optimized=$(REP_DIR)/vec_report_ultra.txt

# Library specific flags
OMP_FLAGS  = -fopenmp
CUDA_FLAGS = -O3 -arch=sm_70 -lineinfo

# ==========================================
# Targets
# ==========================================

all: directories serial serial_ultra openmp openmp_ultra mpi mpi_ultra hybrid hybrid_ultra cuda cuda_ultra

directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(REP_DIR)

# --- Serial ---
serial: $(SRC_DIR)/nbody_serial.cpp
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial

serial_ultra: $(SRC_DIR)/nbody_serial_ultra.cpp
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial_ultra

# --- OpenMP ---
openmp: $(SRC_DIR)/nbody_openmp.cpp
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp

openmp_ultra: $(SRC_DIR)/nbody_openmp_ultra.cpp
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp_ultra

# --- MPI ---
mpi: $(SRC_DIR)/nbody_mpi.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi

mpi_ultra: $(SRC_DIR)/nbody_mpi_ultra.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi_ultra

# --- Hybrid ---
hybrid: $(SRC_DIR)/nbody_hybrid.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid

hybrid_ultra: $(SRC_DIR)/nbody_hybrid_ultra.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid_ultra

# --- CUDA ---
cuda: $(SRC_DIR)/nbody_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -std=c++17 $< -o $(BIN_DIR)/nbody_cuda

cuda_ultra: $(SRC_DIR)/nbody_cuda_ultra.cu
	$(NVCC) $(CUDA_FLAGS) --use_fast_math -std=c++17 $< -o $(BIN_DIR)/nbody_cuda_ultra

# ==========================================
# Utility
# ==========================================

clean:
	rm -f $(BIN_DIR)/* $(REP_DIR)/*.txt

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all           : Compile ALL versions"
	@echo "  serial        : Standard Serial"
	@echo "  serial_ultra  : Optimized AVX-512 Serial"
	@echo "  openmp        : Standard OpenMP"
	@echo "  openmp_ultra  : Optimized OpenMP"
	@echo "  mpi           : Standard MPI"
	@echo "  mpi_ultra     : Optimized MPI"
	@echo "  hybrid        : Standard Hybrid (MPI+OpenMP)"
	@echo "  hybrid_ultra  : Optimized Hybrid"
	@echo "  cuda          : Standard CUDA"
	@echo "  cuda_ultra    : Optimized CUDA (Fast Math)"
	@echo "  clean         : Remove binaries"
