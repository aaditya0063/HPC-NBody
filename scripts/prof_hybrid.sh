#!/bin/bash
#SBATCH --job-name=prof_hyb
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=logs/prof_hybrid_%j.log
#SBATCH --error=logs/errors/prof_hybrid_%j.err
#SBATCH --exclusive

# --- Configuration ---
N=${1:-5000}
STEPS=1000
MODE="bench"
RESULT_DIR="vtune_reports/hybrid_N${N}_ID${SLURM_JOB_ID}"

# --- Modules ---
module purge
module load spack
. /home/apps/spack/share/spack/setup-env.sh
spack find gcc
spack load gcc@13.1.0%gcc@13.1.0

module spider vtune
module load oneapi/vtune/2021.7.1

module spider openmpi
module load openmpi/4.1.1

echo "Modules used:"
gcc --version
vtune --version
mpicxx --version
mpicxx -show

# --- Hybrid Environment ---
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "============================================"
echo "Profiling Hybrid (Standard)"
echo "Particles: $N"
echo "Ranks: 2 | Threads/Rank: 24"
echo "Output: $RESULT_DIR"
echo "============================================"

# We verify the socket mapping while profiling
mpirun -np 2 \
    --mca btl tcp,self,vader \
    --map-by socket:PE=24 \
    vtune -collect hotspots \
    -result-dir ${RESULT_DIR} \
    -- ./bin/nbody_hybrid $N $STEPS $MODE
