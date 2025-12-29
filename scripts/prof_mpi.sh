#!/bin/bash
#SBATCH --job-name=prof_mpi
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/prof_mpi_%j.log
#SBATCH --error=logs/errors/prof_mpi_%j.err
#SBATCH --exclusive

# --- Configuration ---
N=${1:-5000}
STEPS=1000
MODE="bench"
RESULT_DIR="vtune_reports/mpi_N${N}_ID${SLURM_JOB_ID}"

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
echo ""
echo "Modules used:"
gcc --version
vtune --version
mpicxx --version
mpicxx -show

echo "============================================"
echo "Profiling MPI (Standard)"
echo "Particles: $N | Ranks: 48"
echo "Output: $RESULT_DIR"
echo "============================================"

# Note: We wrap the application with vtune, but launch via mpirun
# For single node, this traces all ranks.
mpirun -np 48 --mca btl tcp,self,vader \
	vtune -collect hotspots \
      -result-dir ${RESULT_DIR} \
      -- ./bin/nbody_mpi $N $STEPS $MODE
