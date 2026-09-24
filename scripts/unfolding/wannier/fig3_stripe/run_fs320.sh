#!/bin/bash
#SBATCH --job-name=fig3_unfold320
#SBATCH --output=fs_%j.out
#SBATCH --error=fs_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00        # short wall so Slurm can BACKFILL it: the work is ~2 min
#SBATCH --mem=0
#SBATCH --partition=mpi
#SBATCH --account=<ACCOUNT>
# 4 independent (material x spin) unfoldings, one per Slurm task. 2 nodes x 2 tasks
# x 32 cores; each piece takes about 2 minutes.
source /etc/profile.d/modules.sh
module load miniconda
conda activate <python env with numpy>
export OMP_NUM_THREADS=1          # the parallelism is python-level; keep BLAS from oversubscribing
export FS_NK=${FS_NK:-241}
cd "$SLURM_SUBMIT_DIR"
echo "start $(date)  nk=$FS_NK"
srun --cpu-bind=cores python3 -u fs_worker320.py
echo "end $(date)"
