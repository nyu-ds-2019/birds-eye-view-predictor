filename = $1
sbatch --export=filename='$filename' sbatch_run.slurm
