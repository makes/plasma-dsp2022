#!/bin/bash
#SBATCH --job-name=vdfsample     # create a short name for your job
#SBATCH --output=slurm-%A.%a.out # stdout file
#SBATCH --error=slurm-%A.%a.err  # stderr file
#SBATCH --partition=test         # partition (test, short, medium, ...)
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # number of processes
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=0-00:20:00        # total run time limit (d-hh:mm:ss)
#SBATCH --array=0-15%15          # job array with index values 0, 1, 2, 3, ...
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=yourusername@example.com

python vdfsample.py
