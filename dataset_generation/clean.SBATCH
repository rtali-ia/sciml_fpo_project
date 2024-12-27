#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=pr_95_general
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mem=180GB
#SBATCH --job-name=cleanFlowBench
#SBATCH --output=cleanFlowBench_%j.out  # Save standard output and errors to a file
#SBATCH --mail-type=START,END,FAIL                                   # Send email on job completion or failure
#SBATCH --mail-user=au2216@nyu.edu                    # Replace with your email address

# Load necessary modules (if required by your cluster)
module purge

# Run Singularity with overlay and CUDA-enabled container
singularity exec \
    --overlay /scratch/au2216/my_env/overlay-25GB-500K_fpo_download.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; python data_prep.py"