#!/bin/bash

#SBATCH --nodes=1
#SBATCH --account=pr_95_tandon_priority
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=180GB
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=deepoNetTest
#SBATCH --output=deepoNetTest_%j.out  # Save standard output and errors to a file
#SBATCH --mail-type=START,END,FAIL                                   # Send email on job completion or failure
#SBATCH --mail-user=au2216@nyu.edu                    # Replace with your email address

# Load necessary modules (if required by your cluster)
module purge

export WANDB_CACHE_DIR=/scratch/au2216/flowbench_fpo/sciml_fpo_project/model_training/wandb/.cache/

# Run Singularity with overlay and CUDA-enabled container
singularity exec --nv\
    --overlay /scratch/au2216/my_env/overlay-25GB-500K_fpo_download.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; python fpo_sweep.py --model deeponet --equation ns"
