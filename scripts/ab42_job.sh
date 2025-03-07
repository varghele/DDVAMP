#!/bin/bash
#SBATCH --job-name=ab42_train_nc${SLURM_ARRAY_TASK_ID}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=paula
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

# Set job name properly with array ID since the above won't expand at submission time
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    JOB_NAME="ab42_train_nc${SLURM_ARRAY_TASK_ID}"
    scontrol update JobId=${SLURM_JOB_ID} Name=${JOB_NAME}
    echo "Updated job name to: ${JOB_NAME}"
fi

# Load required modules
module purge
module load CUDA/12.4.0
module load Anaconda3/2024.02-1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ddvamp

# Create log directory if it doesn't exist
mkdir -p logs

# Print job array information
echo "Running job array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Using num_classes = ${SLURM_ARRAY_TASK_ID}"

# Print GPU information
nvidia-smi

# Run the training script with the array task ID as num_classes
python run_training.py \
    --protein-name "ab42" \
    --topology "datasets/ab42/trajectories/red/topol.gro" \
    --traj-folder "datasets/ab42/trajectories/red/" \
    --num_neighbors 10 \
    --num_classes ${SLURM_ARRAY_TASK_ID} \
    --n_conv 4 \
    --h_a 16 \
    --h_g 16 \
    --hidden 32 \
    --dropout 0.4 \
    --dmin 0.0 \
    --dmax 5.0 \
    --step 0.2 \
    --conv_type "SchNet" \
    --residual \
    --atom_init "normal" \
    --learning_rate_a 0.0005 \
    --learning_rate_b 0.0001 \
    --tau 20 \
    --batch_size 500 \
    --val_frac 0.2 \
    --epochs 5 \
    --pre-train-epoch 5 \
    --score_method "VAMPCE" \
    --save_checkpoints \
    --steps "training" \
    --n-structures 10

# Print completion message
echo "Job array task ${SLURM_ARRAY_TASK_ID} completed at: $(date)"
