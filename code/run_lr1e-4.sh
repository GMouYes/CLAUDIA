#!/bin/bash
#SBATCH -p short
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH --mem=50G
#SBATCH --exclude=gpu-4-02
#SBATCH -t 1-00:00:00 # wall time (D-HH:MM) latest HH:MM:SS
#SBATCH -e ../log/slurm.%j.err # STDERR (%j = JobId)
#SBATCH -o ../log/slurm.%j.out # STDOUT (%j = JobId)                                                                         
#SBATCH --mail-type=END # Send a notification when a job starts,
#stops, or fails
#SBATCH --mail-user=wge@wpi.edu
module load python/3.9.12
module load cuda11.7/blas/11.7.1
module load cuda11.7/fft/11.7.1
module load cuda11.7/toolkit/11.7.1
source /home/wge/gnn_cuda11/bin/activate

EXPNAME='binary_extrasensory_lr1e-4'
echo "EXPNAME $EXPNAME"

python3 -u trainer.py \
    --config_path "../config/config_lr1e-4.yml" \
    --expName $EXPNAME \
    --outputPath '../output/'$EXPNAME \
    --lossPath $EXPNAME'.jpg' \

