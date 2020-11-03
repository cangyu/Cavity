#!/bin/bash

#SBATCH --job-name=h80Re300
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

../bin/CAVITY --mesh $MESH_PATH/CUBE/Fluent/hex/80_translated.txt --Re 300  --time-span 25 --write-interval 200 --tag $SLURM_JOB_NAME

