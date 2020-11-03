#!/bin/bash

#SBATCH --job-name=h90Re400
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

../bin/CAVITY --mesh $MESH_PATH/CUBE/Fluent/hex/90_translated.txt --Re 400 --time-span 40 --write-interval 200 --tag $SLURM_JOB_NAME

