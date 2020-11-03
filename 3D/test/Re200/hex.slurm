#!/bin/bash

#SBATCH --job-name=h64Re200
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

../bin/CAVITY --mesh $MESH_PATH/CUBE/Fluent/hex/64_translated.txt --Re 200  --time-span 20 --write-interval 100 --tag $SLURM_JOB_NAME

