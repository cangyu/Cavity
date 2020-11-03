#!/bin/bash

#SBATCH --job-name=h50nRe150
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

../bin/CAVITY --mesh $MESH_PATH/CUBE/Fluent/hex/50n_translated.txt --Re 150  --time-span 20  --write-interval 500

