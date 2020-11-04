#!/bin/bash

#SBATCH --job-name=Re150
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

RUN_DIR=`date +%F | sed 's/-//g'`-`date +%T | sed 's/://g'`

$CAVITY_DIR/bin/CAVITY --mesh $MESH_DIR/CUBE/Fluent/hex/50_translated.txt --Re 150 --time-span 15 --write-interval 200 --tag $RUN_DIR

cd $RUN_DIR

mkdir raw

mv *.txt raw/

TEC_DATA_CONVERTER --composition hex --input-dir raw --output-dir tec

