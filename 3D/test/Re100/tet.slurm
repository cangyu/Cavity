#!/bin/bash

#SBATCH --job-name=t7wRe100
#SBATCH --partition=small
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=yu.cang@sjtu.edu.cn
#SBATCH -n 1
#SBATCH --ntasks-per-node=1

RUN_DIR=`date +%F | sed 's/-//g'`-`date +%T | sed 's/://g'`

$CAVITY_DIR/bin/CAVITY --mesh $MESH_DIR/CUBE/Fluent/tet/75470_translated.txt --Re 100 --time-span 10 --write-interval 100 --tag $RUN_DIR

cd $RUN_DIR

mkdir raw

mv *.txt raw/

TEC_DATA_CONVERTER --composition tet --input-dir raw --output-dir tec

