#!/bin/bash

#SBATCH --nodes=32
#SBATCH --tasks-per-node=128
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=08:00:00
#SBATCH --job-name=TGV_512
#SBATCH --account=gts-sbryngelson3
#SBATCH -C graniterapids
#SBATCH -q embers

cd ../../

source tools/modules.sh p

module list

./run_case.sh --petsc -n 4096 3D_taylor_green_vortex

