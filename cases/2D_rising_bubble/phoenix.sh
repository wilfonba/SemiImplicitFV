#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --tasks-per-node=100
#SBATCH --time=08:00:00
#SBATCH --job-name=SIRB
#SBATCH -o SIRB.out
#SBATCH -e SIRB.err
#SBATCH -C graniterapids
#SBATCH -A gts-sbryngelson3
#SBATCH -q embers

cd ../../

source tools/modules.sh p

./sifv.sh run 2D_rising_bubble --petsc -n 200 --srun
