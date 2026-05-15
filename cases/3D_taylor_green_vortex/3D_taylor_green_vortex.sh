#!/usr/bin/env bash
#SBATCH --job-name=3D_taylor_green_vortex
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:RTX_6000:4
#SBATCH --time=04:00:00
#SBATCH --account=gts-sbryngelson3
#SBATCH --output=/storage/scratch1/6/bwilfong3/software/SemiImplicitFV/cases/3D_taylor_green_vortex/3D_taylor_green_vortex.out
#SBATCH --error=/storage/scratch1/6/bwilfong3/software/SemiImplicitFV/cases/3D_taylor_green_vortex/3D_taylor_green_vortex.err

#SBATCH --qos=embers

eval "$(/storage/scratch1/6/bwilfong3/software/SemiImplicitFV/sifv.sh load -c phoenix-gpu)"

cd "/storage/scratch1/6/bwilfong3/software/SemiImplicitFV/cases/3D_taylor_green_vortex"
mpirun -np 32 /storage/scratch1/6/bwilfong3/software/SemiImplicitFV/build/sifv /storage/scratch1/6/bwilfong3/software/SemiImplicitFV/cases/3D_taylor_green_vortex/3D_taylor_green_vortex.jsonc
