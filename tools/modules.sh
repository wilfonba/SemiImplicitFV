#!/usr/bin/env sh

module purge
if [ "$1" = "p" ]; then
    module load gcc/12.3.0 openmpi/4.1.5 python/3.12.5
fi
