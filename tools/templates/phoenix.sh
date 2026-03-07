#!/usr/bin/env bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes={{NODES}}
#SBATCH --ntasks-per-node={{PPN}}
#SBATCH --time={{WALLTIME}}
#SBATCH --account={{ACCOUNT}}
#SBATCH --output={{OUTPUT_DIR}}/{{JOB_NAME}}.out
#SBATCH --error={{OUTPUT_DIR}}/{{JOB_NAME}}.err
#SBATCH -Cgraniterapids
{{PARTITION_LINE}}
{{QOS_LINE}}

eval "$({{SIFV_DIR}}/sifv.sh load -c {{TEMPLATE_NAME}})"

cd "{{OUTPUT_DIR}}"
srun -N {{NODES}} --ntasks-per-node={{PPN}} {{RUN_COMMAND}}
