#!/bin/bash

#OAR -n test_T5
#OAR -l /nodes=1/gpu=2,walltime=06:00:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-statify
#OAR -p gpumodel='A100' and mem_per_gpu > 74

source /applis/environments/cuda_env.sh bigfoot 11.7
source /applis/environments/conda.sh
conda activate SMOOTHING

python3 ./prompt.py -n 1 -r 1 -N 50 -vvvv -t 10

