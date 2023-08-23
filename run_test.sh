#!/bin/bash

#OAR -n test_T5
#OAR -l /nodes=1/gpu=1,walltime=06:00:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-statify
#OAR -p gpumodel='A100'

source /applis/environments/cuda_env.sh dahu 10.2
source /applis/environments/conda.sh
conda activate SMOOTHING
python3 ./prompt.py -vvvv
