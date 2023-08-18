#!/bin/bash

#OAR -n test_T5
#OAR -l /nodes=1/gpu=1,walltime=0:30:00
#OAR --stdout %jobid%.out
#OAR --stderr %jobid%.err
#OAR --project pr-statify
#OAR -p gpumodel='V100'

echo "$PWD"
source /applis/environments/cuda_env.sh dahu 10.2
source /applis/environments/conda.sh
conda activate SMOOTHING
echo "$(ls)"
python3 ./testHF.py