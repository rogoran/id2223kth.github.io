#!/bin/bash

set -e

cd src/serverless-ml-intro/serverless-ml-lab

python wine-feature-pipeline-daily.py
python wine-batch-inference-pipeline.py