#!/usr/bin/env bash
cd "$(dirname "$0")"
set -o xtrace  # print commands as they are being run

env_name="${1:-anonymizer}"
conda create -y -n ${env_name} python=3.7  cudatoolkit=10.0 cudnn=7.6.5 pip tensorflow-gpu=1.15.0 scipy=1.1.0 && \
conda run -n ${env_name} pip install . develop
