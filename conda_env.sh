#!/usr/bin/env bash
cd "$(dirname "$0")"
echo running conda create -y -n $1 python=3.7 && \
conda create -y -n $1 python=3.7 && \
echo running conda install -n $1 -y cudatoolkit=10.0 cudnn=7.6.5 pip tensorflow-gpu=1.15.0 scipy=1.1.0 && \
conda install -n $1 -y cudatoolkit=10.0 cudnn=7.6.5 pip tensorflow-gpu=1.15.0 scipy=1.1.0 && \
echo running conda run -n $1 pip install -r requirements.txt
conda run -n $1 pip install -r requirements.txt
