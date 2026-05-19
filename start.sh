#!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /root
source /root/venv/bin/activate
python3 app.py
