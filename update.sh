#!/bin/bash
cd /root git pull source 
venv/bin/activate pip install -r 
requirements.txt -q systemctl 
restart kb echo "更新完成 ✅"
