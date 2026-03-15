#!/bin/bash
set -e
cd /root
echo "🐕 拉取最新代码..."
git pull origin main
echo "📦 检查依赖..."
source venv/bin/activate
pip install -r requirements.txt -q
echo "🔄 重启服务..."
nohup bash -c "sleep 2 && systemctl restart kb" > /dev/null 2>&1 &
echo "✅ 更新完成，服务将在2秒后重启"
