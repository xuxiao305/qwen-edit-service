#!/bin/bash
# Stop old qwen server, start new dual-GPU one, show status.
set -u
pkill -9 -f qwen_edit_server.py 2>/dev/null
sleep 3
echo "=== GPU before warmup ==="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv
echo "=== Starting new server (dual-GPU loader) ==="
cd /project/qwen_edit_service
bash server/run_server.sh --bg
sleep 5
echo "=== Server log ==="
tail -30 /project/qwen_edit/logs/server.out
echo "=== Process ==="
ps -ef | grep qwen_edit_server | grep -v grep || echo "NO_PROC"
echo "=== Health (local) ==="
curl -s http://127.0.0.1:8765/health
