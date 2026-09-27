#!/bin/bash
set -e
echo "=== stopping old container ==="
docker stop ninfer-qwen38-27b 2>/dev/null || true
docker rm ninfer-qwen38-27b 2>/dev/null || true
echo "=== starting new container from ninfer:codex ==="
docker run -d --name ninfer-qwen38-27b \
  --runtime nvidia \
  --restart unless-stopped \
  -p 8020:8020 \
  -v /home/rjman/Servers/local-ai/profiles/qwen38-27b/chat_template_lenient.jinja:/chat_template_lenient.jinja:ro,z \
  -v /home/rjman/models/ninfer/Qwen3.8-27B-swift15abl-nvfp4full-dflash2-NInfer-v3:/models:ro,z \
  -v /home/rjman/ninfer-logs:/reqlog:z \
  ninfer:latest \
  ninfer-serve /models/qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer --host 0.0.0.0 --port 8020 --cors --request-log-jsonl /reqlog/requests.jsonl --chat-template /chat_template_lenient.jinja --api-key rjman --model-id local --max-context 262144 --kv-capacity auto --kv-dtype k8v4 --max-concurrency 4 --host-kv-mib 32768 --host-state-slots 16 --vision --preserve-thinking --spec dflash2 --draft-tokens 7 --lm-head-draft --temperature 0.9 --min-p 0.05 --pending-timeout-ms 90000 --default-max-tokens 32768 --default-thinking-budget 16384 --prefill-chunk 4096 --log-stats-interval-ms 2000
echo "=== container started, id above ==="
