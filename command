● source venv/bin/activate && python quantize.py \
    --in_dir   /home/a/code/ai-ml/models/MiniMaxAI--MiniMax-M2.7 \
    --out_dir  /home/a/code/ai-ml/models/MiniMaxAI--MiniMax-M2.7-exl3-4.5bpw \
    --work_dir /home/a/code/ai-ml/models/MiniMaxAI--MiniMax-M2.7-exl3-work \
    --bits     4.5 \
    --resume

  Or background version:
  source venv/bin/activate && nohup python quantize.py \
    --in_dir   /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct \
    --out_dir  /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw \
    --work_dir /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-work \
    --bits     4.5 \
    --resume > /home/a/code/ai-ml/Vaayu/quantize.log 2>&1 &

claude --resume 23c6b129-7a65-4cbe-a5d3-f1c8ba24bcab


nvidia-smi -q | grep -A 10 "Link Width"



MAX_PINNED_LAYERS = 20       # Leaves ~2-3 GB of headroom to be safe
EAGLE_HEAD_PATH = None
DRAFTER_MODEL_PATH = "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct"



python -u test_quick.py > run.log 2>&1