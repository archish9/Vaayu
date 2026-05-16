● source venv/bin/activate && python quantize.py \
    --in_dir   /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct \
    --out_dir  /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw \
    --work_dir /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-work \
    --bits     4.5 \
    --resume

  Or background version:
  source venv/bin/activate && nohup python quantize.py \
    --in_dir   /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct \
    --out_dir  /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw \
    --work_dir /home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-work \
    --bits     4.5 \
    --resume > /home/a/code/ai-ml/Vaayu/quantize.log 2>&1 &

