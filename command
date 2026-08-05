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



nvidia-smi -q | grep -A 10 "Link Width"

MAX_PINNED_LAYERS = 20       # Leaves ~2-3 GB of headroom to be safe
EAGLE_HEAD_PATH = None
DRAFTER_MODEL_PATH = "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct"



python -u test_quick.py > run.log 2>&1

python3 test_generate.py > run.log 2>&1

https://huggingface.co/models?other=exllamav3&sort=most_params


model-agnostic engine
                        


===

this plan will only help users who has low am and gpu.

4. C — dynamic RAM ceiling (2026-07-19-vaayu-dynamic-ram-cache-ceiling-plan.md) - Last. Tiny (2 tasks) but lowest/speculative value (portability, only one box today). Flag-gated + needs on-device calibration. Could even defer until a 2nd box exists.

===

  docs/superpowers/plans/2026-07-19-vaayu-dynamic-ram-cache-ceiling-plan.md


Execute the plan at docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic.md using superpowers:subagent-driven-development


python -u test_kv_persist_coherence.py --phase run1 > run1.log 2>&1
python -u test_kv_persist_coherence.py --phase run2 > run2.log 2>&1
python -u test_kv_persist_coherence.py --phase control > control.log 2>&1


_PREFIX_CACHE_RAM_RESERVE


https://huggingface.co/QuantTrio



claude --resume 348226a4-723f-42cd-aa21-d255b21f046f






peremeter:
expert_budget
delete_original
PREFIX_COMPRESS
