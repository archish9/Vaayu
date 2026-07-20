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

python3 test_generate.py > run.log 2>&1

https://huggingface.co/models?other=exllamav3&sort=most_params


model-agnostic engine





                                                                                       
  Execute the plan at docs/superpowers/plans/2026-07-12-vaayu-prefix-caching-plan.md                                             
  using superpowers:executing-plans. Task-by-task with review checkpoints.                                                       



  1. RAM prefix cache (2026-07-12-vaayu-prefix-caching-plan.md)
  - Hard prerequisite for A + top TTFT lever. Must go first.

  2. A — KV disk persistence (2026-07-19-vaayu-kv-persist-plan.md)
  - Extends the prefix cache you just built (infra warm in head). Tasks 1–4 need nothing; Task 5 needs step 1 done.

  3. B — expert usage histogram (2026-07-19-vaayu-expert-usage-histogram-plan.md)
  - Independent, medium value (kills cold-start, matches loop-engineering). Biggest + riskiest (touches MoE hot path) → do once A
  is stable. Internal gate: Phase 1 (Tasks 1–6) → on-device verify (Task 7) → Phase 2 (8–9). Don't start Phase 2 till the gate
  passes.

  4. C — dynamic RAM ceiling (2026-07-19-vaayu-dynamic-ram-cache-ceiling-plan.md)
  - Last. Tiny (2 tasks) but lowest/speculative value (portability, only one box today). Flag-gated + needs on-device
  calibration. Could even defer until a 2nd box exists.


Execute the plan at docs/superpowers/plans/2026-07-19-vaayu-kv-persist-plan.md using superpowers:subagent-driven-development.


python -u test_kv_persist_coherence.py --phase run1 > run1.log 2>&1
python -u test_kv_persist_coherence.py --phase run2 > run2.log 2>&1
python -u test_kv_persist_coherence.py --phase control > control.log 2>&1

claude --resume 11724729-d32b-4e19-a5ae-1aac73da9ae2
