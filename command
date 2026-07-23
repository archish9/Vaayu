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
                        


  4. C — dynamic RAM ceiling (2026-07-19-vaayu-dynamic-ram-cache-ceiling-plan.md)
  - Last. Tiny (2 tasks) but lowest/speculative value (portability, only one box today). Flag-gated + needs on-device
  calibration. Could even defer until a 2nd box exists.


Execute the plan at docs/superpowers/plans/2026-07-23-ram-accounting-diagnostic.md using superpowers:subagent-driven-development.


python -u test_kv_persist_coherence.py --phase run1 > run1.log 2>&1
python -u test_kv_persist_coherence.py --phase run2 > run2.log 2>&1
python -u test_kv_persist_coherence.py --phase control > control.log 2>&1


_PREFIX_CACHE_RAM_RESERVE


https://huggingface.co/QuantTrio


Best for High-Performance API & Cost-Efficiency: DeepSeek V4 Pro and DeepSeek V4 Flash are standout models offering elite performance across LiveCodeBench while being highly efficient for integration into coding agents.

DeepSeek-R1 is the headline answer. Released January 2025, MIT-licensed, open weights, and explicitly a reasoning model with visible chain-of-thought — the direct open answer to OpenAI's o1/o3-mini line. On math-style reasoning (AIME and similar) it traded blows with o3-mini and sometimes beat it. On coding specifically — LiveCodeBench, the thing your quote emphasizes — it was genuinely competitive, though o3-mini-high often edged it on the hardest coding benchmarks. So "as good" is fair on reasoning broadly; on pure competitive coding it was more of a real contest.

QwQ-32B (Qwen's reasoning model, March 2025) is the one that surprised people: it approached R1-level reasoning at only 32B parameters, putting o3-mini-class thinking in a dramatically smaller package.

Qwen3 (2025) is probably the strongest open family for your use case now. The models have a built-in "thinking mode," come in sizes from tiny up to a 235B MoE, and the larger ones beat o3-mini on many reasoning and coding benchmarks. Worth also knowing: Phi-4-reasoning (Microsoft, 14B), GLM-Z1 (Zhipu), and Nvidia's Llama-Nemotron reasoning models all landed in this space too.


claude --resume 348226a4-723f-42cd-aa21-d255b21f046f
