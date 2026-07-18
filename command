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





How to start each — paste this in a NEW session (on the GPU box)                                                               
                                                                                                                                 
  8-task plan:                                                                                                                   
  Execute the plan at docs/superpowers/plans/2026-07-12-vaayu-perf-fixes-and-features.md                                         
  using superpowers:subagent-driven-development. Task-by-task, stop for my review                                                
  at each task's checkpoint. Qwen only for on-device A/B (no MiniMax checkpoint).                                                
                                                                                                                                 
  A1 prefix caching (after 8-task lands):                                                                                        
  Execute the plan at docs/superpowers/plans/2026-07-12-vaayu-prefix-caching-plan.md                                             
  using superpowers:executing-plans. Task-by-task with review checkpoints.                                                       
                                                                                                                                 
  Feature B (brainstorm, not execute yet):                                                                                       
  Run superpowers:brainstorming on Feature B (multi-query batched decode) using                                                  
  docs/superpowers/plans/2026-07-12-vaayu-agent-workload-brainstorm-prep.md.                                                     
  Read §0-§1 and §3, brainstorm from §3's questions, then write a plan.




Execute the plan at docs/superpowers/plans/2026-07-18-vaayu-batched-decode.md using superpowers:subagent-driven-development.