"""Vaayu server configuration -- edit this file, there are no CLI flags.

    ./venv/bin/python -m vaayu.server

Every value below is documented in `vaayu/server/README.md`, with the measured
cost of getting it wrong. The defaults are already the fast ones and differ from
`vaayu_base`'s library defaults on purpose: a server holds one model for its
whole life and should fill the box.

Machine-specific values (MODEL_PATH above all) belong in `config_local.py`,
which is gitignored and shadows anything set here -- see the tail of this file.
"""

# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

# Checkpoint directory. REQUIRED -- the server refuses to start without it.
MODEL_PATH = ""

# The `id` reported by /v1/models and echoed in responses. None uses the
# checkpoint path; set something short and stable so clients can hardcode it.
SERVED_MODEL_NAME = None

HOST = "127.0.0.1"          # "0.0.0.0" to expose on the LAN
PORT = 9999
DEVICE = "cuda:0"

# Waiting requests allowed before 503. ONE runs at a time regardless.
QUEUE_DEPTH = 8

# Refuse a request that names no session. An unnamed request still gets prefix
# reuse (the pool auto-matches), but its key is a hash of the prompt's first 64
# tokens -- so two agents sharing a system prompt land on ONE entry and evict
# each other, and reuse collapses to zero rather than degrading. OpenAI's `user`
# field counts as a session, so an OpenAI-native client is not locked out.
REQUIRE_SESSION_ID = True

# Refuse to start while another process holds GPU memory on DEVICE. Vaayu sizes
# its pinning and cache budgets ONCE, from what is free at load, so a forgotten
# container costs every token for the life of the run -- and the loss is
# invisible in the output. See the REQUIREMENT section in the top-level
# README.md. Set False for a box you deliberately share; it then only warns.
REQUIRE_EXCLUSIVE_GPU = True

# How much VRAM another process must hold before it is worth refusing over.
# A bare CUDA context is 128 MiB on this box, so at 0 a stray notebook -- or any
# process merely importing torch -- refuses the boot while costing nothing.
# Anything under this is still REPORTED, just not treated as a blocker. Only
# processes on DEVICE count at all; a job on another GPU is not Vaayu's problem.
EXCLUSIVE_GPU_MIN_MIB = 256

# Delete the ORIGINAL downloaded shards once Vaayu's own split copy is verified
# complete (every layer has its .safetensors AND its .done marker), reclaiming
# 39-66 GB per checkpoint. A server holds one checkpoint for its whole life, so
# the duplicate is pure waste -- hence ON by default.
#
# IRREVERSIBLE. Recovery is a re-download. The removal follows symlinks, so a
# shard linked into the shared HuggingFace cache takes the cached blob with it,
# which affects every other tool sharing that cache. Set False to keep the
# originals -- for example if another framework reads the same checkpoint.
DELETE_ORIGINAL = True

# Let the model emit a reasoning block before its answer. OFF by default: the
# trace costs the same tok/s as the answer and the caller did not ask for it.
# Templates that expose their own flag are told; one that hard-codes an open
# `<think>` (MiniMax-M2.7) gets the block closed for it. A request may override
# per call with `"enable_thinking": true`.
ENABLE_THINKING = False

# What `stream` means when a request omits it. True, NOT OpenAI's false: a
# non-streaming response sends nothing until the last token. An explicit
# `"stream": false` in the request still wins.
DEFAULT_STREAM = True

# Where logs are written. One directory per process, `<LOG_DIR>/run-<UTC>/`,
# with `<LOG_DIR>/latest` always pointing at the newest. Within it,
# `server.log` receives EVERYTHING the process emits -- Python prints,
# uvicorn, and the vaayu_rust io_uring lines, which are captured at the
# file-descriptor level because Rust writes them straight to fd 2.
# `sessions/<session_id>.log` gets the whole process stream for that session
# (not just lifecycle events), which is the file to read when one session
# misbehaves.
LOG_DIR = "logs"
CAPTURE_STDIO = True        # False leaves the console alone; only session logs
LOG_ROTATE_MB = 50          # server.log rolls to server.log.1 past this

# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------

# -1 auto-fills the VRAM budget with transformer layers. 0 pins the shared
# layers only, so every transformer layer streams from NVMe on every token
# while the pinning budget sits idle: measured 13x on Llama-3.2-1B, 2.0x on
# MiniMax-M2.7, 1.27x on Llama-3.3-70B. N pins the first N.
MAX_PINNED_LAYERS = -1

# Cross-turn KV prefix reuse -- the largest measured lever here (136-377x TTFT)
# and exactly the workload an agent resending its history every turn creates.
ENABLE_PREFIX_CACHE = True

# Let a request with NO session id still find a prefix entry, by matching the
# longest shared prompt prefix instead of an exact key. This is the behaviour
# REQUIRE_SESSION_ID's note above describes; until 2026-09-10 the server could
# not turn it off. Leave it True unless you have measured a reason: with it
# False an unnamed request gets NO reuse at all rather than approximate reuse,
# which on this workload is the larger loss. It changes nothing for a request
# that names its session -- that path keys exactly, matched or not.
AUTO_PREFIX_MATCH = True

# Context to plan VRAM for. None DERIVES the largest context whose predicted
# peak fits the reserve floor -- the shipped path, and the one every measured
# figure here was taken at.
#
# The only reason to set it: asking for a SHORTER context hands the difference
# back to the expert cache, which is the decode bottleneck on both MoE targets.
# The cost of getting it wrong is two-sided and both sides are loud, not silent:
# above the box's S_max the load REFUSES (VramPlanError, naming S_max), and
# above the derived operating point but under S_max the load succeeds while the
# larger KV reserve eats the pinning budget -- at 0 the engine warns that every
# layer now streams from disk on every token. Neither degrades quietly.
MAX_SEQ_LEN = None

# Tokens per prefill chunk. 512 is measured best; smaller costs TTFT
# (+24.7% MoE, +213% dense at 128) and buys nothing.
PREFILL_CHUNK_SIZE = 512

# Requests served together as ONE batch. 1 is today's behaviour and the safe
# default. A CEILING, never a target: max width is model- and
# context-dependent and is NOT 3.
#
# What getting it wrong costs, measured 2026-09-08 at ~4.15k-token agentic
# prompts, 2 requests x 64 tokens, engine-level:
#   2  MiniMax-M2.7 454.80s -> 283.13s = +1.606x, TTFT 317.2 -> 223.3s, and
#      peak VRAM LOWER batched (12.355 vs 13.259 GB).
#   3  OOMs MiniMax at model.layers.22; on Qwen3-235B it is SLOWER than 2
#      (1.278 against 1.582 aggregate tok/s -- a bigger KV starves the expert
#      cache).
#   Qwen3-235B admits N=1 only today, so any value here buys it nothing.
#
# INERT unless ATTN_IMPLEMENTATION is "sdpa" below -- batched decode requires
# that kernel, and a non-sdpa load serializes with a warning at startup rather
# than failing a request.
MAX_BATCH_SIZE = 1

# Milliseconds a forming batch waits for company before running what it has.
# INERT at MAX_BATCH_SIZE = 1 (nothing to form) and on a non-sdpa load.
#
# Without it the server never batches two concurrent clients AT ALL, whatever
# MAX_BATCH_SIZE says: a request wakes the dispatcher the instant it is queued,
# so the first of two simultaneous clients runs alone while the second is still
# being tokenized in its own route -- measured on Llama-3.2-1B, 2 clients, warm
# server, batch=None on both requests in 3 runs of 3. It also un-sticks two
# STEADY clients, which otherwise alternate one-at-a-time forever and never
# meet in the queue.
#
# The cost of getting it wrong is bounded and one-sided: a lone request at an
# idle server is delayed by this much, once (0.02 % of MiniMax-M2.7's ~220 s
# time-to-first-token at a 4.15k-token prompt, ~4 % of a Llama-3.2-1B request).
# A batch that fills stops waiting immediately, so a genuinely concurrent load
# pays nothing. 0 restores the pre-2026-09-09 behaviour.
#
# It does NOT help a client whose turnaround is longer than the window -- an
# agent that runs a tool between turns comes back seconds later, and admitting
# a latecomer into a RUNNING batch is continuous batching, which Vaayu does not
# do.
BATCH_WAIT_MS = 50

# Pack pooled KV at N bits (9 is the tested floor) so the same RAM budget holds
# ~2x the sessions. Reuse collapses to ZERO, not proportionally, once the pool
# is over capacity -- so this matters when serving several agent sessions.
PREFIX_COMPRESS = None

# Concurrent agent sessions the prefix pool must hold before the MoE expert
# cache claims the rest. Lower = faster decode, fewer cached sessions.
PREFIX_SESSIONS = 3

# "auto" picks bf16 for EXL3. "fp8" is 1.5x LARGER, not smaller -- it keeps a
# dequantized mirror.
KV_CACHE_MODE = "auto"

# None PREFERS flash_attention_2 when flash_attn imports, then sdpa, then eager
# (vaayu_base.init_model). It does NOT mean sdpa -- this line said so until
# 2026-09-08 and it is load-bearing: MAX_BATCH_SIZE above is inert on flash.
# Every shipped max_seq_len figure was verified on sdpa; "flash_attention_2"
# derives a larger context at the same reserve, and costs -2.3 % MiniMax /
# -4.1 % Qwen end-to-end against sdpa.
ATTN_IMPLEMENTATION = None

# GB of PAGEABLE RAM for the dense layer cache. None derives it from TOTAL RAM
# so throughput is reproducible across restarts (keying off momentary free RAM
# swung Llama-3.3-70B decode 38% run to run). 0 disables it. No effect on MoE,
# whose experts live in the pinned expert RAM cache instead.
RAM_CACHE_BUDGET_GB = None

# Persist prefix KV to disk so reuse survives a restart.
ENABLE_KV_PERSIST = False

# Where those entries are written. None is ~/.cache/vaayu/kv -- the system
# drive, which is the reason this is a knob: until 2026-09-10 the feature above
# could be turned ON but not AIMED.
#
# The cost of getting it wrong is a disk, not a token: the write budget is
# DERIVED as half the free space of whatever filesystem this lands on, so a
# small or shared partition silently caps how much reuse survives a restart,
# and a large one lets the cache grow to hundreds of GB. Point it at the drive
# you want to spend.
KV_PERSIST_DIR = None

# fsync each persisted entry before returning. False (the default) leaves the
# write in the page cache, so a clean shutdown keeps it and a power loss or
# kernel panic can leave a TORN entry that is read back at the next boot. True
# costs a disk flush on the prefix-store path. Inert while ENABLE_KV_PERSIST is
# False.
KV_PERSIST_SYNC = False

# Record which experts are routed and which are evicted, for a triage bundle.
# MoE-only BY CONSTRUCTION -- on a dense checkpoint the engine warns that no
# tracker will be built and does nothing; that is not dense being second-class,
# there are no experts to count.
#
# OFF by default because its cost is UNMEASURED: it puts track_evicted_keys on
# the expert-cache path, which decode hits on every layer visit. Measure your
# own decode both ways before leaving it on in front of users.
ENABLE_EXPERT_USAGE = False

# --------------------------------------------------------------------------
# Engine parameters deliberately NOT exposed here (CLAUDE.md section 8 asks
# that "no knob" be a STATED decision, not a silence). The suite enforces this
# list: a new engine parameter fails tests/test_engine_param_reachability.py
# until it is either given a knob above or added to the decline list there.
#
#   dtype, prefetching             loader plumbing; no measured operator reason
#   layer_names_dict,              dev overrides for a checkpoint the loader
#   tokenizer_class                misreads -- detection is the supported path
#   layer_shards_saving_path       library-facing: where a one-off split goes
#   profiling_mode                 dev instrumentation
#   compression, use_torch_compile `__init__` overrides these in places; they
#                                  are not live paths
#   hf_token                       HF auth belongs in the environment
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Speculative decoding (opt-in)
# --------------------------------------------------------------------------

# Small drafter model. Measured 5.57x on streamed dense. Vocab sizes must match
# the target exactly or the pairing is refused before any weights are read.
# A WEIGHTED drafter is still refused at load on a MoE model whose verify pass
# cannot fit the expert cache -- it has real VRAM and load time to lose, so that
# static pre-check stays. Prompt lookup below has neither and is not refused.
DRAFTER_MODEL_PATH = None

# n-gram speculative decoding, zero VRAM. Measured 1.94x on streamed dense.
# "auto" resolves at load, where is_moe is known: ON for dense, OFF for MoE.
#
# MoE is excluded by DEFAULT and MEASURE BEFORE OPTING IN -- whether it pays
# depends on what the model is emitting, not on which model it is:
#   Qwen3-235B    gate REFUSES to draft at all (off_union: observed verify
#                 union 19.31 GB against a ~17 GB budget, 0 drafting rounds).
#                 Turning it on buys nothing.
#   MiniMax-M2.7  measured BOTH ways on 2026-09-07, one 128-token code prompt:
#                 * server defaults (ENABLE_THINKING = False): acceptance 0.14
#                   against a 0.6259 break-even -> gate latches off_acceptance
#                   after ~1 call, which costs ~10% on THAT call and nothing
#                   after (self.speculator is dropped, later calls take the
#                   ordinary path).
#                 * thinking ON: acceptance 0.6875, gate said run -- and it
#                   still LOST 12.9% total / 16.0% decode wall-clock. That
#                   second case is the defect row 13 fixed (2026-09-10).
#                 An earlier arm on a different prompt measured 1.307x, so this
#                 is workload-dependent, not a property of the model.
# The gate runs its own A/B (row 13, 2026-09-11): its trial alternates 4
# drafting and 4 suppressed rounds for >= 4 pairs and >= 48 drafted tokens,
# and admits only if acceptance clears the expert-byte break-even AND the
# drafting rounds were cheaper per token than their suppressed partners, on
# average over the pairs. It reports `bar_used` ("pooled"; "wall" / "bytes"
# only when too few pairs were timed and the older single-window rule
# decided), `pooled_ratio`, `pairs_used`, `alpha`, `wall_breakeven` and
# `base_from` in /v1/models and env.json. Measured on MiniMax-M2.7: the arm
# the single-window gate admitted and lost -19.7% on is now refused, 3/3
# replicates (forcing it on: -18.6%); the winning arm is still admitted,
# +15.4%. READ THIS BEFORE TRUSTING A `run`: the whole trial still sits in
# the first ~40-60 rounds, the cheapest part of a call, and the refusal
# above rests on the wider window (its first four pairs alone averaged
# 0.99). `run` is a claim about the start of the call, not a promise about
# your whole generation -- TIME YOUR OWN WORKLOAD BOTH WAYS.
# The default stays "auto" because CLAUDE.md section 5 forbids defaulting into
# a one-model-wins asymmetry. Set True to opt one in; an explicit value wins.
USE_PROMPT_LOOKUP = "auto"

# Candidates per speculative round. 5-8 with a drafter, 10-15 with lookup.
# The MoE numbers above were measured at 3 (= a 4-position verify pass), which
# is also the width the VRAM expert-cache prefix is sized for once the gate
# latches run (P9, +4.16% on MiniMax-M2.7 on top of the 1.307x).
DRAFT_TOKENS = 5

# --------------------------------------------------------------------------
# Machine-local overrides. Keep this last: config_local.py is gitignored and
# anything it defines shadows the values above.
# --------------------------------------------------------------------------
try:
    from .config_local import *  # noqa: F401,F403
except ImportError:
    pass
