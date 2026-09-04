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

# Tokens per prefill chunk. 512 is measured best; smaller costs TTFT
# (+24.7% MoE, +213% dense at 128) and buys nothing.
PREFILL_CHUNK_SIZE = 512

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

# None uses sdpa, which every shipped max_seq_len figure was verified on.
# "flash_attention_2" derives a larger context at the same reserve.
ATTN_IMPLEMENTATION = None

# GB of PAGEABLE RAM for the dense layer cache. None derives it from TOTAL RAM
# so throughput is reproducible across restarts (keying off momentary free RAM
# swung Llama-3.3-70B decode 38% run to run). 0 disables it. No effect on MoE,
# whose experts live in the pinned expert RAM cache instead.
RAM_CACHE_BUDGET_GB = None

# Persist prefix KV to disk so reuse survives a restart.
ENABLE_KV_PERSIST = False

# --------------------------------------------------------------------------
# Speculative decoding (opt-in; refused at load on both MoE targets)
# --------------------------------------------------------------------------

# Small drafter model. Measured 5.57x on streamed dense. Vocab sizes must match
# the target exactly or the pairing is refused before any weights are read.
DRAFTER_MODEL_PATH = None

# n-gram speculative decoding, zero VRAM. Measured 1.94x on streamed dense.
USE_PROMPT_LOOKUP = False

# Candidates per speculative round. 5-8 with a drafter, 10-15 with lookup.
DRAFT_TOKENS = 5

# --------------------------------------------------------------------------
# Machine-local overrides. Keep this last: config_local.py is gitignored and
# anything it defines shadows the values above.
# --------------------------------------------------------------------------
try:
    from .config_local import *  # noqa: F401,F403
except ImportError:
    pass
