"""
Vaayu tuning advisor — tells you what to set, for YOUR model and YOUR workload.

Answers the question the RESULTS docs cannot: "given the model I run and the
way I use it, which knobs should I turn on?"

Everything is DERIVED at run time — model geometry from the config, pool budget
from live free RAM via the engine's own `compute_prefix_cache_budget`, VRAM from
the device. No numbers from any particular box are baked in, so this stays
correct when the model or the machine changes.

Loads NO weights. Runs in seconds, needs no GPU.

Run:
    source venv/bin/activate
    python tune_advisor.py                       # uses the defaults below
    K_SESSIONS=4 TOKENS_PER_SESSION=8000 python tune_advisor.py

See documentation/TUNING.md for the reasoning behind each recommendation.
"""
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ── Describe your workload ───────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3")
# How many conversations are alive at once, each with its own session_id.
# One agent with sub-agents fanning out = one session per sub-agent.
K_SESSIONS = int(os.environ.get("K_SESSIONS", "3"))
# Tokens of context each of those carries (system prompt + files + history).
TOKENS_PER_SESSION = int(os.environ.get("TOKENS_PER_SESSION", "8000"))
# Bits/channel for pooled-KV compression when it is recommended.
COMPRESS_BITS = int(os.environ.get("COMPRESS_BITS", "9"))
# The pool budget is computed at MODEL-LOAD time, after the expert RAM cache has
# taken its share -- so free RAM measured now (before loading) badly overstates
# it. If you have run the model, read the real figure off the startup line
#     [Vaayu] cross-turn prefix cache ENABLED: X.XX GB budget
# and pass it here. Without it this script falls back to an estimate and says so.
#
# Pass it. The budget is NOT a property of the box, and two things move it:
#   - The model. Post-Plan-J (2026-08-05) MiniMax measures 11.49-13.76 GB while
#     Qwen3-235B measures 4.96 GB on the same box -- Plan J's arena clamp fires
#     for one geometry and is correctly skipped for the other. A single "typical"
#     figure is now meaningless across models.
#   - The run. It is `free RAM at load - reserve`, and free RAM at load depends
#     on what ran before: 11.49 / 12.16 / 12.87 / 13.70 / 13.76 GB across five
#     loads of the SAME model on the SAME day.
# (The 3.5-5.4 GB range this comment used to quote was measured before Plan J
# reclaimed ~8.6 GB of pinned-allocator round-up, and understates MiniMax by
# ~2.5x. Any estimate here will be wrong for one model or the other.)
POOL_BUDGET_GB = float(os.environ["POOL_BUDGET_GB"]) if os.environ.get("POOL_BUDGET_GB") else None

# DECIMAL GB, to match what the engine prints. Every GB figure this script reads
# or echoes comes from a vaayu_base.py log line that divides by 1e9 -- the prefix
# pool budget, free RAM, _VRAM_RESERVE_BYTES. Using 1024**3 here silently inflated
# a pasted POOL_BUDGET_GB by 7.4%, which is enough to move the session count
# across a boundary: at MiniMax's measured 11.49 GB pool it reported room for 6
# sessions where only 5 fit -- and overshooting pool capacity costs 100% of the
# reuse, not a share of it (todo.md "Settled", Plan J Task 4).
GB = 1e9
MiB = 1024 ** 2   # display only, for per-session KV


def pool_capacity(pool_bytes: float, per_session_bytes: float) -> int:
    """Whole sessions the pool holds. Floor, never negative.

    This is the number the whole script turns on, and it is where the units bug
    lived: `pool_bytes` must be DECIMAL bytes, matching the engine's own
    `X.XX GB budget` startup line, or the count comes out one too high near a
    boundary. Overshooting costs 100% of the reuse, not a share (Plan J Task 4).
    """
    if per_session_bytes <= 0:
        return 0
    return max(0, int(pool_bytes // per_session_bytes))


def recommend_compress_bits(pool_bytes: float, per_sess_bf16: float,
                            per_sess_comp: float, k_sessions: int,
                            compress_bits: int) -> int | None:
    """The `prefix_compress=` value to recommend, or None for uncompressed.

    None when a single session is live (the pool always keeps one entry
    regardless of budget, so it is cached already and compression is pure cost),
    or when every session fits uncompressed (compression would add per-turn work
    and buy nothing). Otherwise the compressed width — including the case where
    even compressed does not fit, since it still widens the range and the caller
    reports the shortfall separately.
    """
    if k_sessions < 2:
        return None
    if k_sessions * per_sess_bf16 <= pool_bytes:
        return None
    return compress_bits


def main() -> int:
    import psutil
    from transformers import AutoConfig
    from vaayu.vaayu_base import compute_prefix_cache_budget, _VRAM_RESERVE_FLOOR_BYTES

    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    layers = cfg.num_hidden_layers
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = cfg.hidden_size // cfg.num_attention_heads

    # Bytes per token of KV, whole model. K and V, every layer, every KV head.
    elems_per_token = layers * 2 * kv_heads * head_dim
    bf16_rate = 2.0                                     # bytes/element
    # TurboQuant pooled layout: (bits-1) index bits + 1 QJL sign bit per channel,
    # plus two fp16 norms per (token, head).
    comp_rate = COMPRESS_BITS / 8 + 4 / head_dim
    bf16_per_tok = elems_per_token * bf16_rate
    comp_per_tok = elems_per_token * comp_rate

    free_ram = psutil.virtual_memory().available
    if POOL_BUDGET_GB is not None:
        pool = int(POOL_BUDGET_GB * GB)
        pool_src = f"measured, from your run's startup line"
    else:
        pool = compute_prefix_cache_budget(free_ram)
        pool_src = "ESTIMATED — see the warning below"

    print(f"MODEL   {os.path.basename(MODEL_PATH)}")
    print(f"        {layers} layers x {kv_heads} KV heads x head_dim {head_dim}")
    print(f"WORKLOAD {K_SESSIONS} concurrent session(s) x {TOKENS_PER_SESSION} tokens "
          f"= {K_SESSIONS * TOKENS_PER_SESSION} combined")
    print(f"MACHINE free RAM {free_ram/GB:.2f} GB -> prefix pool budget "
          f"{pool/GB:.2f} GB  ({pool_src})")
    print()
    if POOL_BUDGET_GB is None:
        print("!! WARNING: that budget is measured BEFORE the model loads, and the")
        print("   engine computes the real one AFTER the expert RAM cache is")
        print("   allocated. The gap is large AND model-dependent -- on the")
        print("   development box, 23 GB free before load became 11.49-13.76 GB")
        print("   of pool on MiniMax but 4.96 GB on Qwen3-235B, and it varied by")
        print("   2.3 GB across five loads of the same model on the same day.")
        print("   This run's advice is therefore OPTIMISTIC: it will under-recommend")
        print("   compression. For a real answer, load the model once, read")
        print("     [Vaayu] cross-turn prefix cache ENABLED: X.XX GB budget")
        print("   and re-run as:  POOL_BUDGET_GB=X.XX python tune_advisor.py")
        print()

    if pool <= 0:
        print("!! Prefix pool budget is 0 — free RAM is below the engine's reserve.")
        print("   Close something, or expect one cached session at most.")

    per_sess_bf16 = TOKENS_PER_SESSION * bf16_per_tok
    per_sess_comp = TOKENS_PER_SESSION * comp_per_tok
    need_bf16 = K_SESSIONS * per_sess_bf16
    need_comp = K_SESSIONS * per_sess_comp

    print(f"KV per session   bf16 {per_sess_bf16/MiB:8.1f} MiB   "
          f"{COMPRESS_BITS}-bit {per_sess_comp/MiB:8.1f} MiB "
          f"({comp_rate/bf16_rate:.3f}x)")
    print(f"KV for all {K_SESSIONS:>2}    bf16 {need_bf16/MiB:8.1f} MiB   "
          f"{COMPRESS_BITS}-bit {need_comp/MiB:8.1f} MiB")
    if pool > 0:
        print(f"Pool holds       bf16 {pool_capacity(pool, per_sess_bf16):>3} session(s)     "
              f"{COMPRESS_BITS}-bit {pool_capacity(pool, per_sess_comp):>3} session(s)")
        print(f"Token ceiling    bf16 {int(pool / bf16_per_tok):>7,} tok   "
              f"{COMPRESS_BITS}-bit {int(pool / comp_per_tok):>7,} tok  (summed over sessions)")
    print()

    # ── The recommendation ───────────────────────────────────────────────────
    rec = recommend_compress_bits(pool, per_sess_bf16, per_sess_comp,
                                  K_SESSIONS, COMPRESS_BITS)
    fits_bf16 = need_bf16 <= pool
    fits_comp = need_comp <= pool
    single = K_SESSIONS < 2

    print("=" * 72)
    if single:
        verdict = "prefix_compress=None"
        why = ("only one session. The pool always keeps one entry regardless of "
               "budget, so it is already cached and compression would be pure cost.")
    elif fits_bf16:
        verdict = "prefix_compress=None"
        why = (f"all {K_SESSIONS} sessions already fit uncompressed "
               f"({need_bf16/MiB:.0f} of {pool/MiB:.0f} MiB). Compression would add "
               f"work per turn and buy nothing.")
    elif fits_comp:
        verdict = f"prefix_compress={COMPRESS_BITS}   <-- THIS IS YOUR CASE"
        why = (f"uncompressed needs {need_bf16/MiB:.0f} MiB and the pool has "
               f"{pool/MiB:.0f} MiB, so sessions get evicted and every turn "
               f"re-prefills from scratch. Compressed needs {need_comp/MiB:.0f} MiB "
               f"and fits. Measured 34x TTFT in exactly this situation.")
    else:
        verdict = f"prefix_compress={COMPRESS_BITS} (helps, but will still thrash)"
        why = (f"even compressed you need {need_comp/MiB:.0f} MiB against a "
               f"{pool/MiB:.0f} MiB pool. Compression widens the range but does not "
               f"close this gap — reduce concurrent sessions or context length, or "
               f"free RAM.")
    print(f"RECOMMENDED: {verdict}")
    print(f"WHY: {why}")
    print("=" * 72)
    print()

    # ── Reuse warning: caching only pays if turns share a prefix ─────────────
    print("NOTE: all of this assumes your turns SHARE a prefix (same system prompt")
    print("      and mostly-unchanged files). If every request is unrelated, the")
    print("      prefix cache cannot hit and none of these settings help.")
    print()

    # ── VRAM sanity: the live cache is bf16 no matter what ──────────────────
    try:
        import torch
        if torch.cuda.is_available():
            free_vram, _total = torch.cuda.mem_get_info()
            live = per_sess_bf16      # the in-flight session is never compressed
            print(f"VRAM    free {free_vram/GB:.2f} GB; the LIVE session's KV is "
                  f"{live/GB:.2f} GB (always bf16 — compression applies only to the pool)")
            if live > free_vram - _VRAM_RESERVE_FLOOR_BYTES:
                print(f"        !! At {TOKENS_PER_SESSION} tokens the live KV alone may not fit "
                      f"beside the {_VRAM_RESERVE_FLOOR_BYTES/GB:.1f} GB reserve floor (the "
                      f"real reserve grows with context length).")
                print(f"        Shrinking the pool cannot help — that needs attention over")
                print(f"        compressed keys, which is not built. Reduce context length.")
    except Exception as e:
        print(f"VRAM    check skipped ({e!r})")
    print()

    print("Copy into your AutoModel.from_pretrained(...):")
    print()
    print("    model = AutoModel.from_pretrained(")
    print(f"        MODEL_PATH,")
    print(f"        enable_prefix_cache=True,      # required for any of this")
    print(f"        prefix_compress={rec},")
    print(f"        enable_kv_persist=True,        # survives restarts; costs disk only")
    print(f"        kv_cache_mode=\"auto\",          # bf16; do not use \"turboquant\"")
    print("    )")
    print()
    print("Full reasoning and the other knobs: documentation/TUNING.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
