# Missing buffers — when a checkpoint drops part of the model

Some converted checkpoints (quantized builds, re-uploads) leave out tensors the model was trained with. Vaayu detects
this at load, tells you, and can load the missing tensors from the original model.

## What Vaayu checks

A PyTorch module holds two kinds of tensors: **parameters** (weights) and **buffers**. A *persistent* buffer is part of
the model's saved state, just like a weight — the model was trained with it. If a checkpoint has no tensor for one,
nothing restores it, so it silently keeps whatever value the model code initialises it to (usually zeros).

At load, for every persistent buffer, Vaayu looks in the header of the layer file that owns it. Anything missing is
reported once:

```
[Vaayu] WARNING: 62 persistent buffer(s) have no tensor in this checkpoint and keep their initial values
(e_score_correction_bias; e.g. model.layers.0.block_sparse_moe.e_score_correction_bias). The model runs, but not
as trained. Fetch them from the original model repo with:
    ./venv/bin/python scripts/fetch_missing_buffers.py --repo <org/original-model> --checkpoint <model dir>
  then reload.
```

Non-persistent buffers (for example rotary `inv_freq`, which is computed from the config) are never reported. The check
costs nothing on models without persistent buffers — of the checkpoints Vaayu is tested on, only MiniMax-M2.7 has any.

## Why it matters: the MoE routing bias

The case this was built for is **`e_score_correction_bias`**, the per-expert routing bias of MiniMax-M2 (the same
mechanism DeepSeek-V3 introduced; GLM-4.5 and others use it too).

An MoE layer picks a few experts per token — MiniMax-M2.7 picks 8 of 256. Models that balance expert load *without* an
auxiliary training loss do it with a learned bias:

- **Choosing experts:** the router adds the bias to each expert's score, and the 8 highest *biased* scores win.
- **Mixing outputs:** the mixing weights come from the *unbiased* scores.
- **During training** the bias is adjusted so no expert is overloaded or starved. Every training token was routed with
  it, and the experts learned their specialities under that routing.

With the bias missing, tokens are routed by raw scores instead: a set of experts the trained model would not have
picked. Only the *differences* between experts inside a layer matter (a constant offset cancels), and on MiniMax-M2.7
those differences are about the size of the score gap between a token's 7th and 8th choice — so it is mostly the
borderline experts that change.

**Known affected checkpoint:** `NeuroSenko--MiniMax-M2.7-exl3` ships no `e_score_correction_bias` on any of its 62
layers; the original `MiniMaxAI/MiniMax-M2.7` does. exllamav3 looks for the tensor under a different name and treats it
as optional, which is the likely reason the conversion dropped it.

### What it changed on MiniMax-M2.7 (EXL3), measured

Decode-mode perplexity, 48-token prompt then 256 scored single-token steps, one process, same box, same day, the
model's own 8 experts per token. Lower is better. "Zero" is the checkpoint as shipped; "upstream" loads the original
bias.

| text | passage | zero | upstream | change |
|---|---|---|---|---|
| code | Vaayu source (`vaayu/persist/expert_cache.py`) | 32.970 | 29.764 | **−9.7 %** |
| code | Rudra source (`src/rudra/memory/prefetch.py`) | 5.865 | 6.232 | **+6.2 %** |
| code | Python stdlib `textwrap.py` | 1.277 | 1.253 | −1.9 % |
| code | Python stdlib `json/decoder.py` | 1.192 | 1.133 | −5.0 % |
| code | Vaayu's `DECODE_TOP_K="auto"` calibration code | 1.775 | 1.859 | +4.7 % |
| prose | a technical passage (row 33's) | 7.263 | 7.021 | −3.3 % |
| prose | `documentation/architecture.md` | 7.668 | 7.639 | −0.4 % |
| prose | `documentation/TUNING.md` | 29.475 | 30.234 | +2.6 % |
| prose | Rudra `README.md` | 16.659 | 16.096 | −3.4 % |

On average the bias helps (the four code passages measured together: −2.6 %; prose: −1.1 %; 6 of 9 passages better),
but **it is passage-dependent, from −9.7 % to +6.2 %, and 2 of 5 code passages got clearly worse.** It also does not
make a smaller `DECODE_TOP_K` cheaper on code: dropping to 7 experts cost +2.0 % on average without the bias and
+2.6 % with it.

**So the bias is optional, not a recommended default.** A likely reason it is not a clean win: the EXL3 build was
quantized while running *without* the bias (exllamav3 does not find it either), so its experts were tuned for the
unbiased routing. If you run this checkpoint, measure both on text like yours — the file is one command to add and one
delete to remove.

## How to add the missing tensors

1. Load the model with Vaayu once, so the split checkpoint exists, and note the warning.
2. Find the **original** model repo on Hugging Face — the one the converted build was made from (its model card
   usually links it). For MiniMax-M2.7 that is `MiniMaxAI/MiniMax-M2.7`.
3. Run:

   ```bash
   PYTHONPATH=/path/to/Vaayu ./venv/bin/python scripts/fetch_missing_buffers.py \
       --repo MiniMaxAI/MiniMax-M2.7 \
       --checkpoint /path/to/models/NeuroSenko--MiniMax-M2.7-exl3
   ```

   It builds the model skeleton without weights, finds what is missing, and downloads **only those tensors** — each
   shard's header plus the tensor bytes, by HTTP Range request (about 70 KB for MiniMax-M2.7, out of a 481 GB repo).
   The revision is pinned to a commit sha, which it prints. Set `HF_TOKEN` for gated repos. It stops rather than
   download a whole shard if the server ignores Range requests.
4. Restart the server (or reload the model). Vaayu prints
   `[Vaayu] Loaded 62 missing buffer(s) from …/vaayu_missing_buffers.safetensors` and the warning is gone.

**What it writes:** `vaayu_missing_buffers.safetensors` in the model's `splitted_model/` directory. The tensors keep
their original dtype (float32) — rounding the bias to bf16 would blur about a third of the differences that matter.

**To undo:** delete that file and reload.

**License:** the fetched tensors are part of the original model's weights and carry its license (MiniMax-M2.7's is
non-commercial). Vaayu does not ship them for that reason; you download them yourself, as you do the model.

**Side effects:**
- There is no server knob — the file's presence is the switch, so it applies through the server as soon as the model
  reloads.
- If you use `DECODE_TOP_K="auto"`, adding or removing the file makes the cached pick re-measure once on the next load
  (the routing changed, so the old measurement no longer describes the model). On MiniMax-M2.7 that first load takes
  roughly 20–30 minutes.
- The expert offset index rebuilds once (a few seconds) because the split directory changed.
