# 🔌 Running Vaayu so it's reachable from anywhere (Tailscale)

A copy-paste runbook for serving Vaayu to another machine — a laptop, a coding agent
like Rudra, anything — over **Tailscale**. Tailscale is a private WireGuard mesh: every
device you enroll gets a stable `100.x` address and they reach each other **directly and
encrypted from any network on earth**, with no public exposure, no reverse proxy, and
**none of the idle/duration timeouts that drop long streams** on tunnels like ngrok.

> **Why not ngrok / a public tunnel?** Vaayu generations are long-lived SSE streams
> (prefill alone is tens of seconds to minutes). Public HTTP tunnels on free tiers drop
> those mid-stream. A tailnet has no edge in the middle, so there is nothing to time out.
> See `documentation/server.md` §9 and the SSE keepalive knob (`SSE_KEEPALIVE_S`).

---

## This machine's values (fill in / confirm with the commands below)

| What | Value on this box | How to get it |
|---|---|---|
| Vaayu host device | **`a`** | `tailscale status` (the `-` row is this device) |
| Tailnet IPv4 | **`100.94.62.117`** | `tailscale ip -4` |
| Tailnet DNS name | **`a.taila2bfc0.ts.net`** | `tailscale status --json` → `DNSName` |
| Server port | **`9999`** | `PORT` in `vaayu/server/config.py` |
| Example client device | `archishs-macbook-air-2` | `tailscale status` |

If Tailscale is reinstalled or the box re-enrolled, re-run those commands — the IP and
name can change. The `.ts.net` name is stable as long as the device keeps its name.

---

## 0. Prerequisites (one-time)

- Tailscale installed and **up** on BOTH the Vaayu box and the client device, **signed
  into the same account**. Check: `tailscale status` lists both.
- The model is set: `MODEL_PATH` in `vaayu/server/config.py` (or `config_local.py`,
  which is gitignored and shadows it).

Bring Tailscale up if it is down: `sudo tailscale up`.

---

## 1. Start Vaayu

```bash
cd /home/a/code/ai-ml/Vaayu
./venv/bin/python -m vaayu.server
```

The load is not instant (pinning, NVMe warmup, expert-cache sizing — seconds for a small
dense model, minutes for a large MoE one). It is ready when it prints the "listening"
line. Leave it running.

---

## Option A1 — bind Vaayu to the tailnet (plain HTTP, zero setup)

Simplest, works immediately, **no admin console toggle**. Tailscale already encrypts the
wire (WireGuard), so plain `http://` over the tailnet is **not** insecure — HTTPS would
only add a prettier scheme, not real transport security here.

**1. Bind to the tailnet IP.** Edit `vaayu/server/config_local.py` (create it if absent;
it is gitignored):

```python
HOST = "100.94.62.117"   # this box's tailnet IP — binds ONLY the tailscale interface
```

> Use the **tailnet IP**, not `"0.0.0.0"`. `0.0.0.0` also exposes port 9999 to any
> untrusted LAN the box sits on, and **Vaayu has no authentication**. Binding the
> `100.x` IP keeps it reachable only from your tailnet.

**2. Restart Vaayu** (step 1).

**3. From the client**, point the OpenAI base URL at:

```
http://100.94.62.117:9999/v1
```

**Verify from the client device:**

```bash
curl http://100.94.62.117:9999/health
```

---

## Option A2 — `tailscale serve` (HTTPS `.ts.net` URL, Vaayu stays on localhost)

Use this when the client insists on an `https://` URL, or you just want the clean name.
Vaayu stays bound to `127.0.0.1` (most locked-down), and Tailscale fronts it with TLS,
**reachable by your tailnet devices only** (this is *serve*, not *funnel* — not public).

**1. Enable HTTPS for the tailnet** — one-time, in the browser (cannot be done from the
CLI): admin console → **DNS** → **Enable HTTPS…**

`https://login.tailscale.com/admin/dns`

(Without this, `tailscale serve` fails: "HTTPS cert support is not enabled". Confirm it
took with `tailscale cert a.taila2bfc0.ts.net`.)

**2. Leave `HOST = "127.0.0.1"`** (the default — do NOT set the A1 `HOST` line; A1 and A2
are alternatives, not both).

**3. Start Vaayu** (step 1), then on the Vaayu box:

```bash
tailscale serve --bg 9999      # proxy https://a.taila2bfc0.ts.net  ->  127.0.0.1:9999
tailscale serve status         # confirm the mapping and print the URL
```

`--bg` makes it persistent — it survives reboots, so you set it once.

**4. From the client**, point the OpenAI base URL at:

```
https://a.taila2bfc0.ts.net/v1
```

**Verify from the client device:**

```bash
curl https://a.taila2bfc0.ts.net/health
```

**Turn it off:** `tailscale serve reset`.

---

## Option B — truly public (any non-tailnet client): `tailscale funnel`

Only if the client **cannot** join your tailnet. This exposes Vaayu to the **public
internet**.

```bash
tailscale funnel --bg 9999     # public:  https://a.taila2bfc0.ts.net
```

> ⚠️ **Vaayu has NO authentication.** A public Funnel means anyone who learns the URL can
> run inference on your GPU and read/influence whatever flows through it. Do **not** use
> Funnel unless you put an authenticating reverse proxy in front of Vaayu first. For your
> own devices, always prefer A1 or A2 — Funnel buys nothing there and removes the privacy.

Turn it off: `tailscale funnel reset`.

---

## Pick one

| You want… | Use |
|---|---|
| Fastest, client accepts `http://`, your own devices | **A1** |
| HTTPS URL / client requires `https://`, your own devices | **A2** |
| A client that cannot join the tailnet | **B** (with auth added first) |

A1 and A2 are mutually exclusive on the `HOST` setting — A1 binds the tailnet IP, A2 keeps
localhost. Switching from A1 back to A2: remove the `HOST` line from `config_local.py` and
restart.

---

## Client settings that matter (any option)

Vaayu is OpenAI-wire, but these keep long agentic streams healthy:

- **`session_id` (or OpenAI `user`) on every request** — required (`REQUIRE_SESSION_ID`),
  and it drives the cross-turn prefix cache. Use a **stable id per conversation**, with the
  unchanging prompt prefix (system prompt, files) on top.
- **Stream, and ignore SSE comment lines** — any line starting with `:` (e.g.
  `: keepalive`) is a heartbeat the server sends during the silent prefill window so an
  idle timer never fires. Standard SSE clients skip these; a hand-rolled parser must too.
- **Read (inactivity) timeout ≥ 60 s**, comfortably above `SSE_KEEPALIVE_S` (15 s). Do
  **not** set a tight total/overall timeout — a legitimate answer on a slow model runs
  minutes.
- **`finish_reason == "length"` means incomplete** — the cap was hit or the stream was
  cut; do not treat it as a final answer.
- **One request at a time.** Concurrent calls queue (503 past `QUEUE_DEPTH`). Keep an
  agent's calls sequential unless `/v1/models` says fan-out helps.

Model choice is the biggest lever for remote agentic use: a disk-streamed MoE
(MiniMax-M2.7, ~1.8 tok/s) makes every call minutes long; a faster target (Qwen3-Next-80B,
gpt-oss-20b) finishes well inside any timeout. See `documentation/server.md` §7.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `curl http://100.94.62.117:9999/health` hangs from the client | Vaayu not bound to the tailnet (A1 `HOST` not set / not restarted), or Tailscale down on one side (`tailscale status`). |
| `tailscale serve` → "HTTPS cert support is not enabled" | Enable HTTPS in the admin console DNS page (A2 step 1). |
| Client reaches `/health` but streams still drop mid-answer | Not an idle timeout (keepalive covers that) — a **total-duration** limit on the client, or genuinely too-long generation. Raise/disable the client's overall timeout, or serve a faster model. |
| Connection refused on the tailnet IP | Wrong IP (re-check `tailscale ip -4`), or Vaayu bound to `127.0.0.1` while using A1. |
| Works on LAN, not from a remote network | That's fine for Tailscale — it routes over the internet. If it fails, one side's Tailscale is down or a key expired (`sudo tailscale up`). |

---

*Companion docs: `documentation/server.md` (full server reference), `vaayu/server/README.md`
(every config knob with its measured cost).*
