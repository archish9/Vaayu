# Vaayu Performance Benchmark Report: Era 5 Achieved

This report documents the performance metrics and system improvements observed after implementing the `O_DIRECT` zero-copy `io_uring` read architecture, physically relocating the GPU, and fixing the generation prompt format.

## 1. Top-Level Summary

**Before (Era 3 Baseline):** 
- Total per-token time: **31.1s**
- Total NVMe throughput: **4.4 GB/s**
- Hallucinated outputs due to raw prompt format.

**After (Current Era 5):**
- Total per-token time: **23.2s**
- Total NVMe throughput: **~5.8 GB/s**
- Total generation time for 13 tokens: **5.07 minutes**
- Perfect factual outputs: *"The capital of the United States is Washington, D.C."*

**Total Speedup:** 25% faster generation time per token.

---

## 2. Hardware Bottleneck Breakdowns

The `[VAAYU-IO]` logs give us a perfect diagnostic picture of where the execution time went.

### A. The NVMe I/O Improvement (`io_uring` + `O_DIRECT`)
*   **Previous Throughput:** 4.40 GB/s (using 16-thread Python `os.preadv`)
*   **New Throughput:** **5.50 – 5.95 GB/s** (using Rust `io_uring` + `O_DIRECT`)
*   **Analysis:** The "Read Wide, Slice Narrow" strategy with `_ALIGN = 4096` was a resounding success. The system is completely bypassing the Linux page cache and doing Direct Memory Access (DMA) straight from the Samsung 990 PRO flash chips into the PyTorch pinned memory. The exact 4096-byte padding is visible and working perfectly in the logs.

### B. The PCIe Bandwidth Improvement (`gpu_load`)
*   **Previous Time:** 20.4s
*   **New Time:** **6.9s**
*   **Analysis:** Moving the RTX 5060 Ti from the `PCI_E3` chipset slot to the `PCI_E1` CPU-direct slot caused PCIe transfer time to plummet by 66%. The GPU is now transferring 132 GB of data at approximately **19 GB/s**. 
*   *Note:* While 19 GB/s is drastically better than the old 6.5 GB/s, it indicates the card is likely running at **PCIe 4.0 x8** rather than the full x16 width (which would yield >30 GB/s and drop `gpu_load` to ~2-3s).

### C. The Software Overhead Elimination (`other`)
*   **Previous Time:** 9.7s
*   **New Time:** **0.5s**
*   **Analysis:** We completely eradicated Python overhead. Removing `gc.collect()` from the layer eviction loop stopped the Python garbage collector from pausing execution 83 times per token. Additionally, the Rust `io_uring` module eliminated all thread-pool scheduling overhead.

---

## 3. The "Scary" Metric: Why did `io_wait` go UP?

If you look at the summary, `io_wait` jumped from `0.9s` (Era 3) to `15.8s` (Era 5). **This is actually proof of success.**

In Era 3, the GPU was so choked by the bad PCIe slot (taking 20.4s to load) that the background NVMe prefetch thread always had plenty of time to finish reading the next layer. The main thread rarely had to "wait" for the SSD.

Now, the GPU is blazing fast (6.9s). It finishes transferring the layer to VRAM so quickly that it has to sit and wait for the SSD to catch up. The total time to read 132 GB at 5.8 GB/s is about 22.7 seconds. `15.8s (wait) + 6.9s (gpu_load) = 22.7s`. 

**The SSD is officially the primary bottleneck again, which is exactly where an AirLLM architecture should be!**

---

## 4. Prompt Engineering Fix

The model hallucination is completely resolved. 
*   **The Issue:** Llama 3.3 Instruct models expect a strict `<|start_header_id|>` chat template. Without it, the model thinks it's supposed to continue writing a list of trivia questions.
*   **The Fix:** By wrapping the text in `apply_chat_template`, the model correctly identified its role as an assistant, completely stopping the hallucinated continuation patterns and yielding the factual response.

---

## 5. Next Steps / Remaining Headroom

To push the 23.2s token time down to the ultimate theoretical goal of ~10s, there are two remaining hardware constraints to conquer:

1.  **Saturate the NVMe (5.8 GB/s → 7.4 GB/s):** The Samsung 990 PRO can hit 7.45 GB/s. We may need to tune the `io_uring` queue depth (e.g., QD64) or chunk size in the Rust module to extract the last 1.5 GB/s of performance.
2.  **Fix the PCIe Link Width (8x → 16x):** The PCI_E1 slot appears to be running at 8x width. Checking BIOS settings or Motherboard M.2 lane-sharing rules could bump the `gpu_load` from 6.9s down to ~2s.
