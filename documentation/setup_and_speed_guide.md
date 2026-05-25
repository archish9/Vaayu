# Vaayu Setup & Hardware Acceleration Guide

This is the definitive handbook to setting up your environment and configuring your hardware and operating system to unlock Vaayu's absolute maximum execution speeds. By combining optimized virtual environment builds, PCIe motherboard corrections, and block-layer bypasses, you will transition Vaayu from a standard streaming script to a high-speed, local-first engine.

---

## Part 1: Environment & Dependency Installation

Installing compilation-heavy libraries like `flash-attn` and `exllamav3` requires a strict build order. Standard `pip install -r requirements.txt` commands often fail because Pip attempts to compile C++/CUDA extensions in isolated sandboxes where PyTorch is invisible.

### 1. Step-by-Step Clean Setup

Execute these commands in the exact order below:

```bash
# 1. Clone the repository and navigate to root
cd Vaayu

# 2. Create and activate a clean Python 3.12 virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install core PyTorch and toolchain utilities first
pip install \
    "torch>=2.6.0" \
    "transformers==5.5.4" \
    "accelerate>=0.28.0" \
    "safetensors>=0.4.0" \
    "optimum>=1.17.0" \
    "huggingface-hub>=0.20.0" \
    "bitsandbytes>=0.43.0" \
    ninja tqdm einops scipy sentencepiece peft evaluate scikit-learn wandb kernels maturin Pillow

# 4. Install flash-attn using build-isolation bypass (compiles CUDA kernels, 5-15 mins)
pip install flash-attn --no-build-isolation

# 5. Install exllamav3 from GitHub source to compile native CUDA modules
pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation
# Note: If your system OOMs during compilation, restrict job compilation cores:
# MAX_JOBS=4 pip install git+https://github.com/turboderp-org/exllamav3 --no-build-isolation

# 6. Fetch exllamav3 calibration standard UTF-8 files
CAL_DIR="$(python3 -c "import exllamav3.conversion.calibration_data as m; import os; print(os.path.join(os.path.dirname(os.path.abspath(m.__file__)), 'standard_cal_data'))")"
BASE="https://raw.githubusercontent.com/turboderp-org/exllamav3/master/exllamav3/conversion/standard_cal_data"
for f in c4.utf8 code.utf8 multilingual.utf8 technical.utf8 tiny.utf8 wiki.utf8; do
    echo "Downloading calibration file: $f..."
    curl -s -o "$CAL_DIR/$f" "$BASE/$f"
done
```

---

## Part 2: Compiling the Native Rust `vaayu_rust` Extension

The zero-copy `io_uring` disk-streaming engine lives inside the native C-extension `vaayu_rust`. 

```bash
# 1. Navigate to the Rust package directory
cd vaayu_rust

# 2. Compile and develop the Rust extension in release mode
# (Maturin compiles Cargo assets and installs the shared library (.so) into your active venv)
maturin develop --release

# 3. Verify the installation
cd ..
python3 -c "import vaayu_rust; print('✅ vaayu_rust extension compiled & loaded successfully!')"
```

---

## Part 3: The Critical PCIe Motherboard Relocation

Software optimizations are useless if your GPU is choked by a narrow hardware bottleneck. If you log Vaayu runs and see `gpu_load` consuming ~20 seconds per token, your GPU is installed in the wrong slot.

### The Motherboard Topology Bug
Many motherboards have multiple full-length PCIe slots, but only the top slot is wired directly to the CPU. Lower slots route through the Motherboard Chipset, sharing lanes and restricting bandwidth.

* **Incorrect Slot (PCI_E3)**: Routes through the Z790 chipset, limiting link speeds to **PCIe 4.0 x4** (~6.5 GB/s transfer bandwidth). Moving 132 GB takes **20.4 seconds**.
* **Correct Slot (PCI_E1)**: Connected directly to the CPU, engaging the full **PCIe 5.0 x16** bandwidth (~64 GB/s). Shifting the GPU to this slot drops `gpu_load` from **20.4s to ~2s**.

### How to Relocate and Verify
1. Power down your machine, unplug the power supply, and open the PC chassis.
2. Locate your GPU in the lower slots. Unscrew the mounting bracket, release the plastic PCIe lock latch, and carefully pull the GPU out.
3. Insert the GPU into the **topmost metal-armored PCIe slot (labeled PCI_E1 on the motherboard PCB)**. Press firmly until the latch clicks, and screw the bracket back in.
4. Boot up, and run this terminal check:
   ```bash
   nvidia-smi -q | grep -A 10 "Link Width"
   ```
   * **Victory**: Under `Link Width`, both `Max` and `Current` should report **16x**. (If it reports `4x`, the GPU is still in a chipset-limited slot or PCIe power saving is active).

---

## Part 4: OS-Level NVMe Kernel Bypasses (Linux)

You can extract an extra **5% to 15% sequential throughput** directly from your NVMe drive (like the Samsung 990 PRO) by turning off legacy kernel processing and power management structures.

### 1. Set I/O Scheduler to `none`
By default, Linux uses the `mq-deadline` scheduler. This scheduler attempts to merge and reorder block requests before sending them to disk. Modern NVMe controllers have high-powered internal processors that handle scheduling far better. Setting the scheduler to `none` eliminates this redundant software layer.

```bash
# Check current scheduler (active is wrapped in brackets e.g. [mq-deadline])
cat /sys/block/nvme0n1/queue/scheduler

# Shorthand optimized OS script provided in Vaayu:
sudo bash scripts/vaayu_optimize_os.sh /path/to/your/model/directory
```

*What the script does under the hood:*
It writes a permanent udev rule to `/etc/udev/rules.d/60-vaayu-nvme.rules` so that all NVMe drives are automatically bound to the `none` scheduler on system boot.

### 2. Disable Autonomous Power State Transition (APST)
During Vaayu's layer-streaming generation loop, the NVMe SSD experiences short idle phases while the GPU is executing computations. The operating system's power manager (APST) will aggressively put the SSD into a micro-sleep state. Waking the drive up on the next layer read introduces **50 to 500 microseconds of wakeup latency**. Over 80+ layers, these wakeups cause significant latency jitter.

To disable APST permanently:
1. Open `/etc/default/grub` in your preferred text editor (with sudo).
2. Append the default latency limit parameters to `GRUB_CMDLINE_LINUX_DEFAULT`:
   ```text
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash nvme_core.default_ps_max_latency_us=0"
   ```
3. Update GRUB and reboot:
   ```bash
   sudo update-grub
   sudo reboot
   ```

### 3. Verify the OS Tuning Settings
After rebooting, verify that both optimizations are active:
```bash
# Scheduler check: should output '[none]'
cat /sys/block/nvme0n1/queue/scheduler

# APST check: should output '0'
cat /sys/module/nvme_core/parameters/default_ps_max_latency_us
```
*(Replace `nvme0n1` with your actual system boot drive identifier if different).*
