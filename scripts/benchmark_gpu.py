"""Time training steps per model and input size (synthetic data) to plan the compute budget.

    python -m scripts.benchmark_gpu            # GPU if available, else CPU

Run with the laptop on AC power: on battery the GPU stays in a low-power state and
timings are not representative. Batch size shrinks with resolution so memory stays
roughly constant (avoids Windows silently spilling VRAM into system RAM).
"""

from __future__ import annotations

import argparse
import time

import torch

from medxai.models import build_model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["small_cnn", "resnet18"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[28, 64, 128, 224])
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}")
    for name in args.models:
        for size in args.sizes:
            bs = max(4, int(64 * (64 / max(size, 64)) ** 2))
            if dev == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            m = build_model(name, in_channels=args.channels, num_classes=args.classes).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            x = torch.randn(bs, args.channels, size, size, device=dev)
            y = torch.randint(0, args.classes, (bs,), device=dev)
            for i in range(args.steps + 3):  # 3 warm-up steps
                if i == 3:
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t = time.time()
                opt.zero_grad()
                torch.nn.functional.cross_entropy(m(x), y).backward()
                opt.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t) / args.steps / bs * 1000
            peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
            print(f"{name:10s} {size:4d}px bs={bs:3d} {ms:9.3f} ms/img  peak {peak:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
