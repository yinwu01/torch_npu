"""在 NPU 机器上检查 profiler 崩溃的候选原因。

依赖：Python 3.14t（free-threaded），torch 2.13.0，已安装待测的 torch_npu，一张 NPU。不需要网络。

用法（每条命令单独运行，记录退出码和最后 30 行输出）：
  python npu_check.py --stack            # 单线程，开 with_stack，调用 nn.Module
  python npu_check.py                    # 单线程，不开 with_stack（对照）
  python npu_check.py --threads 4 --repeat 20           # 多线程，不开 with_stack
  python npu_check.py --threads 4 --repeat 20 --stack   # 多线程，开 with_stack

崩溃时用 gdb 取调用栈：
  gdb -batch -ex run -ex bt --args python npu_check.py --stack
"""

import argparse
import faulthandler
import os
import sys
import tempfile
import threading

faulthandler.enable()

import torch
import torch_npu  # noqa: F401
from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler


def work(model, x, iters):
    for _ in range(iters):
        model(x)  # 经过 nn.Module.__call__
    torch.npu.synchronize()


def run_once(args, out_dir):
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU()).npu()
    x = torch.randn(64, 256, device="npu")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        with_stack=args.stack,
        with_modules=args.stack,
        on_trace_ready=tensorboard_trace_handler(out_dir),
    ):
        if args.threads <= 1:
            work(model, x, args.iters)
        else:
            ts = [threading.Thread(target=work, args=(model, x, args.iters)) for _ in range(args.threads)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stack", action="store_true", help="开启 with_stack 和 with_modules")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=1)
    args = p.parse_args()

    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    print(f"python={sys.version.split()[0]} gil_enabled={gil} torch={torch.__version__} "
          f"torch_npu={torch_npu.__version__} stack={args.stack} threads={args.threads} "
          f"iters={args.iters} repeat={args.repeat}", flush=True)
    for i in range(args.repeat):
        out_dir = tempfile.mkdtemp(prefix="npu_check_")
        run_once(args, out_dir)
        print(f"run {i + 1}/{args.repeat} ok, output {out_dir}", flush=True)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
