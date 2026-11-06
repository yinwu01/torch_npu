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
import shutil
import sys
import tempfile
import threading

faulthandler.enable()

import torch
import torch_npu  # noqa: F401
from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch_npu.profiler import ProfilerActivity, profile, tensorboard_trace_handler, _ExperimentalConfig, ExportType

def work(model, x, iters):
    for _ in range(iters):
        model(x)  # 经过 nn.Module.__call__
    torch.npu.synchronize()



def run_once(args, out_dir):
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU()).npu()
    x = torch.randn(64, 256, device="npu")
    extra = {}
    if args.export == "db":
        # Db 模式下 CANNTimelineParser 等 msprof_*.db，不等 csv
        extra["experimental_config"] = _ExperimentalConfig(export_type=[ExportType.Db])
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        with_stack=args.stack,
        with_modules=args.stack,
        on_trace_ready=tensorboard_trace_handler(out_dir),
        **extra,
    ):
        if args.threads <= 1:
            work(model, x, args.iters)
        else:
            ts = [threading.Thread(target=work, args=(model, x, args.iters)) for _ in range(args.threads)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

def summarize(out_dir):
    """打印每个 *_ascend_pt 目录的产物清单。"""
    for pt in sorted(os.listdir(out_dir)):
        root = os.path.join(out_dir, pt)
        if not pt.endswith("_ascend_pt") or not os.path.isdir(root):
            continue
        print(f"  {pt}", flush=True)
        out = os.path.join(root, "ASCEND_PROFILER_OUTPUT")
        if os.path.isdir(out):
            for name in sorted(os.listdir(out)):
                print(f"    ASCEND_PROFILER_OUTPUT/{name} {os.path.getsize(os.path.join(out, name))}", flush=True)
        else:
            print("    ASCEND_PROFILER_OUTPUT: missing", flush=True)
        dbs = [os.path.relpath(os.path.join(d, f), root)
               for d, _, files in os.walk(root) for f in files
               if f.startswith("msprof_") and f.endswith(".db")]
        print(f"    msprof_*.db: {len(dbs)} {dbs}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stack", action="store_true", help="开启 with_stack 和 with_modules")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--keep-output", action="store_true", help="保留每次的 profiler 输出（默认跑完即删，避免占满磁盘）")
    p.add_argument("--export", choices=["text", "db"], default="text", help="解析结果的导出格式")
    args = p.parse_args()

    gil = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
    print(f"python={sys.version.split()[0]} gil_enabled={gil} torch={torch.__version__} "
          f"torch_npu={torch_npu.__version__} stack={args.stack} export={args.export} threads={args.threads} "
          f"iters={args.iters} repeat={args.repeat}", flush=True)
    for i in range(args.repeat):
        out_dir = tempfile.mkdtemp(prefix="npu_check_")
        run_once(args, out_dir)
        summarize(out_dir)
        print(f"run {i + 1}/{args.repeat} ok, output {out_dir}", flush=True)
        if not args.keep_output:
            shutil.rmtree(out_dir, ignore_errors=True)
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
