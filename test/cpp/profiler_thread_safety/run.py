"""Build and run the profiler thread-safety stress tests against a source tree.

Two test binaries per build:
- dumper: profiler_thread_safety_test.cpp with the tree's real data_dumper.cpp,
  ring_buffer.h, data_dumper.h and thread.h (stubs/ supplies only npu_log.h, utils.h and
  data_reporter.h).
- pytracer: python_tracer/python_tracer_test.cpp with the tree's real
  torch_npu/csrc/profiler/profiler_python.cpp, built with -DPy_GIL_DISABLED
  (python_tracer/stubs/ stands in for CPython, pybind11, torch and ProfilerMgr). The same
  source is also built once without -DPy_GIL_DISABLED as a build-only check of the
  GIL code path.
Each build runs every scenario --repeats times. Exit code 0 means every run passed: exit 0,
scenario reported passed, and no sanitizer report on stderr. Standard library only.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
DUMPER_CPP = 'torch_npu/csrc/toolkit/profiler/src/data_dumper.cpp'
PYTRACER_CPP = 'torch_npu/csrc/profiler/profiler_python.cpp'

# name -> (test source, stub dir, source file under the tree, extra flags)
BINARIES = {
    'dumper': (HERE / 'profiler_thread_safety_test.cpp', HERE / 'stubs', DUMPER_CPP, []),
    'pytracer': (HERE / 'python_tracer' / 'python_tracer_test.cpp', HERE / 'python_tracer' / 'stubs',
                 PYTRACER_CPP, ['-DPy_GIL_DISABLED']),
}

BUILD_FLAGS = {
    'tsan': ['-O1', '-g', '-fsanitize=thread'],
    'asan': ['-O1', '-g', '-fno-omit-frame-pointer', '-fsanitize=address,undefined'],
    'plain': ['-O2', '-g'],
}

# Sanitizer builds are 5-20x slower, so they get fewer iterations.
# trace uses a small buffer so it fills up: a Push that fails publishes nothing, so only
# Stop()'s wait for writers orders it before UnInit frees the buffer, and TSan reports a
# missing wait (removing WaitWriters() from Stop() fails trace 3 of 3 with capacity 64).
# (scenario name, binary, argv)
SCENARIOS = {
    'tsan': [('rb-cap64', 'dumper', ['rb', '8', '50000', '64']),
             ('rb-cap1024', 'dumper', ['rb', '8', '50000', '1024']),
             ('dumper', 'dumper', ['dumper', '{work}', '6', '100', '1024']),
             ('trace', 'dumper', ['trace', '{work}', '6', '300', '64']),
             ('pytracer', 'pytracer', ['6', '100'])],
    'asan': [('rb-cap64', 'dumper', ['rb', '8', '50000', '64']),
             ('dumper', 'dumper', ['dumper', '{work}', '6', '60', '1024']),
             ('trace', 'dumper', ['trace', '{work}', '6', '150', '64']),
             ('pytracer', 'pytracer', ['6', '100'])],
    'plain': [('rb-cap64', 'dumper', ['rb', '8', '500000', '64']),
              ('rb-cap4096', 'dumper', ['rb', '8', '500000', '4096']),
              ('dumper', 'dumper', ['dumper', '{work}', '8', '200', '1024']),
              ('trace', 'dumper', ['trace', '{work}', '9', '400', '64']),
              ('pytracer', 'pytracer', ['8', '100'])],
}

SANITIZER_MARKERS = ('WARNING: ThreadSanitizer', 'ERROR: AddressSanitizer', 'runtime error:',
                     'ERROR: LeakSanitizer')


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=HERE.parents[2],
                        help='tree holding torch_npu/csrc/... (default: this repo)')
    parser.add_argument('--log-dir', type=Path, default=None,
                        help='where logs go (default: results/<timestamp> next to this file)')
    parser.add_argument('--builds', default='tsan,plain', help='comma list of: ' + ','.join(BUILD_FLAGS))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=120, help='seconds per run')
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    log_dir = (args.log_dir or HERE / 'results' / str(time.time_ns())).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['TSAN_OPTIONS'] = 'halt_on_error=0 exitcode=66 second_deadlock_stack=1'
    env['ASAN_OPTIONS'] = 'detect_leaks=1 exitcode=67'
    env['UBSAN_OPTIONS'] = 'print_stacktrace=1'

    records = []

    def save(label, row):
        text = ['label: ' + label, 'command: ' + ' '.join(row['command']),
                'returncode: ' + str(row['returncode']), 'verdict: ' + row['verdict'],
                '--- stdout ---', row['stdout'], '--- stderr ---', row['stderr']]
        (log_dir / (label + '.log')).write_text('\n'.join(text) + '\n')
        records.append({k: v for k, v in row.items() if k not in ('stdout', 'stderr')})
        print('%-6s %-28s rc=%-4s %s' % (row['verdict'], label, row['returncode'], row.get('reason', '')),
              flush=True)

    def compile_binary(build, name, out, extra):
        test_src, stub_dir, tree_src, flags = BINARIES[name]
        command = (['g++', '-std=c++17', '-pthread', '-Wall', '-Wextra'] + BUILD_FLAGS[build] + flags + extra +
                   ['-I' + str(stub_dir), '-I' + str(source_root), str(test_src), str(source_root / tree_src),
                    '-o', str(out)])
        result = subprocess.run(command, capture_output=True, text=True)
        return command, result

    # Build-only check of profiler_python.cpp's GIL code path (no Py_GIL_DISABLED). The binary
    # is not run: the stubs do not emulate the GIL, so it would race by design.
    command, result = compile_binary('plain', 'pytracer', log_dir / 'test-pytracer-gil', ['-UPy_GIL_DISABLED'])
    save('build-pytracer-gil', {'build': 'plain', 'scenario': 'build', 'command': command,
                                'returncode': result.returncode, 'stdout': result.stdout,
                                'stderr': result.stderr, 'verdict': 'PASS' if result.returncode == 0 else 'FAIL'})

    for build in args.builds.split(','):
        binaries = {}
        for name in BINARIES:
            binary = log_dir / ('test-%s-%s' % (name, build))
            command, result = compile_binary(build, name, binary, [])
            row = {'build': build, 'scenario': 'build', 'command': command, 'returncode': result.returncode,
                   'stdout': result.stdout, 'stderr': result.stderr,
                   'verdict': 'PASS' if result.returncode == 0 else 'FAIL'}
            save('build-%s-%s' % (name, build), row)
            if result.returncode == 0:
                binaries[name] = binary
        for rep in range(args.repeats):
            for name, binary_name, argv in SCENARIOS[build]:
                if binary_name not in binaries:
                    continue
                label = '%s-%s-%d' % (build, name, rep)
                work = log_dir / 'work' / label
                cmd = [str(binaries[binary_name])] + [a.replace('{work}', str(work)) for a in argv]
                row = {'build': build, 'scenario': name, 'repeat': rep, 'command': cmd}
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                                            env=env, preexec_fn=no_core)
                    row.update(returncode=result.returncode, stdout=result.stdout,
                               stderr=result.stderr[:200000])
                except subprocess.TimeoutExpired as exc:
                    row.update(returncode='timeout', stdout=(exc.stdout or b'').decode(errors='replace'),
                               stderr=(exc.stderr or b'').decode(errors='replace')[:200000])
                reasons = []
                if row['returncode'] != 0:
                    reasons.append('returncode %s' % row['returncode'])
                stats = None
                for line in row['stdout'].splitlines():
                    if line.startswith('{'):
                        stats = json.loads(line)
                row['stats'] = stats
                if stats is None:
                    reasons.append('no result line')
                elif not stats.get('passed'):
                    reasons.append('scenario check failed')
                hits = sorted({m for m in SANITIZER_MARKERS if m in row['stderr']})
                if hits:
                    reasons.append('sanitizer: ' + ', '.join(hits))
                row['verdict'] = 'FAIL' if reasons else 'PASS'
                row['reason'] = '; '.join(reasons)
                save(label, row)
                shutil.rmtree(work, ignore_errors=True)

    shutil.rmtree(log_dir / 'work', ignore_errors=True)
    failed = [r for r in records if r['verdict'] != 'PASS']
    summary = {'status': 'passed' if not failed else 'failed', 'source_root': str(source_root),
               'builds': args.builds, 'repeats': args.repeats, 'runs': len(records),
               'failed': len(failed), 'records': records}
    (log_dir / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'status': summary['status'], 'runs': len(records), 'failed': len(failed),
                      'log_dir': str(log_dir)}))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
