# Experimental sequence task queues

## Upstream integration snapshot

This branch starts at Ascend/pytorch master commit
`0d167b0302cbaca25f793a61562dd8f78098d257` and incorporates PR #184 commit
`d538cb535626c2ba8e26541dfbd19f3cc85e4d10`. It keeps upstream's Python
tracer unchanged, including its per-thread caches and callback shutdown.

Only the profiler RingBuffer, DataDumper and TraceDataDumper fixes are
ported from `fix/profiler-thread-safety` at
`82831f4486e7c5973b5b895cf5ff6690410dcaaf`. The old Python tracer changes
and the fork's host-region extensions are not part of this branch.

The PR adapter is adjusted for current upstream: GIL waits use PythonGilBridge,
and the release worker starts lazily for aclop tasks. If worker creation fails,
parameter release falls back to the upstream synchronous path. The queue core
and its standalone tests remain identical to the PR.

Source comparisons and whitespace/conflict checks are the only validation
performed on the Windows integration host. C++ compilation, sanitizer runs,
and Linux/NPU validation are pending. No earlier branch's test results qualify
this new combination. Keep the experimental option OFF until it is validated.

The upload-path regression tests compile the actual ring buffer, dumper and
Linux Thread implementation, with stubs only for reporting and file/log helpers.
They do not validate the upstream Python tracer or real reporting encoders:

```sh
cmake -S test/cpp/profiler_thread_safety -B build/profiler-upload -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/profiler-upload --parallel 2
ctest --test-dir build/profiler-upload --output-on-failure
```

Repeat in fresh build directories with `-DPROFILER_SANITIZER=thread` and
`-DPROFILER_SANITIZER=address`. Also run the queue core tests below. Full builds
with the experimental option OFF and ON must both be tested on the target NPU
with repeated profiler sessions, multiple producers, and with_stack workloads
on the actual Python/PyTorch/CANN combination.

The opt-in bounded MPSC queue uses per-slot sequence numbers for producer
payload transfer, allowing producers to copy into independently reserved slots
without the old enqueue transfer mutex. Repository still executes one task at a time;
ReleaseQueue still retires copied parameters after execution. Task callbacks,
parameter ownership, stream ordering and error reporting remain in the adapter.
This change does not implement a CANN-internal queue.

## Build and compatibility

The original implementation remains the default. To build the experimental
implementation, use a clean build directory and set:

```sh
TORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE=ON python setup.py build
```

The setup entry point forwards the same option to the CMake libraries and the
separately compiled Python extension. Direct CMake builds can pass
`-DTORCH_NPU_EXPERIMENTAL_LOCKFREE_QUEUE=ON`. Rebuild every translation unit
that includes NPUQueue.h and all consumers of its concrete classes: this
compile-time option changes private class layouts. Do not mix libraries or
extensions built with different option values.

## Data flow

```mermaid
sequenceDiagram
    participant P as Producers
    participant Q as Repository slots
    participant C as Task consumer
    participant R as ReleaseQueue
    P->>Q: Reserve ticket and copy current parameters
    P->>Q: Release-publish slot generation
    Q->>C: Acquire published head slot
    C->>C: Execute original task callback
    C->>R: Transfer release parameters
    C->>Q: Complete callback and return slot generation
    R->>R: Invoke original retirement callback
```

A producer owns a reserved slot until it publishes either its complete payload
or an aborted-slot marker. The consumer observes published payload through an
acquire load. A slot cannot be reused until the callback finishes. Draining
must observe completion, rather than only the consumer claiming the head.
Production capacities remain 4096 tasks and 8192 release slots.

The queue is bounded. Full producers wait and stop requests reject new work.
The first recorded failure is propagated to waiters; copying or consuming a
failed item must not strand a reserved ticket. Discarding metadata is distinct
from invoking parameter-release callbacks. Repository shutdown coordinates
its worker threads before freeing their backing parameter storage. Restarting
a core requires the previous producer and consumer calls to have finished.

Stop and join all queue workers while the Python interpreter is still alive;
concurrent interpreter finalization is not a supported shutdown protocol.

Error snapshots and lifecycle transitions still use synchronization. A mutex
protects the single-consumer lease and head inspection; producers publish their
payload without acquiring that mutex. Ticket exhaustion is checked before a
reservation can wrap and raises an overflow error; stop/join/restart is required
for a new generation. An unpublished earlier ticket can delay later tickets; neither this queue nor the
whole adapter claims a formal lock-free or wait-free progress guarantee.

## Standalone verification

The core tests require C++17 and pthreads, without PyTorch or an NPU:

```sh
cmake -S test/cpp/queue -B build/queue-tests
cmake --build build/queue-tests --parallel 2
ctest --test-dir build/queue-tests --output-on-failure
```

These tests cover host-side concurrency and ownership protocols. Full adapter
builds, real-device execution, error recovery and representative workloads are
separate integration checks. Host-simulator timing does not establish a
real-NPU performance improvement. This experimental option should remain off
in production until those checks and community review are complete.
