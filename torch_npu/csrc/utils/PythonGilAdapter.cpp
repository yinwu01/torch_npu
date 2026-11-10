#include <torch/csrc/python_headers.h>

#include "torch_npu/csrc/core/npu/PythonGilBridge.h"
#include "torch_npu/csrc/utils/PythonGilAdapter.h"

namespace torch_npu::utils {
namespace {
void* releaseIfHeld() {
#if PY_VERSION_HEX >= 0x030D0000
  return PyThreadState_GetUnchecked() != nullptr ? PyEval_SaveThread() : nullptr;
#else
  return _PyThreadState_UncheckedGet() != nullptr ? PyEval_SaveThread() : nullptr;
#endif
}

void restore(void* state) {
  PyEval_RestoreThread(static_cast<PyThreadState*>(state));
}

void runWithGilReleased(c10::function_ref<void()> fn) {
  // Restore on exceptions too; callbacks may report an ACL error by throwing.
  struct RestoreState {
    void* state = releaseIfHeld();
    ~RestoreState() { if (state != nullptr) restore(state); }
  } saved;
  fn();
}

const c10_npu::impl::PythonGilHooks hooks{releaseIfHeld, restore, runWithGilReleased};
} // namespace

void registerPythonGilAdapter() {
  c10_npu::impl::registerPythonGilHooks(&hooks);
}
} // namespace torch_npu::utils
