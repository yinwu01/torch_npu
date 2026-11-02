#pragma once
#include "stub_all.h"
inline THPCodeObjectPtr PyFrame_GetCode_NPU(PyFrameObject* frame) { return THPCodeObjectPtr(PyFrame_GetCode(frame)); }
inline PyFrameObject* PyEval_GetFrame_NPU() { return PyEval_GetFrame(); }
inline THPObjectPtr PyFrame_GetLocals_NPU(PyFrameObject* frame) { return THPObjectPtr(PyFrame_GetLocals(frame)); }
