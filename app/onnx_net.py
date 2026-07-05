"""
ONNX Runtime drop-in for AlphaZeroNet inference on CPU.

Exposes the exact calling convention MCTS/encoder expect from the torch
network — net(x, policyMask=...) -> (value, masked policy softmax) as torch
tensors — but runs the network through onnxruntime, which is substantially
faster than eager PyTorch on CPU-only hosts.

Create the .onnx file with tools/export_onnx.py.
"""

import os

import numpy as np
import torch


class OnnxAlphaZero:

    def __init__(self, onnx_path, threads=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads or (os.cpu_count() or 2)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, options, providers=['CPUExecutionProvider'])

    def __call__(self, x, valueTarget=None, policyTarget=None, policyMask=None):
        inputs = x.detach().cpu().numpy().astype(np.float32, copy=False)
        value, policy = self.session.run(None, {'input': inputs})

        # Masked softmax over legal moves, mirroring AlphaZeroNet's eval path.
        mask = policyMask.detach().cpu().numpy().reshape(policy.shape[0], -1)
        policy_exp = np.exp(policy) * mask.astype(np.float32, copy=False)
        policy_softmax = policy_exp / policy_exp.sum(axis=1, keepdims=True)

        return torch.from_numpy(value), torch.from_numpy(policy_softmax)
