"""
Export AlphaZeroNet weights to ONNX for fast CPU inference (onnxruntime).

The exported graph is the raw network: input planes -> (value, policy logits).
The legal-move masking + softmax happens outside the graph (app/onnx_net.py),
mirroring the torch eval path.

Usage:
    python tools/export_onnx.py --weights weights/HPC_20x256.pt
    # writes weights/HPC_20x256.onnx
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AlphaZeroNetwork


class RawAlphaZero(torch.nn.Module):
    """The network without masking/softmax, with a standard forward()."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net.convBlock1(x)
        for block in self.net.residualBlocks:
            x = block(x)
        value = self.net.valueHead(x)
        policy = self.net.policyHead(x)
        return value, policy


def main():
    parser = argparse.ArgumentParser(description='Export AlphaZeroNet to ONNX')
    parser.add_argument('--weights', required=True, help='Path to .pt weights')
    parser.add_argument('--blocks', type=int, default=20)
    parser.add_argument('--filters', type=int, default=256)
    args = parser.parse_args()

    net = AlphaZeroNetwork.AlphaZeroNet(args.blocks, args.filters)
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()
    for p in net.parameters():
        p.requires_grad = False

    raw = RawAlphaZero(net).eval()
    dummy = torch.zeros(1, 16, 8, 8, dtype=torch.float32)

    out_path = os.path.splitext(args.weights)[0] + '.onnx'
    torch.onnx.export(
        raw, dummy, out_path,
        input_names=['input'],
        output_names=['value', 'policy'],
        dynamic_axes={'input': {0: 'batch'}, 'value': {0: 'batch'}, 'policy': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )
    print('Exported to', out_path)

    # Parity check against the torch eval path
    import numpy as np
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=['CPUExecutionProvider'])
    x = torch.randn(4, 16, 8, 8)
    with torch.no_grad():
        tv, tp = raw(x)
    ov, op = sess.run(None, {'input': x.numpy()})
    print('value max diff: ', float(np.abs(tv.numpy() - ov).max()))
    print('policy max diff:', float(np.abs(tp.numpy() - op).max()))


if __name__ == '__main__':
    main()
