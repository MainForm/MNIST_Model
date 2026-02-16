import argparse
from pathlib import Path

import torch

from model import TestModel


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load a PyTorch model and export to ONNX for ONNX Runtime.'
    )
    parser.add_argument('--modelPath', '-m', required=True, type=str, help='Path to .pt/.pth file')
    parser.add_argument('--onnxPath', '-o', default='model.onnx', type=str, help='Output ONNX path')
    parser.add_argument('--dropoutProb', type=float, default=0.2, help='Dropout prob for TestModel init')
    parser.add_argument('--batchSize', type=int, default=1, help='Dummy input batch size for export')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    return parser.parse_args()


def load_model(model_path: str, dropout_prob: float) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = TestModel(prob=dropout_prob)

        if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
    else:
        raise TypeError(
            f'Unsupported checkpoint type: {type(checkpoint)}. '
            'Expected nn.Module or state_dict dict.'
        )

    model.eval()
    return model


def export_onnx(model: torch.nn.Module, onnx_path: str, batch_size: int, opset: int) -> None:
    dummy_input = torch.randn(batch_size, 1, 28, 28, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )


if __name__ == '__main__':
    args = parse_args()

    model_file = Path(args.modelPath)
    if not model_file.exists():
        raise FileNotFoundError(f'Model file not found: {model_file}')

    onnx_file = Path(args.onnxPath)
    if onnx_file.parent and not onnx_file.parent.exists():
        onnx_file.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(str(model_file), args.dropoutProb)
    export_onnx(model, str(onnx_file), args.batchSize, args.opset)

    print(f'ONNX export complete: {onnx_file.resolve()}')
