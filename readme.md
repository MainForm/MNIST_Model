# MNIST_cnn_basic_model

Basic CNN training and ONNX export project using the MNIST dataset.

## Project Structure

```text
.
|-- train.py          # Train the CNN model on MNIST
|-- model.py          # CNN model definition
|-- convertONNX.py    # Export trained PyTorch model to ONNX
|-- models/           # Saved model files (.pth, .onnx)
`-- datasets/         # Auto-downloaded MNIST dataset
```

## Prerequisites

- Python 3.9+ (recommended)
- pip

Install required packages:

```bash
pip install torch torchvision tqdm matplotlib onnx
```

## How to Run

### 1. Train the model

```bash
python train.py --BatchSize 512 --epochs 10 --learningRate 0.001 --savePath ./models/MNIST.pth
```

What this does:

- Downloads MNIST to `./datasets/MNIST` (if not already present)
- Trains `TestModel` from `model.py`
- Saves the trained weights to `./models/MNIST.pth`

### 2. Export to ONNX

```bash
python convertONNX.py --modelPath ./models/MNIST.pth --onnxPath ./models/model.onnx --batchSize 1 --opset 17
```

What this does:

- Loads the trained PyTorch weights
- Exports an ONNX model with dynamic batch size support
- Saves the file to `./models/model.onnx`

## Optional Arguments

### `train.py`

- `--BatchSize`, `-bs`: training batch size (default: `512`)
- `--epochs`: number of epochs (default: `10`)
- `--learningRate`, `-lr`: learning rate (default: `0.001`)
- `--savePath`: path to save trained weights (default: `None`)

### `convertONNX.py`

- `--modelPath`, `-m`: input `.pth` / `.pt` model path (required)
- `--onnxPath`, `-o`: output ONNX path (default: `model.onnx`)
- `--dropoutProb`: dropout probability for model init (default: `0.2`)
- `--batchSize`: dummy input batch size for export (default: `1`)
- `--opset`: ONNX opset version (default: `17`)
