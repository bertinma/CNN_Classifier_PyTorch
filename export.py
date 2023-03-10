import torch
from model.classifier import Classifier
import argparse
from pathlib import Path 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="weights/cnn.pt")
    parser.add_argument("--num-classes", type=int, default=10)
    return parser.parse_args()

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


if __name__ == '__main__':
    opt = get_args()
    # Instantiate your model. This is just a regular PyTorch model that will be exported in the following steps.
    # Load model
    model = Classifier(
        input_dim=(1, 28, 28),
        num_classes=opt.num_classes
    )
    # Evaluate the model to switch some operations from training mode to inference.
    model.eval()


    # Create dummy input for the model. It will be used to run the model inside export function.
    dummy_input = torch.randn(1, 1, 28, 28)


    model_name = Path(opt.model_path).stem
    # Call the export function
    torch.onnx.export(
        model, 
        (dummy_input, ), 
        f'weights/{model_name}.onnx',
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=11,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input'],   # the model's input names 
        output_names = ['output'], # the model's output names )
    )