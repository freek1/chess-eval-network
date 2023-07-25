import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from chess_evaluation_prediction import ChessDataset, Geohotz
import pandas as pd
from collections import OrderedDict
import numpy as np

def convert_weights_to_onnx(model_state_dict, input_shape, output_path):
    # Create an instance of your model
    # Replace 'YourModel' with the actual class name of your model.
    model = Geohotz()

    # Load the state_dict into the model
    model.load_state_dict(model_state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Create a sample input tensor
    bbdf = pd.read_csv('processed_chessData.csv')
    bbdf = bbdf[0:10]
    l = len(bbdf)
    train_df, test_df, cv_df = bbdf[:int(.8 * l)], bbdf[int(.8 * l): int(.9 * l)], bbdf[int(.9 * l):]
    d_train, d_test, d_cv = map(ChessDataset, [train_df, test_df, cv_df])
    
    dummy_input = torch.randn((3, 29, 8, 8))

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, output_path, opset_version=11)

if __name__ == "__main__":
    # Replace 'path_to_saved_model.pt' with the path to your saved model file.
    path_to_saved_model = 'geohotz_20k.pt'

    # Replace (3, 224, 224) with the input shape of your model.
    # Typically, this would be (C, H, W) for an image-based model.

    # Replace 'path_to_output.onnx' with the desired output path and filename for the ONNX file.
    output_path = 'geohotz_20k.onnx'

    # Load the saved model state_dict
    saved_state_dict = torch.load(path_to_saved_model)

    # Convert the weights to ONNX format
    convert_weights_to_onnx(saved_state_dict, (29,8,8), output_path)
    
    
    # Conversion from gpu to cpu model:
    model = Geohotz()
    model.load_state_dict(torch.load('geohotz_20.pt', map_location='cpu'))
    torch.save(model.state_dict(), 'geohotz_20k_cpu.pt')
