# file adapted from a ST notebook example

from datetime import datetime
import glob
import os
import random
import shutil
from typing import Dict, List
import numpy as np 
from tqdm import tqdm

import torch
from torch import nn
import torchvision

import onnx
import onnxruntime
from onnx import version_converter
from onnxruntime import quantization
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class ResidualBlock(nn.Module):
    def __init__(self, d, dim_feedforward: int) -> None:
        super(ResidualBlock, self).__init__()
        # self.layer_norm = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d)
    
    def forward(self, x):
        residual = x
        # x = self.layer_norm(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x + residual
    
class AttentionBlock(nn.Module):
    def __init__(self, d, nhead) -> None:
        super(AttentionBlock, self).__init__()
        # self.layer_norm = nn.LayerNorm(d)
        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.n_head = nhead
        self.d = d
    
    def forward(self, x):
        residual = x
        # x = self.layer_norm(x)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        head_dim = self.d // self.n_head
        attn_outputs = []
        for i in range(self.n_head):
            q_i = q[:, :, i * head_dim:(i + 1) * head_dim]
            k_i = k[:, :, i * head_dim:(i + 1) * head_dim]
            v_i = v[:, :, i * head_dim:(i + 1) * head_dim]
            attn_weights = torch.matmul(q_i, k_i.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_outputs.append(torch.matmul(attn_weights, v_i))
        attn_output = torch.cat(attn_outputs, dim=-1)
        attn_output = self.linear_out(attn_output)

        return attn_output + residual
    
class MyTransformer(nn.Module):
    def __init__(self, d, nhead, dim_feedforward, num_layers) -> None:
        super(MyTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AttentionBlock(d, nhead))
            self.layers.append(ResidualBlock(d, dim_feedforward))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
class CallibationDataset(CalibrationDataReader):
    """
    A class used to read calibration data for a given model.

    Attributes
    ----------
    calibration_image_folder : str
        The path to the folder containing calibration images
    model_path : str
        The path to the ONNX model file

    Methods
    -------
    get_next() -> Dict[str, List[float]]
        Returns the next item from the enumerator
    rewind() -> None
        Resets the enumeration of calibration data
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ImageNetDataReader class.

        Parameters
        ----------
        model_path : str
            The path to the ONNX model file
        """

        # Use inference session to get input shape
        session = onnxruntime.InferenceSession(model_path, None)
        input_shape = session.get_inputs()[0].shape
        self.input_name = session.get_inputs()[0].name

        # Generate random calibration data
        self.data_list = [np.random.randn(1, *input_shape[1:]).astype(np.float32) * 1 for _ in range(10)]

        self.enum_data = None  # Initialize enumerator to None


    def get_next(self) -> Dict[str, List[float]]:
        """
        Returns the next item from the enumerator.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the input name and corresponding data
        """

        if self.enum_data is None:
            # Create an iterator that generates input dictionaries
            # with input name and corresponding data
            self.enum_data = iter(
                [{self.input_name: d} for d in self.data_list]
            )
        
        return next(self.enum_data, None)  # Return next item from enumerator

    def rewind(self) -> None:
        """
        Resets the enumeration of calibration data.
        """

        self.enum_data = None  # Reset the enumeration of calibration data

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    os.makedirs("generated_files", exist_ok=True)
    input_model = "generated_files/transformer.onnx"
    infer_model = "generated_files/transformer_infer.onnx"
    quant_model = "generated_files/transformer_quant.onnx"

    model = MyTransformer(d=256, nhead=1, dim_feedforward=256, num_layers=6)

    # export the model to ONNX format
    example_inputs = torch.zeros((1, 196, 256), dtype=torch.float32)
    print(f"Model parameters: {count_parameters(model)}")
    model.eval()
    print(len(list(model.parameters())))

    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
    )
    onnx_program.save(input_model)

    # Quantize the model
    quantization.quant_pre_process(input_model_path=input_model, output_model_path=infer_model, skip_optimization=False)
    dr = CallibationDataset(input_model)
    quantize_static(
            infer_model,
            quant_model,
            dr,
            calibrate_method=CalibrationMethod.MinMax, 
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8, 
            activation_type=QuantType.QInt8, 
            reduce_range=True,
            extra_options={'WeightSymmetric': True, 'ActivationSymmetric': False})

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(current_time + ' - ' + '{} model has been created.'.format(os.path.basename(quant_model)))

    # Run inference with the quantized model
    quantized_session = onnxruntime.InferenceSession(quant_model)
    input_name = quantized_session.get_inputs()[0].name
    label_name = quantized_session.get_outputs()[0].name
    data = example_inputs.numpy()
    result = quantized_session.run([label_name], {input_name: data.astype(np.float32)})[0]
    print(result)