import model
import torch

import importlib
importlib.reload(model)

batch_size = 256
model = model.Net().cuda().eval()
inputs = torch.randn(batch_size, 1, 29, 29, requires_grad=True).cuda()
torch_out = model(inputs)

torch.onnx.export(
    model,
    inputs,
    'bert.onnx',
    input_names=['inputs'],
    output_names=['outputs'],
    export_params=True)

print('done, onnx model')