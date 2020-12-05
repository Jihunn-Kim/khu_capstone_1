import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import math
import copy
import time

import model
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub


def run_benchmark(model_file, img_loader):
    elapsed = 0
    # myModel = torch.jit.load(model_file)
    # torch.backends.quantized.engine='fbgemm'
    # myModel.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    # myModel.eval()
    myModel = model.Net()
    # myModel = torch.quantization.quantize_dynamic(myModel, {torch.nn.Linear, torch.nn.Sequential}, dtype=torch.qint8)
    # print(myModel)
    # set quantization config for server (x86)
    myModel.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    num_batches = 10
    # # insert observers
    torch.quantization.prepare(myModel, inplace=True)
    # # Calibrate the model and collect statistics
    with torch.no_grad():
      for i, (images, target) in enumerate(img_loader):
          images = images.float()
          target = target.long()
          if i < num_batches:
              start = time.time()
              output = myModel(images)
              end = time.time()
              # elapsed = elapsed + (end-start)
          else:
              break

    # # convert to quantized version
    torch.quantization.convert(myModel, inplace=True)

    # quant = QuantStub()
    with torch.no_grad():
      for i, (images, target) in enumerate(img_loader):
          images = images.float()
          target = target.long()
          if i < num_batches:
              start = time.time()
              output = myModel(images)
              end = time.time()
              elapsed = elapsed + (end-start)
          else:
              break
    num_images = images.size()[0] * num_batches
    print(elapsed)
    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed


def init_models(n_nets, args):
    models = []
    layer_shape = []
    layer_type = []

    for idx in range(n_nets):
        models.append(model.Net())

    for (k, v) in models[0].state_dict().items():
        layer_shape.append(v.shape)
        layer_type.append(k)

    return models, layer_shape, layer_type
    