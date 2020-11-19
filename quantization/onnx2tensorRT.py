import tensorrt as trt
 
onnx_file_name = 'bert.onnx'
tensorrt_file_name = 'bert.plan'
fp16_mode = True
# int8_mode = True
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
 
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT_BATCH)
parser = trt.OnnxParser(network, TRT_LOGGER)
 
builder.max_workspace_size = (1 << 30)
builder.fp16_mode = fp16_mode
# builder.int8_mode = int8_mode
 
with open(onnx_file_name, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print (parser.get_error(error))

# for int8 mode
# print(network.num_layers, network.num_inputs , network.num_outputs)
# for layer_index in range(network.num_layers):
#   layer = network[layer_index]
#   print(layer.name)
#   tensor = layer.get_output(0)
#   print(tensor.name)
#   tensor.dynamic_range = (0, 255)

  # input_tensor = layer.get_input(0)
  # print(input_tensor)
  # input_tensor.dynamic_range = (0, 255)
 
engine = builder.build_cuda_engine(network)
buf = engine.serialize()
with open(tensorrt_file_name, 'wb') as f:
    f.write(buf)

print('done, trt model')