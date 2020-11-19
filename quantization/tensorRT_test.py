import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import torch
import pycuda.autoinit
import dataset
import model
import time
# print(dir(trt))
 
tensorrt_file_name = 'bert.plan'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
 
with open(tensorrt_file_name, 'rb') as f:
    engine_data = f.read()
engine = trt_runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem
 
#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
 
#     def __repr__(self):
#         return self.__str__()
 
# inputs, outputs, bindings, stream = [], [], [], []
# for binding in engine:
#     size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
#     dtype = trt.nptype(engine.get_binding_dtype(binding))
#     host_mem = cuda.pagelocked_empty(size, dtype)
#     device_mem = cuda.mem_alloc(host_mem.nbytes)
#     bindings.append(int(device_mem))
#     if engine.binding_is_input(binding):
#         inputs.append( HostDeviceMem(host_mem, device_mem) )
#     else:
#         outputs.append(HostDeviceMem(host_mem, device_mem))

# input_ids = np.ones([1, 1, 29, 29])
 
# numpy_array_input = [input_ids]
# hosts = [input.host for input in inputs]
# trt_types = [trt.int32]
 
# for numpy_array, host, trt_types in zip(numpy_array_input, hosts, trt_types):
#     numpy_array = np.asarray(numpy_array).ravel()
#     np.copyto(host, numpy_array)

# def do_inference(context, bindings, inputs, outputs, stream):
#     [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#     [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#     stream.synchronize()
#     return [out.host for out in outputs]

# trt_outputs = do_inference(
#                         context=context,
#                         bindings=bindings,
#                         inputs=inputs,
#                         outputs=outputs,
#                         stream=stream)

def infer(context, input_img, output_size, batch_size):
    # Load engine
    # engine = context.get_engine()
    # assert(engine.get_nb_bindings() == 2)
    # Convert input data to float32
    input_img = input_img.astype(np.float32)
    # Create host buffer to receive data
    output = np.empty(output_size, dtype = np.float32)
    # Allocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.execute_async(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Synchronize threads
    stream.synchronize()
    # Return predictions
    return output


# kwargs = {"./dataset/DoS_dataset.csv" : './DoS_dataset.txt'}
# train_data_set, data_idx_map, net_class_count, net_data_count, test_data_set = dataset.GetCanDatasetUsingTxtKwarg(100, 0, **kwargs)
# testloader = torch.utils.data.DataLoader(test_data_set, batch_size=256,
#                                         shuffle=False, num_workers=2)

check_time = time.time()
cnt = 0
temp = np.ones([256, 1, 29, 29])
for idx in range(100):
# for i, (inputs, labels) in enumerate(testloader):
    trt_outputs = infer(context, temp, (256, 2), 256)

    print(trt_outputs.shape)
    # print(trt_outputs)
    # print(np.argmax(trt_outputs, axis=0))
    # cnt += 1
    # if cnt == 100:
    #     break
print(time.time() - check_time)


tensorrt_file_name = 'bert_int.plan'
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
 
with open(tensorrt_file_name, 'rb') as f:
    engine_data = f.read()
engine = trt_runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()
check_time = time.time()
cnt = 0
temp = np.ones([256, 1, 29, 29])
for idx in range(100):
# for i, (inputs, labels) in enumerate(testloader):
    trt_outputs = infer(context, temp, (256, 2), 256)

    print(trt_outputs.shape)
    # print(trt_outputs)
    # print(np.argmax(trt_outputs, axis=0))
    # cnt += 1
    # if cnt == 100:
    #     break
print(time.time() - check_time)


test_model = model.Net().cuda()
check_time = time.time()
cnt = 0
temp = torch.randn(256, 1, 29, 29).cuda()
for idx in range(100):
# for i, (inputs, labels) in enumerate(testloader):
    # inputs = inputs.float().cuda()
    normal_outputs = test_model(temp)
    # print(normal_outputs)
    print(normal_outputs.shape)
    cnt += 1
    if cnt == 100:
        break
print(time.time() - check_time)



import tensorrt as trt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda 
import time

model_path = "bert.onnx"
input_size = 32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# def build_engine(model_path):
#     with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser: 
#         builder.max_workspace_size = 1<<20
#         builder.max_batch_size = 1
#         with open(model_path, "rb") as f:
#             parser.parse(f.read())
#         engine = builder.build_cuda_engine(network)
#         return engine

def alloc_buf(engine):
    # host cpu mem
    h_in_size = trt.volume(engine.get_binding_shape(0))
    h_out_size = trt.volume(engine.get_binding_shape(1))
    h_in_dtype = trt.nptype(engine.get_binding_dtype(0))
    h_out_dtype = trt.nptype(engine.get_binding_dtype(1))
    in_cpu = cuda.pagelocked_empty(h_in_size, h_in_dtype)
    out_cpu = cuda.pagelocked_empty(h_out_size, h_out_dtype)
    # allocate gpu mem
    in_gpu = cuda.mem_alloc(in_cpu.nbytes)
    out_gpu = cuda.mem_alloc(out_cpu.nbytes)
    stream = cuda.Stream()
    return in_cpu, out_cpu, in_gpu, out_gpu, stream


def inference(engine, context, inputs, out_cpu, in_gpu, out_gpu, stream):
    # async version
    # with engine.create_execution_context() as context:  # cost time to initialize
    # cuda.memcpy_htod_async(in_gpu, inputs, stream)
    # context.execute_async(1, [int(in_gpu), int(out_gpu)], stream.handle, None)
    # cuda.memcpy_dtoh_async(out_cpu, out_gpu, stream)
    # stream.synchronize()

    # sync version
    cuda.memcpy_htod(in_gpu, inputs)
    context.execute(1, [int(in_gpu), int(out_gpu)])
    cuda.memcpy_dtoh(out_cpu, out_gpu)
    return out_cpu

if __name__ == "__main__":
    inputs = np.random.random((1, 1, 29, 29)).astype(np.float32)

    tensorrt_file_name = '/content/drive/My Drive/capstone1/CAN/bert.plan'
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    
    with open(tensorrt_file_name, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    # engine = build_engine(model_path)
    context = engine.create_execution_context()
    for _ in range(10):
        t1 = time.time()
        in_cpu, out_cpu, in_gpu, out_gpu, stream = alloc_buf(engine)
        res = inference(engine, context, inputs.reshape(-1), out_cpu, in_gpu, out_gpu, stream)
        print(res)
        print("cost time: ", time.time()-t1)