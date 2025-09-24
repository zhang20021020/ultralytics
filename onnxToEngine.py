import tensorrt as trt

onnx_file = r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.onnx"
engine_file = r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)

parser = trt.OnnxParser(network, logger)
with open(onnx_file, "rb") as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB

# 启用 FP16 精度
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

engine = builder.build_engine(network, config)
with open(engine_file, "wb") as f:
    f.write(engine.serialize())

print("转换完成，保存为:", engine_file)
