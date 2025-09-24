from ultralytics import YOLO

# 加载你训练好的模型
model = YOLO(r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.pt")

# 尝试导出 TensorRT engine
model.export(format="engine", half=True, device=0)
