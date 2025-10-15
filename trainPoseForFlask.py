import json
from ultralytics import YOLO
import argparse
import torch

# 避免 cudnn 出现 NOT_SUPPORTED 问题
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = YOLO(cfg["model"])
    print("训练进程已启动，正在加载数据和初始化...")

    results = model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 10),
        imgsz=cfg.get("imgsz", 640),
        batch=cfg.get("batch", 2),
        workers=cfg.get("workers", 0),
        device=cfg.get("device", 0)
    )

    print(f"✅ 训练完成，结果保存于: {results.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
