# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import json

import torch

from ultralytics import YOLO

# 避免 cudnn 出现 NOT_SUPPORTED 问题
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(config_path):
    # 1️⃣ 读取配置文件
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # 2️⃣ 加载模型
    model = YOLO(cfg["model"])
    print("训练进程已启动，正在加载数据和初始化...")

    # 3️⃣ 开始训练
    results = model.train(
        data=cfg["data"],  # 数据集配置路径
        epochs=cfg.get("epochs", 150),  # 训练轮数（默认150）
        imgsz=cfg.get("imgsz", 640),  # 输入图像尺寸
        batch=cfg.get("batch", 8),  # 批次大小
        workers=cfg.get("workers", 0),  # 数据加载线程数（Windows建议0）
        device=cfg.get("device", 0),  # 设备编号
    )

    print(f"✅ 训练完成，结果保存于: {results.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    main(args.config)
