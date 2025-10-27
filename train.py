# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics import YOLO


def main():
    # Load a pretrained YOLO11n model
    model = YOLO("E:\\myRsearch\\ultralytics\\ultralytics\\cfg\\models\\11\\yolo11.yaml")

    # Train the model on your dataset
    model.train(
        data="E:\\myRsearch\\ultralytics\\dataset\\data.yaml",  # 数据集配置
        epochs=150,  # 训练轮数
        imgsz=640,  # 输入图片大小
        device=0,  # 使用第 1 块 GPU
        batch=8,  # 批次大小
        workers=0,  # ⚡ Windows 下建议设为 0，避免多进程冲突
    )


if __name__ == "__main__":
    main()
