from ultralytics import YOLO

def main():
    # 加载训练好的模型
    model = YOLO(r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.pt")

    # 调用 val() 在 test 数据集上跑
    results = model.val(
        data=r"E:\myRsearch\ultralytics\dataset\data.yaml",  # 你的 data.yaml 文件
        split="test",        # 指定用 test 集合
        imgsz=640,           # 输入图像尺寸
        conf=0.25,           # 置信度阈值
        batch=4,
        save_json=True,      # 保存 COCO 格式结果 (可选)
        save_txt=True,       # 保存 txt 检测结果 (可选)
        save_hybrid=False    # 不保存混合标签
    )

    print("✅ 测试集检测完成")
    print(results)

if __name__ == "__main__":
    main()
