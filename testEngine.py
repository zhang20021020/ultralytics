from ultralytics import YOLO

def main():
    # 加载 TensorRT 引擎模型
    model = YOLO(r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.engine")

    # 在 test 集目录下跑推理
    results = model.predict(
        source=r"E:\myRsearch\ultralytics\dataset\test\images",  # test集图片文件夹
        imgsz=640,
        conf=0.25,
        save=True,      # 保存预测结果（带框的图片）
        save_txt=True,  # 保存 txt 检测结果
        save_conf=True  # 在 txt 中保存置信度
    )

    print("✅ 使用 TensorRT engine 测试完成")
    print(results)

if __name__ == "__main__":
    main()
