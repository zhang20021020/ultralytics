# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2

from ultralytics import YOLO


def main():
    # 1. 加载你训练好的模型（改成你自己训练的权重路径）
    model = YOLO(r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.pt")

    # 2. 输入视频路径
    video_path = r"E:\BaiduNetdiskDownload\excavator\挖掘机_258.mp4"

    # 3. 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    # 4. 获取视频帧率和尺寸，用于保存结果视频
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("excavator_detected.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # 5. 循环读取视频帧并推理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 推理
        results = model(frame, conf=0.5)  # conf=0.5 表示置信度阈值

        # 将检测结果画到帧图像上
        annotated_frame = results[0].plot()

        # 写入输出视频
        out.write(annotated_frame)

        # 同时显示（可选）
        cv2.imshow("YOLOv11 Excavator Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 6. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("检测完成，结果已保存到 excavator_detected.mp4")


if __name__ == "__main__":
    main()
