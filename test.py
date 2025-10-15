# from ultralytics import YOLO
# import cv2
#
# def main():
#     # 1. 加载你训练好的模型（改成你自己训练的权重路径）
#     model = YOLO(r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.pt")
#
#     # 2. 输入视频路径
#     video_path = r"E:\BaiduNetdiskDownload\excavator\挖掘机_258.mp4"
#
#     # 3. 打开视频
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"无法打开视频: {video_path}")
#         return
#
#     # 4. 获取视频帧率和尺寸，用于保存结果视频
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(
#         "excavator_detected.mp4",
#         cv2.VideoWriter_fourcc(*"mp4v"),
#         fps,
#         (width, height)
#     )
#
#     # 5. 循环读取视频帧并推理
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 推理
#         results = model(frame, conf=0.5)  # conf=0.5 表示置信度阈值
#
#         # 将检测结果画到帧图像上
#         annotated_frame = results[0].plot()
#
#         # 写入输出视频
#         out.write(annotated_frame)
#
#         # 同时显示（可选）
#         cv2.imshow("YOLOv11 Excavator Detection", annotated_frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
#     # 6. 释放资源
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print("检测完成，结果已保存到 excavator_detected.mp4")
#
# if __name__ == "__main__":
#     main()
# -*- coding: utf-8 -*-
from ultralytics import YOLO
import cv2
import csv
import os
from collections import defaultdict, Counter

def format_ts(seconds: float) -> str:
    """把秒转成 mm:ss.mmm 的字符串"""
    ms = int((seconds - int(seconds)) * 1000)
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}.{ms:03d}"

# 自定义导出目录（建议提前创建）
SAVE_DIR = r"E:\myRsearch\ultralytics\results"

# 自动拼接输出路径
frames_csv = os.path.join(SAVE_DIR, "frame_details.csv")
intervals_csv = os.path.join(SAVE_DIR, "interval_stats.csv")
VIDEO_OUT = os.path.join(SAVE_DIR, "excavator_detected.mp4")

# 若目录不存在则创建
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # ========= 1) 基本配置 =========
    WEIGHTS = r"E:\myRsearch\ultralytics\runs\detect\train4\weights\best.pt"
    VIDEO_IN = r"E:\project\testvideos\ecavaor2.mp4"

    CONF_THRES = 0.5
    IOU_THRES = 0.45
    INTERVAL_SEC = 1.0          # 时间区间聚合的粒度（秒）

    # ========= 2) 加载模型 =========
    model = YOLO(WEIGHTS)

    # ========= 3) 打开视频、准备导出 =========
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"无法打开视频: {VIDEO_IN}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # 防御：有些视频元数据异常时，强制设个默认 fps
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )



    # 帧级明细 CSV 头
    frame_fields = [
        "frame_idx", "timestamp_s", "timestamp_str",
        "cls_id", "cls_name", "conf",
        "x1", "y1", "x2", "y2"
    ]
    # 写入头
    with open(frames_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=frame_fields)
        writer.writeheader()

    # 区间聚合：key = 区间起点秒(浮点或四舍五入)，value 里存总数与类别计数
    # 例如：interval_bins[10.0] 代表 [10.0s, 11.0s) 这 1 秒内的统计
    interval_bins_total = defaultdict(int)      # 每区间总目标数
    interval_bins_by_cls = defaultdict(Counter) # 每区间按类别计数

    names = model.model.names if hasattr(model, "model") else {}

    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 当前帧时间戳（秒）
        t_sec = frame_idx / fps
        t_str = format_ts(t_sec)

        # ========= 5) 推理 =========
        results = model(
            frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )

        r = results[0]
        boxes = getattr(r, "boxes", None)

        # 收集帧级信息
        per_frame_count = 0
        per_frame_class_names = []

        if boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []

            # 写入帧级 CSV
            with open(frames_csv, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.DictWriter(fcsv, fieldnames=frame_fields)

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = int(clss[i]) if len(clss) > i else -1
                    conf = float(confs[i]) if len(confs) > i else 0.0
                    cls_name = names.get(cls_id, str(cls_id))

                    writer.writerow({
                        "frame_idx": frame_idx,
                        "timestamp_s": f"{t_sec:.3f}",
                        "timestamp_str": t_str,
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": f"{conf:.3f}",
                        "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    })

                    per_frame_count += 1
                    per_frame_class_names.append(cls_name)

                    # 在图像上画框和标签
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), max(0, int(y1)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ========= 6) 区间聚合统计（按 INTERVAL_SEC 分箱）=========
        # 例如 INTERVAL_SEC=1，则 0.0~0.999... 属于 0.0 桶；1.0~1.999... 属于 1.0 桶
        bin_key = (int(t_sec // INTERVAL_SEC)) * INTERVAL_SEC
        interval_bins_total[bin_key] += per_frame_count
        if per_frame_class_names:
            interval_bins_by_cls[bin_key].update(per_frame_class_names)

        # ========= 7) 在视频帧上叠加文本（帧号、时间戳、计数）=========
        hud = f"Frame: {frame_idx} | Time: {t_str} | Count: {per_frame_count}"
        cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 180, 255), 2)

        # ========= 8) 写入输出视频 & 显示 =========
        out.write(frame)
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ========= 9) 导出区间统计 CSV =========
    # 输出列：interval_start_s, interval_end_s, total_count, 以及每个类别的计数列
    # 先收集全量类别，确保列齐全
    all_classes = set()
    for c in interval_bins_by_cls.values():
        all_classes.update(c.keys())
    all_classes = sorted(all_classes)

    interval_fields = ["interval_start_s", "interval_end_s", "interval_start_str", "interval_end_str", "total_count"] + [f"cls_{c}" for c in all_classes]

    with open(intervals_csv, "w", newline="", encoding="utf-8") as icsv:
        writer = csv.DictWriter(icsv, fieldnames=interval_fields)
        writer.writeheader()
        for k in sorted(interval_bins_total.keys()):
            start_s = float(k)
            end_s = start_s + INTERVAL_SEC
            row = {
                "interval_start_s": f"{start_s:.3f}",
                "interval_end_s": f"{end_s:.3f}",
                "interval_start_str": format_ts(start_s),
                "interval_end_str": format_ts(end_s),
                "total_count": interval_bins_total[k]
            }
            cls_counter = interval_bins_by_cls.get(k, Counter())
            for c in all_classes:
                row[f"cls_{c}"] = cls_counter.get(c, 0)
            writer.writerow(row)

    # ========= 10) 释放资源 =========
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"检测完成：\n- 可视化视频：{os.path.abspath(VIDEO_OUT)}\n- 帧级明细：{os.path.abspath(frames_csv)}\n- 区间统计：{os.path.abspath(intervals_csv)}")

if __name__ == "__main__":
    main()
