import cv2
import numpy as np
from ultralytics import YOLO

# ===============================
# 几何规则：判断是否挖掘
# ===============================
import numpy as np

# 保存上一帧关键点，全局变量
_prev_kpts = None

def is_digging(kpts, thr_boom=40, thr_bucket=-5, thr_height=0.02, motion_thr=2):
    """
    升级版挖掘状态判断
    kpts: (9,3) 当前帧关键点 (x,y,v)
    thr_boom: 动臂角度阈值
    thr_bucket: 铲斗角度阈值
    thr_height: 铲斗尖端与底盘的相对高度差阈值
    motion_thr: 铲斗尖端帧间y方向移动阈值(像素)
    """

    global _prev_kpts
    digging = False

    def calc_angle(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return np.degrees(np.arctan2(dy, dx))

    # 关键点索引（需和你的标注对应）
    tip = kpts[0][:2]       # 铲斗尖端
    hinge = kpts[1][:2]     # 铲斗铰点
    boom_mid = kpts[2][:2]  # 动臂中段
    boom_root = kpts[3][:2] # 动臂根部
    chassis = kpts[5][:2]   # 底盘中心

    # 几何特征
    boom_angle = calc_angle(boom_root, boom_mid)     # 动臂角度
    bucket_angle = calc_angle(hinge, tip)            # 铲斗角度
    rel_h = (chassis[1] - tip[1]) / max(1, chassis[1])  # 相对高度差

    # 单帧规则：动臂低 或 铲斗朝下，并且尖端接近地面
    if (boom_angle < thr_boom or bucket_angle < thr_bucket) and rel_h > thr_height:
        digging = True

    # 连续帧趋势：铲斗尖端是否明显下移
    if _prev_kpts is not None:
        prev_tip = _prev_kpts[0][:2]
        delta_y = tip[1] - prev_tip[1]  # y增大表示向下
        if delta_y > motion_thr:
            digging = True

    # 更新上一帧
    _prev_kpts = kpts.copy()

    return digging

# ===============================
# 可视化绘制
# ===============================
def draw_results(frame, boxes, keypoints, digging_state):
    h, w, _ = frame.shape  # 获取画面大小，用于边界判断

    # 绘制边界框和文字
    for box, kpts in zip(boxes, keypoints):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制关键点
        for (x, y, v) in kpts:
            if v > 0:  # 可见
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        # 固定文字：挖掘机
        text_y1 = y1 - 30
        if text_y1 < 20:  # 如果超出画面上边界，就移到框内
            text_y1 = y1 + 30
        cv2.putText(frame, "excavator", (x1, text_y1),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 状态文字
        if digging_state:
            label = "working"
            color = (0, 0, 255)   # 红色
        else:
            label = "unworking"
            color = (0, 255, 0)   # 绿色

        text_y2 = y1 - 10
        if text_y2 < 20:  # 如果超出画面上边界，就移到框内
            text_y2 = y1 + 60
        cv2.putText(frame, label, (x1, text_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame




# ===============================
# 视频/实时检测
# ===============================
def process_video(model_path, video_path=0, save_path="output.avi", realtime=False):
    # 加载模型
    model = YOLO(model_path)

    # 打开视频或摄像头
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    # 保存结果（如果是文件模式）
    out = None
    if not realtime:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4编码
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        results = model.predict(frame, verbose=False)
        if len(results) == 0:
            if realtime:
                cv2.imshow("Excavator", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # (N,4)
            kpts = r.keypoints.xy.cpu().numpy()  # (N,K,2)
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

            all_keypoints = []
            for i, kp in enumerate(kpts):
                v = confs[i] if confs is not None else np.ones(len(kp))
                kp_full = np.concatenate([kp, v.reshape(-1,1)], axis=1)  # (K,3)
                all_keypoints.append(kp_full)

            # 只取第一台挖掘机
            if len(all_keypoints) > 0:
                digging = is_digging(all_keypoints[0])
                frame = draw_results(frame, boxes, all_keypoints, digging)

        if realtime:
            cv2.imshow("Excavator", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            out.write(frame)

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    process_video(
        r"E:\myRsearch\ultralytics\runs\pose\train\weights\best.pt",  # 模型路径
        video_path=r"E:\myRsearch\database\ecavator.mp4",  # 输入视频
        save_path=r"E:\myRsearch\ultralytics\output.mp4",  # 输出视频（MP4）
        realtime=False
    )

    # 模式2: 实时摄像头
    # process_video("best.pt", video_path=0, realtime=True)
