# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO


# ===============================
# LSTM 时序模型
# ===============================
class ExcavatorLSTM(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: (B, T, 27)
        out = out[:, -1, :]  # 取最后一帧
        out = self.fc(out)
        return out


# ===============================
# 推理函数
# ===============================
def run_inference(yolo_path, video_path, lstm_path=None, save_path="output_lstm.mp4", window_size=30):
    # 加载 YOLO pose 模型
    model = YOLO(yolo_path)

    # 加载 LSTM 模型（如果有）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_model = ExcavatorLSTM().to(device)
    if lstm_path:
        seq_model.load_state_dict(torch.load(lstm_path, map_location=device))
    seq_model.eval()

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    # 缓冲区
    buffer = deque(maxlen=window_size)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推理
        results = model.predict(frame, verbose=False)
        state_label = "Idle"
        if len(results) > 0:
            for r in results:
                if r.keypoints is None:
                    continue
                kpts = r.keypoints.xy.cpu().numpy()[0]  # (9,2)
                confs = r.keypoints.conf.cpu().numpy()[0]  # (9,)
                kpts_full = np.concatenate([kpts, confs.reshape(-1, 1)], axis=1)  # (9,3)

                # 送入缓冲区
                buffer.append(kpts_full.flatten())  # (27,)

                # 满足窗口大小时，送入 LSTM
                if len(buffer) == window_size:
                    x = torch.tensor(np.array(buffer), dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,27)
                    with torch.no_grad():
                        logits = seq_model(x)
                        pred = torch.argmax(logits, dim=1).item()
                        state_label = "Digging" if pred == 1 else "Idle"

                # 绘制关键点
                for x, y, v in kpts_full:
                    if v > 0:
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

                # 绘制状态
                cv2.putText(
                    frame,
                    state_label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if state_label == "Digging" else (0, 255, 0),
                    2,
                )

        out.write(frame)
        cv2.imshow("Excavator LSTM", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference(
        yolo_path=r"E:\myRsearch\ultralytics\runs\pose\train\weights\best.pt",  # 你的YOLO模型
        video_path=r"E:\BaiduNetdiskDownload\excavator\挖掘机_163.mp4",  # 你的测试视频
        lstm_path=None,  # 如果你训练了LSTM, 写模型路径；否则用随机初始化的LSTM（效果不准）
        save_path=r"E:\myRsearch\ultralytics\output_lstm.mp4",
        window_size=30,
    )
