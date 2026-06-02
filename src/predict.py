import os
import cv2
import torch
import numpy as np
from config import Config
from src.models import YOLOv8LSTMModel


class DriverMonitorPredictor:
    def __init__(self, task_type=None):
        """
        :param task_type: 可选 'turn', 'phone', 'smoke', 'fatigue'。若不传，则默认读取 Config.TASK_TYPE
        """
        self.device = Config.DEVICE
        self.task_type = task_type if task_type else Config.TASK_TYPE

        self.task_info = Config.TASK_MAP[self.device if not task_type else self.task_type]  # 安全获取
        # 纠正动态指引
        current_map = Config.TASK_MAP[self.task_type]
        self.chinese_name = current_map["excel_name"]
        self.weight_path = current_map["weight_name"]

        # 实例化并载入对应任务的独立权重
        self.model = YOLOv8LSTMModel()
        if os.path.exists(self.weight_path):
            self.model.load_state_dict(torch.load(self.weight_path, map_location=self.device))
            self.model.to(self.device).eval()
        else:
            self.model = None  # 标记模型权重不存在

    def _preprocess_video(self, video_path):
        # 核心改进一：如果未加 .mp4 后缀，自动补齐
        if not video_path.lower().endswith('.mp4'):
            video_path += '.mp4'

        # 核心改进二：强制进行物理文件存在性检查，彻底杜绝全黑虚假矩阵的产生
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"输入视频路径有误，无法找到物理文件: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < Config.TOTAL_FRAMES:
            ret, frame = cap.read()
            if not ret: break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            three_channel_frame = np.stack([gray_frame] * 3, axis=-1)
            three_channel_frame = three_channel_frame / 255.0
            three_channel_frame = np.transpose(three_channel_frame, (2, 0, 1))
            frames.append(three_channel_frame)
        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((3, Config.IMG_HEIGHT, Config.IMG_WIDTH))] * Config.TOTAL_FRAMES
        elif len(frames) < Config.TOTAL_FRAMES:
            padding = [frames[-1]] * (Config.TOTAL_FRAMES - len(frames))
            frames.extend(padding)
        else:
            frames = frames[:Config.TOTAL_FRAMES]
        return torch.tensor(np.array(frames, dtype=np.float32)).unsqueeze(0)

    def predict_status(self, video_path):
        if self.model is None:
            return f"无法预测：未找到【{self.chinese_name}】任务对应的模型权重文件 '{self.weight_path}'。"

        abnormal_threshold = Config.abnormal_threshold
        video_tensor = self._preprocess_video(video_path).to(self.device)
        with torch.no_grad():
            outputs = self.model(video_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            # normal_prob = probabilities[0].item()
            abnormal_prob = probabilities[1].item()

        if abnormal_prob >= abnormal_threshold:
            predicted_class = 1
        else:
            predicted_class = 0
        abnormal_prob = probabilities[1].item() * 100
        status_text = "触发异常" if predicted_class == 1 else "状态安全"
        return f"{self.chinese_name}: {status_text} (置信度: {abnormal_prob:.2f}%)" if predicted_class == 1 else f"{self.chinese_name}: {status_text} (置信度: {(100.0 - abnormal_prob):.2f}%)"


# =====================================================================
# 一键多任务综合扫描器（核心亮点功能）
# =====================================================================
def run_all_task_monitor(video_path):
    print("\n" + "═" * 50)
    print(f"启动全功能车载行为分析模块...")
    print(f"监测目标视频: {os.path.basename(video_path)}")
    print("═" * 50)

    # 循环遍历并载入4个不同网络，并行输出状态
    for task_key in Config.TASK_MAP.keys():
        predictor = DriverMonitorPredictor(task_type=task_key)
        report = predictor.predict_status(video_path)
        print(f" {report}")
    print("═" * 50 + "\n")