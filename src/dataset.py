import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config import Config


class DriverDataset(Dataset):
    def __init__(self, transform=None):
        self.video_dir = Config.VIDEO_DIR
        self.transform = transform
        self.run_mode = Config.RUN_MODE  # "train" 或 "test"

        # 1. 读取标签文件
        print(f"正在读取标签文件: {Config.LABEL_PATH}")
        if Config.LABEL_PATH.endswith('.csv'):
            df = pd.read_csv(Config.LABEL_PATH, encoding="gbk")
        else:
            df = pd.read_excel(Config.LABEL_PATH)

        # 2. 强力数据清洗
        df['is_true'] = df['is_true'].astype(str).str.strip()
        df['result_name'] = df['result_name'].astype(str).str.strip()
        df['back_video'] = df['back_video'].astype(str).str.strip()
        df['is_process'] = df['is_process'].astype(str).str.strip()
        df = df.dropna(subset=['back_video', 'is_true', 'is_process', 'result_name'])

        # 过滤核心逻辑 A：根据运行模式决定是否启用 is_process 过滤
        if self.run_mode == "train":
            df = df[df['is_process'] == '1']

        # 过滤核心逻辑 B：只筛选出当前 Config.TASK_TYPE 所对应的异常类别数据
        target_anomaly = Config.get_task_excel_name()
        df = df[df['result_name'] == target_anomaly]

        # 确保标签转为整型的 0 或 1
        df['is_true'] = df['is_true'].astype(float).astype(int)

        print(f"[任务: {target_anomaly}] 清洗完成！有效进入数据集的样本共 {len(df)} 行。")
        if len(df) == 0:
            print("警告：当前任务筛选出的有效数据量为 0，请检查表格内容或 is_process 是否全为 1！")

        # 3. 扫描本地十万级视频库并解析文件名规则（哈希表 $O(1)$ 级别极速匹配）
        print("正在扫描本地十万级视频库并解析文件名规则...")
        local_video_dict = {}
        with os.scandir(self.video_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith('.mp4'):
                    file_name = entry.name
                    first_underscore_idx = file_name.find('_')
                    if first_underscore_idx != -1:
                        clean_key = file_name[first_underscore_idx + 1:].strip().lower()
                        local_video_dict[clean_key] = file_name
                    else:
                        local_video_dict[file_name.strip().lower()] = file_name


        # 3. 开始精准匹配（拆分为正负两个独立样本池）
        self.pos_samples = []  # 正样本池 (is_true = 1)
        self.neg_samples = []  # 负样本池 (is_true = 0)
        miss_count = 0

        for _, row in df.iterrows():
            url = str(row['back_video']).strip()
            label = int(row['is_true'])
            url_tail = url.split('/')[-1].strip().lower()

            if url_tail in local_video_dict:
                video_path = os.path.join(self.video_dir, local_video_dict[url_tail])
                is_process = str(row['is_process']).strip()
                sample = (video_path, label, local_video_dict[url_tail], is_process)
                if label == 1:
                    self.pos_samples.append(sample)
                else:
                    self.neg_samples.append(sample)
            else:
                miss_count += 1

        print(f"任务【{target_anomaly}】匹配报告:")
        print(f"正样本(异常状态): {len(self.pos_samples)} 个")
        print(f"负样本(正常状态): {len(self.neg_samples)} 个")
        print(f"本地缺失视频: {miss_count} 个\n")

    # 注意：因为我们要动态组合，传统的 __len__ 和 __getitem__ 留作备用，核心组合逻辑移至 train.py
    def __len__(self):
        return len(self.pos_samples) + len(self.neg_samples)

    def _load_video(self, video_path):
        # ... 保持原有的黑白视频加载/3通道复制/150帧对齐逻辑完全不变 ...
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
        return np.array(frames, dtype=np.float32)

    def __getitem__(self, idx):
        # 兼容传统标准 PyTorch 调用的单项获取逻辑
        all_samples = self.pos_samples + self.neg_samples
        video_path, label, filename, is_process = all_samples[idx]
        return torch.tensor(self._load_video(video_path)), torch.tensor(label, dtype=torch.long), filename, is_process