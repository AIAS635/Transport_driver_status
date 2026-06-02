import torch
import torch.nn as nn
from ultralytics import YOLO
from config import Config


class YOLOv8LSTMModel(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=2):
        super(YOLOv8LSTMModel, self).__init__()

        # 1. 加载 YOLOv8 分类模型
        yolo_model = YOLO(Config.YOLO_MODEL_NAME)

        # 提取底层的 PyTorch nn.Module 网络结构，并直接移至 GPU/CPU 设备
        self.yolo_inner = yolo_model.model.to(Config.DEVICE)

        # 冻结 YOLO 全局参数以节省显存
        for param in self.yolo_inner.parameters():
            param.requires_grad = False

        # 根据报错动态修正：YOLOv8n-cls 实际特征维度为 256
        self.feature_dim = 256

        # 2. 时序处理层 (双向 LSTM)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        ).to(Config.DEVICE)  # 确保移至 GPU

        # 3. 最终的司机行为分类输出
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        ).to(Config.DEVICE)  # 确保移至 GPU

        # 将双向 LSTM 的输出映射为一个标量权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 显式确保输入的视频数据在 GPU 上
        x = x.to(Config.DEVICE)

        # 输入维度: [Batch, 150帧, 3通道, 360高, 640宽]
        batch_size, seq_len, c, h, w = x.size()

        # 合并 Batch 和时序维度 -> [Batch * 150, 3, 360, 640]
        x = x.view(batch_size * seq_len, c, h, w)

        with torch.no_grad():
            features = x
            # 动态遍历 YOLOv8 内部模块，在 Classify 模块前截断
            for layer in list(self.yolo_inner.children())[0]:
                if layer.__class__.__name__ == 'Classify':
                    break
                features = layer(features)

            # 全局平均池化，将特征网格压缩
            if features.ndim > 2:
                features = torch.mean(features, dim=[-2, -1])

                # 展平特征
            features = torch.flatten(features, start_dim=1)

        # 动态自适应检查：自动根据 YOLO 导出的实际维度进行调整，防止再次报错
        actual_feature_dim = features.shape[-1]

        # 恢复时序结构 -> [Batch, 150, 实际特征维度]
        features = features.view(batch_size, seq_len, actual_feature_dim)

        # 如果实际维度与初始化不符，动态更新 LSTM 的输入映射（防御性编程）
        if actual_feature_dim != self.feature_dim:
            self.lstm = self.lstm.to(features.device)

        # 送入双向 LSTM -> [Batch, 150, hidden_dim * 2]
        lstm_out, _ = self.lstm(features)

        # 🚀 改造：计算每一帧的注意力得分 -> [Batch, 150, 1]
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)  # 归一化权重

        # 将权重加权到所有帧的特征上 -> [Batch, hidden_dim * 2]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        # 送入全连接层分类
        out = self.fc(context_vector)
        return out