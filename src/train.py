import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime
from src.dataset import DriverDataset
from src.models import YOLOv8LSTMModel
from config import Config


# 🚀 核心修复：将辅助类移到全局作用域（函数外部），使其可以被 pickle 序列化
class EpochSubDataset(torch.utils.data.Dataset):
    def __init__(self, samples, loader_fn):
        self.samples = samples
        self.loader_fn = loader_fn

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        v_path, label, _, _ = self.samples[idx]
        # 调用传进来的视频加载函数
        return torch.tensor(self.loader_fn(v_path)), torch.tensor(label, dtype=torch.long)


def train_model():
    task_chinese_name = Config.get_task_excel_name()
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    weight_path = Config.get_weight_path()"

    print(f"\n==========================================")
    print(f"启动不平衡数据均衡化训练: 【{task_chinese_name}】")
    print(f"==========================================")

    # 1. 实例化双池数据集
    dataset = DriverDataset()
    pos_pool = dataset.pos_samples
    neg_pool = dataset.neg_samples

    if len(pos_pool) == 0 or len(neg_pool) == 0:
        print("正样本或负样本数量为0，无法执行平衡训练。")
        return

    num_pos = len(pos_pool)
    num_neg = len(neg_pool)

    # 计算需要多少个 Epoch 才能把所有负样本至少“轮审”一遍
    neg_chunks_count = int(torch.ceil(torch.tensor(num_neg / num_pos)).item())
    print(f"️负正比例约为 {num_neg}/{num_pos} = {num_neg / num_pos:.2f}:1")
    print(f"系统已自动调整：每轮将负样本切为 {neg_chunks_count} 块，每块数量与正样本绝对 1:1 平衡。")

    # 2. 初始化模型与环境
    model = YOLOv8LSTMModel().to(Config.DEVICE)
    if os.path.exists(weight_path):
        print(f"检测到历史权重 '{weight_path}'，已安全接入断点...")
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE, weights_only=True))

    loss_weights = torch.tensor([1.0, Config.POS_WEIGHT], dtype=torch.float32).to(Config.DEVICE)    # 正类在loss中权重更高，避免fn出现，提高recall
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # 指针：记录当前轮替到了负样本池的哪个位置
    neg_start_idx = 0
    # 打乱负样本池，确保第一轮切块具有高度随机性
    random.shuffle(neg_pool)

    # 建议总 Epoch 数设置为块数的整数倍，确保所有负样本被利用的次数完全均等
    actual_epochs = max(Config.EPOCHS, neg_chunks_count * 2)
    print(f"为了让负样本尽量全部用到且机会均等，本次训练将执行 {actual_epochs} 个 Epoch。\n")

    for epoch in range(actual_epochs):
        model.train()

        # 🚀 动态提取当前 Epoch 的负样本切片
        neg_end_idx = neg_start_idx + num_pos

        if neg_end_idx <= num_neg:
            # 正常在范围内切块
            current_epoch_negs = neg_pool[neg_start_idx:neg_end_idx]
            neg_start_idx = neg_end_idx
        else:
            # 触及负样本池末尾，把剩余的拿出来，不足的部分从头循环补齐
            rem_count = num_neg - neg_start_idx
            current_epoch_negs = neg_pool[neg_start_idx:]
            # 重新打乱，从头补齐
            random.shuffle(neg_pool)
            needed = num_pos - rem_count
            current_epoch_negs.extend(neg_pool[:needed])
            neg_start_idx = needed
            print(f"[Epoch {epoch + 1}] 负样本库已完整消耗一轮，已触发全库重随机打乱，开启新一轮轮替。")

        # 🚀 组合当前 Epoch 的绝对平衡数据集 (数量恰好为 2 * num_pos)
        epoch_samples = pos_pool + current_epoch_negs
        random.shuffle(epoch_samples)  # 混合正负样本顺序

        # 3. 封装为子级 Dataset (此处移出了外部定义的 EpochSubDataset)
        current_epoch_dataset = EpochSubDataset(epoch_samples, dataset._load_video)
        train_loader = DataLoader(current_epoch_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)

        # 4. 执行标准 Epoch 训练
        running_loss, correct, total = 0.0, 0, 0
        for idx, (videos, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch{epoch + 1}/{actual_epochs}进度:"):
            videos, labels = videos.to(Config.DEVICE), labels.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(
            f"Epoch {epoch + 1}/{actual_epochs} 完成 | 本轮平衡样本数: {len(epoch_samples)} | 损失: {epoch_loss:.4f} | 准确率: {epoch_acc:.2f}%")

        # 实时归档保存
        weight_path = weight_path.split(".")[0] + f"_{datetime_str}_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), weight_path)

    print(f"\n极其不平衡对抗训练完美结束！对应专用权重已妥善保存至: {weight_path}\n")