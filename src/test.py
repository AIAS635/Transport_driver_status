import os
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import DriverDataset
from src.models import YOLOv8LSTMModel
from config import Config


def test_model():
    task_chinese_name = Config.get_task_excel_name()
    weight_path = Config.get_weight_path()

    print("=" * 50)
    print(f"开始执行模型性能评估: 【{task_chinese_name}】")
    print("=" * 50)

    # 1. 初始化标准数据集
    dataset = DriverDataset()
    if len(dataset) == 0:
        print("错误: 没有有效的数据进行测试评估。")
        return

    # 为了保证读取绝对稳定，测试时将 num_workers 设为 0
    test_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # 2. 加载模型与权重
    model = YOLOv8LSTMModel().to(Config.DEVICE)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=Config.DEVICE))
        print(f"成功加载目标权重文件: {weight_path}")
    else:
        print(f"错误: 未找到模型权重文件 '{weight_path}'，请先运行训练。")
        return

    model.eval()

    # 3. 统计变量初始化（构建混淆矩阵）
    tp = 0  # 真正例：本来是异常，预测也是异常
    fp = 0  # 假正例：本来是正常，预测成了异常
    fn = 0  # 假负例：本来是异常，预测成了正常
    tn = 0  # 真负例：本来是正常，预测也是正常

    print("正在批量读取视频并进行网络推理...")

    csv_path = os.path.join(f"./predict_{Config.TASK_TYPE}.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow(["video_name", "predicted", "confidence", "is_process", "is_true"])

    with torch.no_grad():
        for idx, (videos, labels, filenames, is_process_list) in tqdm(enumerate(test_loader), total=len(test_loader), desc="测试中："):
            videos = videos.to(Config.DEVICE)
            outputs = model(videos)
            probabilities = torch.softmax(outputs, dim=1)
            abnormal_probs = probabilities[:, 1]
            predicted = (abnormal_probs >= Config.abnormal_threshold).long()

            # 转换成 CPU 数组计算指标
            labels_np = labels.cpu().numpy()
            predicted_np = predicted.cpu().numpy()
            abnormal_np = abnormal_probs.cpu().numpy()

            for t, p, ab_prob, fname, isp in zip(labels_np, predicted_np, abnormal_np, filenames, is_process_list):
                conf = ab_prob if p == 1 else (1 - ab_prob)
                writer.writerow([fname, int(p), f"{conf:.4f}", isp, int(t)])
                csv_file.flush()
                if t == 1 and p == 1:
                    tp += 1
                elif t == 0 and p == 1:
                    fp += 1
                elif t == 1 and p == 0:
                    fn += 1
                elif t == 0 and p == 0:
                    tn += 1

            # if (idx + 1) % 5 == 0:
            #     print(f"评估进度: 批次 [{idx + 1}/{len(test_loader)}]， tp = {tp}， fp = {fp}， fn = {fn}， tn = {tn}")

    csv_file.close()
    print(f"预测结果已保存至: {csv_path}")

    # 4. 计算指标
    total_samples = tp + fp + fn + tn
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 50)
    print(f"【{task_chinese_name}】最终量化评估报告")
    print("=" * 50)
    print(f" 测试样本总数: {total_samples} 个")
    print(f" 混淆矩阵明细: TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    print("-" * 50)
    print(f" 准确率 (Accuracy):  {accuracy * 100:.2f}% (模型判对的总比例)")
    print(f" 精确率 (Precision): {precision * 100:.2f}% (报出的异常里真正对的比例)")
    print(f" 召回率 (Recall):    {recall * 100:.2f}% (实际异常被成功抓到的比例)")
    print(f" F1综合得分 (F1):    {f1_score * 100:.2f}%")
    print("=" * 50 + "\n")