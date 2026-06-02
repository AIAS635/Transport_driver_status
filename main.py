import argparse
from config import Config
from src.train import train_model
from src.test import test_model
from src.predict import run_all_task_monitor

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="公交司机状态监测")
    parser.add_argument("--EPOCHS", type=int, default=Config.EPOCHS, help=f"训练轮数 (默认: {Config.EPOCHS})")
    parser.add_argument("--LR", type=float, default=Config.LR, help=f"学习率 (默认: {Config.LR})")
    parser.add_argument("--TASK_TYPE", type=str, default=Config.TASK_TYPE, choices=["binary", "multiclass"], help=f"任务类型 (默认: {Config.TASK_TYPE})")
    parser.add_argument("--BATCH_SIZE", type=int, default=Config.BATCH_SIZE, help=f"批大小 (默认: {Config.BATCH_SIZE})")
    parser.add_argument("--test_video", type=str, default="/share/data/公交检测/videos/000a385e2edd_02_65_6501_3_f0e4b440123b471592064da5c9cae9d8", help="测试视频路径 (默认: data/videos/000a385e2edd_02_65_6501_3_f0e4b440123b471592064da5c9cae9d8)")
    parser.add_argument("RUN_MODE", type=str, default="train", choices=["train", "test", "predict"], help="运行模式: 'train' 训练, 'test' 评估, 'predict' 综合预测")
    args = parser.parse_args()

    # 用命令行参数覆盖 Config
    Config.EPOCHS = args.EPOCHS
    Config.LR = args.LR
    Config.TASK_TYPE = args.TASK_TYPE
    Config.BATCH_SIZE = args.BATCH_SIZE
    Config.RUN_MODE = args.RUN_MODE

    print(f"参数设置: EPOCHS={Config.EPOCHS}, LR={Config.LR}, TASK_TYPE={Config.TASK_TYPE}, BATCH_SIZE={Config.BATCH_SIZE}")

    # 模式开关：'train' 表示训练，'test' 表示评估，'predict' 表示多模型融合预测
    run_mode = Config.RUN_MODE


    if run_mode == "train":
        # ======= 训练模式 =======
        train_model()

    elif run_mode == "test":
        # ======= 评估模式 =======
        # 该模式会读取当前任务对应的权重（如 model_smoke.pth）并计算准确率、召回率等指标
        test_model()

    elif run_mode == "predict":
        # ======= 单视频综合预测模式 =======
        # 丢入一个本地视频路径（支持自动补全 .mp4 后缀），4个网络同时工作综合诊断
        run_all_task_monitor(args.test_video)