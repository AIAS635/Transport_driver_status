from config import Config
from src.train import train_model
from src.test import test_model
from src.predict import run_all_task_monitor

if __name__ == "__main__":

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
        test_video = "data/videos/000a385e2edd_02_65_6501_3_f0e4b440123b471592064da5c9cae9d8"
        run_all_task_monitor(test_video)