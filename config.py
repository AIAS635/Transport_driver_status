import os


class Config:
    # 路径配置
    ROOT_DATA_DIR = "/share/data/公交检测"
    VIDEO_DIR = "/videos"
    LABEL_PATH = "/labels.csv"

    # 模式配置
    RUN_MODE = "train"
    abnormal_threshold = 0.3

    # 视频处理配置
    IMG_WIDTH = 640
    IMG_HEIGHT = 360
    VIDEO_FPS = 15
    VIDEO_DURATION = 10
    TOTAL_FRAMES = VIDEO_FPS * VIDEO_DURATION  # 150帧

    # 训练与测试通用配置
    BATCH_SIZE = 6  # 保持低批次确保多进程稳定性
    EPOCHS = 100
    LR = 1e-3
    DEVICE = "cuda" if os.path.exists("/usr/bin/nvidia-smi") else "cpu"
    YOLO_MODEL_NAME = "data/weights/yolov8n-cls.pt"
    POS_WEIGHT = 1.5 #对fn的惩罚性权重

    # 当前激活的任务类型，可选值: 'turn', 'phone', 'smoke', 'fatigue'
    TASK_TYPE = "fatigue"

    TASK_MAP = {
        "turn": {"excel_name": "急转弯报警", "weight_name": "model_turn.pth"},
        "phone": {"excel_name": "打电话", "weight_name": "model_phone.pth"},
        "smoke": {"excel_name": "抽烟", "weight_name": "model_smoke.pth"},
        "fatigue": {"excel_name": "集中精神驾驶提示", "weight_name": "model_fatigue.pth"}
    }

    @classmethod
    def get_task_excel_name(cls):
        return cls.TASK_MAP[cls.TASK_TYPE]["excel_name"]

    @classmethod
    def get_weight_path(cls):
        return cls.ROOT_DATA_DIR + cls.TASK_MAP[cls.TASK_TYPE]["weight_name"]