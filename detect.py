import cv2
from ultralytics import YOLO
import yaml
import torch
import gc

# 清理缓存
gc.collect()
torch.cuda.empty_cache()
# -------------------- 1. 数据集训练 --------------------
def train_model():
    # 加载预训练模型（YOLO11n）
    model = YOLO("yolo11n.pt")  # 自动下载基础模型

    # 训练配置
    results = model.train(
        data=r'datasets/my_dataset.yaml',  # 数据集配置文件路径
        epochs=100,  # 训练轮次
        batch=8,  # 批次大小
        imgsz=640,  # 输入图像尺寸
        lr0=0.01,  # 初始学习率
        device="0",  # 使用GPU（"cpu"为CPU）
        name="my_custom_model" , # 保存训练结果的文件夹名称
        amp=True,
        workers=0  # 添加此参数，强制禁用多进程（Windows专用)
    )


# -------------------- 2. 模型验证 --------------------
def validate_model():
    model = YOLO("runs/detect/my_custom_model/weights/best.pt")  # 加载训练好的模型
    metrics = model.val()  # 在验证集上评估
    print(f"mAP@0.5: {metrics.box.map}")  # 输出mAP指标


# -------------------- 3. 图像检测 --------------------
def detect_image(image_path):
    model = YOLO("runs/detect/my_custom_model/weights/best.pt")  # 加载自定义模型
    results = model(image_path)
    results[0].show()  # 显示标注结果
    print("检测到的目标坐标：", results[0].boxes.xyxy.tolist())


# -------------------- 4. 视频检测 --------------------
#def detect_video(video_path):
#    model = YOLO("runs/detect/my_custom_model/weights/best.pt")
#    cap = cv2.VideoCapture(video_path)
#    while cap.isOpened():
#       success, frame = cap.read()
#        if success:
#            results = model.track(frame, persist=True)  # 启用目标追踪
#            annotated_frame = results[0].plot()
#            cv2.imshow("YOLOv11 Detection", annotated_frame)
#           if cv2.waitKey(1) & 0xFF == ord('q'):
#                break
#    cap.release()
#    cv2.destroyAllWindows()


# -------------------- 执行示例 --------------------
if __name__ == "__main__":
    # Step 1: 训练模型（首次运行时取消注释）
    train_model()

    # Step 2: 验证模型精度
    validate_model()

    # Step 3: 测试单张图像
    detect_image(r"E:\learning\deep\yoyo\ultralytics-main\datasets\coco128\images\train2017\000000000165.jpg")

    # Step 4: 测试视频（需准备视频文件）
    # detect_video("test_video.mp4")