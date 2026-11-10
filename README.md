深度学习目标检测算法Yolov8训练无人机火灾检测数据集 建立基于深度学习卷积神经网络的无人机火灾检测识别
@[toc]
以下文字及代码仅供参考。

如何训练自己收集的数据集？
假设你有，
无人机视角下火灾检测数据集  12736张。yolo通用的txt格式。你如果
划分为训练集、验证集、测试集。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/797df6960e6544119cd5bbaed709d707.png)
类别：Fire
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7bc9a1f2cdd74e94ae19f95c1488bec2.png)
2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e67bc3e258e84d65bb6b48b97fb846ea.png)
从安装依赖到构建 GUI 应用程序的详细步骤。
。

---

### **1. 安装依赖**

首先需要安装必要的库，包括 PyTorch、YOLOv8（使用 Ultralytics 提供的实现）、OpenCV 和其他工具。

```bash
pip install torch torchvision opencv-python ultralytics
```

如果需要构建 GUI 应用程序，还需要安装 `tkinter` 或 `PyQt` 等库：

```bash
pip install PyQt5
```

---

### **2. 准备数据集**

假设数据集已经按照 YOLO 格式标注，并分为 `train/`, `val/`, `test/` 文件夹。文件结构如下：

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

确保每个图像文件在 `images/` 文件夹中，对应的标注文件（`.txt`）在 `labels/` 文件夹中。

创建一个 `data.yaml` 文件，用于描述数据集路径和类别信息：

```yaml
# data.yaml
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test

nc: 1  # 类别数量（假设火灾为单类别）
names: ['fire']  # 类别名称
```

---

### **3. 配置 YOLOv8**

使用 Ultralytics 的 YOLOv8 实现进行训练。加载预训练模型并配置参数。

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 使用 YOLOv8 的 nano 模型

# 查看模型结构
model.info()

# 配置训练参数
model.train(
    data='data.yaml',        # 数据集配置文件
    epochs=50,               # 训练轮数
    imgsz=640,               # 输入图像大小
    batch=16,                # 批量大小
    device='cuda' if torch.cuda.is_available() else 'cpu',  # 使用 GPU 或 CPU
    workers=4,               # 数据加载线程数
    name='fire_detection'    # 训练结果保存目录
)
```

---

### **4. 评估数据集**

训练完成后，可以使用验证集或测试集评估模型性能。

```python
# 加载训练好的模型
model = YOLO('runs/detect/fire_detection/weights/best.pt')

# 在验证集上评估
metrics = model.val()
print(metrics)

# 在单张图片上测试
results = model('dataset/images/test/image_001.jpg')
results.show()  # 显示预测结果
```

---

### **5. 构建 GUI 应用程序**

使用 PyQt5 构建一个简单的 GUI 应用程序，用于加载图像、运行检测并显示结果。

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO
import cv2

class FireDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = YOLO('runs/detect/fire_detection/weights/best.pt')  # 加载训练好的模型

    def initUI(self):
        self.setWindowTitle("火灾检测应用")
        self.setGeometry(100, 100, 800, 600)

        # 创建布局
        layout = QVBoxLayout()

        # 图像显示区域
        self.image_label = QLabel(self)
        self.image_label.setText("请选择一张图片进行火灾检测")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # 按钮区域
        self.load_button = QPushButton("加载图像", self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.detect_button = QPushButton("检测火灾", self)
        self.detect_button.clicked.connect(self.detect_fire)
        layout.addWidget(self.detect_button)

        # 设置主窗口布局
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(640, 480))

    def detect_fire(self):
        if hasattr(self, 'image_path'):
            # 加载图像并进行预测
            results = self.model(self.image_path)
            img_with_boxes = results[0].plot()  # 绘制检测框

            # 将结果转换为 QImage 并显示
            height, width, channel = img_with_boxes.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_with_boxes.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_img))
        else:
            self.image_label.setText("请先加载一张图片")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FireDetectionApp()
    window.show()
    sys.exit(app.exec_())
```

---

### **6. 运行应用程序**

1. 确保所有依赖已安装。
2. 启动 GUI 应用程序：
   ```bash
   python fire_detection_gui.py
   ```
3. 在界面中加载图像并点击“检测火灾”按钮，查看检测结果。

---

### **总结**

从安装依赖到构建 GUI 应用程序的完整流程。通过 YOLOv8 训练火灾检测模型，并结合 PyQt5 构建了一个简单的 GUI 工具，能够加载图像并实时显示检测结果。

结构参考。
仅供参考

