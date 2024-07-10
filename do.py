import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化 Dlib 的人脸检测器和特征点检测器
detector = dlib.get_frontal_face_detector()

# 指定shape_predictor_68_face_landmarks.dat文件的路径
dat_file_path = r'D:\life\bdwp\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat'

# 获取文件的绝对路径
absolute_path = os.path.abspath(dat_file_path)

# 检查文件是否存在
if not os.path.exists(absolute_path):
    raise FileNotFoundError(f"File not found: {absolute_path}")

print(f"Absolute path: {absolute_path}")

# 加载shape predictor
predictor = dlib.shape_predictor(absolute_path)

def detect_face_and_landmarks(image):
    # 使用 Dlib 检测人脸
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = []
    for face in faces:
        # 检测人脸特征点
        shape = predictor(gray, face)
        landmarks.append(shape)
    return faces, landmarks

def draw_irregular_contour(image, points):
    contour = np.array(points, dtype=np.int32)
    cv2.polylines(image, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

# 加载图像
image_path = r'D:\诊断模型\面诊\66\p\v2-f24b1b1a7d5b88573dc22fb54d26cd25_r.jpg'
print(f"Loading image from: {image_path}")
print(f"Absolute path: {os.path.abspath(image_path)}")
print(f"File exists: {os.path.exists(image_path)}")
print(f"Is file: {os.path.isfile(image_path)}")

if not os.path.exists(image_path):
    raise ValueError(f"Path does not exist: {image_path}")

# 使用 Pillow 验证图像
try:
    img = Image.open(image_path)
    img.verify()  # 验证图像是否完整
    print("Image loaded successfully using Pillow")
except (IOError, SyntaxError) as e:
    raise ValueError(f"Image not found or unable to load using Pillow: {e}")

# 使用 Pillow 重新打开图像并转换为 OpenCV 格式
img = Image.open(image_path)
image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

faces, landmarks = detect_face_and_landmarks(image)

for shape in landmarks:
    # 提取面部特征点
    nose_points = [(shape.part(27).x, shape.part(27).y),
                   (shape.part(28).x, shape.part(28).y),
                   (shape.part(29).x, shape.part(29).y),
                   (shape.part(30).x, shape.part(30).y)]

    mouth_points = [(shape.part(48).x, shape.part(48).y),
                    (shape.part(49).x, shape.part(49).y),
                    (shape.part(50).x, shape.part(50).y),
                    (shape.part(51).x, shape.part(51).y),
                    (shape.part(52).x, shape.part(52).y),
                    (shape.part(53).x, shape.part(53).y),
                    (shape.part(54).x, shape.part(54).y),
                    (shape.part(55).x, shape.part(55).y),
                    (shape.part(56).x, shape.part(56).y),
                    (shape.part(57).x, shape.part(57).y),
                    (shape.part(58).x, shape.part(58).y),
                    (shape.part(59).x, shape.part(59).y)]

    left_cheek_points = [(shape.part(1).x, shape.part(1).y),
                         (shape.part(2).x, shape.part(2).y),
                         (shape.part(3).x, shape.part(3).y),
                         (shape.part(4).x, shape.part(4).y),
                         (shape.part(5).x, shape.part(5).y)]

    right_cheek_points = [(shape.part(12).x, shape.part(12).y),
                          (shape.part(13).x, shape.part(13).y),
                          (shape.part(14).x, shape.part(14).y),
                          (shape.part(15).x, shape.part(15).y),
                          (shape.part(16).x, shape.part(16).y)]

    forehead_points = [(shape.part(19).x, shape.part(19).y),
                       (shape.part(24).x, shape.part(24).y)]

    draw_irregular_contour(image,
 nose_points)
    draw_irregular_contour(image, mouth_points)
    draw_irregular_contour(image, left_cheek_points)
    draw_irregular_contour(image, right_cheek_points)
    draw_irregular_contour(image, forehead_points)

# 显示图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
