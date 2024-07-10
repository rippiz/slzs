import os
import torch
import librosa
import numpy as np
import pywt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 定义小波去噪函数
def wavelet_denoising(audio_signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(audio_signal, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * 3
    coeffs = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal

# 加载音频数据和标签，并进行小波去噪
def load_audio_data(data_dir):
    data = []
    labels = []

    # 遍历数据目录中的每个类别文件夹
    for class_idx, class_name in enumerate(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)

        # 检查是否是一个目录
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.wav'):  # 检查文件是否是.wav格式的音频文件
                    file_path = os.path.join(class_dir, file_name)
                    try:
                        waveform, sample_rate = librosa.load(file_path, sr=None)  # 加载音频文件

                        # 转换为numpy数组进行小波去噪
                        denoised_signal = wavelet_denoising(waveform)

                        # 提取MFCC特征
                        mfcc = librosa.feature.mfcc(y=denoised_signal, sr=sample_rate, n_mfcc=13)


                        # 取MFCC的平均值并增加一个维度
                        mfcc = torch.tensor(mfcc.mean(axis=1)).unsqueeze(0)


                        # 将处理后的MFCC特征添加到数据列表中
                        data.append(mfcc)

                        # 将类别标签添加到标签列表中
                        labels.append(class_idx)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

    if not data:
        raise RuntimeError("No audio data loaded. Please check the data directory and file formats.")

    # 将数据列表转换为一个张量
    data = torch.stack(data)

    # 将标签列表转换为一个长整型张量
    labels = torch.tensor(labels, dtype=torch.long)

    return data, labels

# 定义CNN模型
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * (13 // 2 // 2 // 2), 128)  # 根据池化后的特征长度进行调整
        self.fc2 = nn.Linear(128, 2)  # 二分类任务
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 数据加载函数
def load_data(data, labels, batch_size=32, val_split=0.2):
    dataset = TensorDataset(data, labels)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Accuracy: {val_correct/val_total:.4f}')

# 主函数
if __name__ == "__main__":
    # 加载数据
    data_dir = r'D:\心音\心音\xin2'  # 替换为实际数据目录
    data, labels = load_audio_data(data_dir)

    # 确保数据形状正确
    data = data.squeeze(2)  # 去掉多余的维度以匹配Conv1d输入
    print(f"Data shape: {data.shape}")  # 调试信息
    print(f"Labels shape: {labels.shape}")  # 调试信息

    # 加载数据
    train_loader, val_loader = load_data(data, labels, batch_size=32, val_split=0.2)

    # 创建模型
    model = AudioCNN()

    # 训练模型
    train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=0.001)

    # 保存模型
    torch.save(model.state_dict(), 'audio_cnn.pth')
