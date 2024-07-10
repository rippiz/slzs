import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# 自定义数据集类，用于加载音频数据
class LungSoundDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=22050 * 5):
        """
        初始化数据集
        :param root_dir: 数据集根目录
        :param transform: 数据预处理方法
        :param max_len: 音频文件的最大长度（默认为5秒）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.max_len = max_len
        self.classes = os.listdir(root_dir)
        self.file_paths = []
        self.labels = []

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.wav'):
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sample_rate = librosa.load(file_path, sr=None)

        # 对音频数据进行裁剪或填充，使其长度一致
        if len(waveform) > self.max_len:
            waveform = waveform[:self.max_len]
        else:
            waveform = np.pad(waveform, (0, self.max_len - len(waveform)), mode='constant')

        if self.transform:
            waveform = self.transform(waveform)
        label = self.labels[idx]
        return waveform, label


# 数据预处理类，将音频数据转换为Mel频谱图
class MelSpectrogramTransform:
    def __init__(self, sample_rate=22050, n_mels=128):
        """
        初始化Mel频谱图转换
        :param sample_rate: 采样率
        :param n_mels: Mel频率带的数量
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    def __call__(self, waveform):
        # 生成Mel频谱图
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=self.sample_rate, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # 增加通道维度
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        return torch.tensor(mel_spec_db, dtype=torch.float32)


# 数据集路径
root_dir = r'D:\诊断模型\肺音\肺音\ffei'
transform = MelSpectrogramTransform()

# 创建数据集和数据加载器
dataset = LungSoundDataset(root_dir=root_dir, transform=transform)
train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


# Transformer特征提取器
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers):
        """
        初始化Transformer特征提取器
        :param input_dim: 输入特征维度
        :param d_model: Transformer模型的特征维度
        :param nhead: 多头注意力机制的头数
        :param num_encoder_layers: 编码层数
        """
        super(TransformerFeatureExtractor, self).__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)  # 设置 batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, x):
        x = self.linear_in(x)
        x += self.pos_encoder[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        return x


# CNN分类器
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        """
        初始化CNN分类器
        :param num_classes: 分类数量
        """
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.7)  # 新增Dropout层

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 调整全连接层的输入大小
        self.fc1 = nn.Linear(64 * 128 * 54, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 调整形状
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 将Transformer和CNN结合起来
class CombinedModel(nn.Module):
    def __init__(self, transformer, cnn):
        """
        初始化组合模型
        :param transformer: Transformer特征提取器
        :param cnn: CNN分类器
        """
        super(CombinedModel, self).__init__()
        self.transformer = transformer
        self.cnn = cnn

    def forward(self, x):
        x = x.squeeze(1)  # 移除通道维度 (batch, 1, feature_dim, seq_len) -> (batch, feature_dim, seq_len)
        x = x.permute(0, 2, 1)  # 变换维度以适应Transformer输入 (batch, seq_len, feature_dim)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)  # 变换回原始维度 (batch, feature_dim, seq_len)
        x = x.unsqueeze(1)  # 增加通道维度 (batch, 1, feature_dim, seq_len)
        x = self.cnn(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
transformer = TransformerFeatureExtractor(input_dim=128, d_model=512, nhead=8, num_encoder_layers=3).to(device)
cnn = CNNClassifier(num_classes=9).to(device)
model = CombinedModel(transformer, cnn).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 40

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for waveforms, labels in train_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

    # 验证循环
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in test_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy after epoch {epoch + 1}: {accuracy:.2f} %')

print('Finished Training')

# 测试循环
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for waveforms, labels in test_loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        outputs = model(waveforms)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test images: {100 * correct / total} %')

# 保存模型
torch.save(model.state_dict(), 'lung_sound_classification_model.pth')

