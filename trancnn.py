import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# 加载保存的特征和标签
features = np.load('final_features.npy')
labels = np.load('final_labels.npy')

# 数据标准化
scaler = StandardScaler()
num_samples, num_features, num_timepoints = features.shape
features = features.reshape(num_samples, -1)  # 展平特征以进行标准化
features = scaler.fit_transform(features)  # 标准化特征
features = features.reshape(num_samples, num_features, num_timepoints)  # 恢复形状

# 将数据转换为PyTorch张量
features = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据集
dataset = TensorDataset(features, labels)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


# 自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


# Transformer编码层
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


# 定义结合BiLSTM和Transformer的特征提取模型
class BiLSTMTransformerFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, embed_size, num_heads, num_transformer_layers,
                 forward_expansion, dropout, max_length):
        super(BiLSTMTransformerFeatureExtractor, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.bn_lstm = nn.BatchNorm1d(hidden_dim * 2)
        self.pos_encoder = PositionalEncoding(hidden_dim * 2, max_length)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim * 2,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.bilstm(x)
        x = self.bn_lstm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.pos_encoder(x)
        for layer in self.transformer_layers:
            x = layer(x, x, x, None)
        x = x.permute(0, 2, 1)  # 转置为 (batch_size, seq_length, hidden_dim * 2) 以适应 CNN 输入
        x = self.dropout(x)
        return x


# 定义CNN分类器
class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=2, padding=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool1d(2, stride=1)  # 确保池化操作不会使输出大小为零

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.mean(dim=2)  # 全局平均池化
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# 设置环境变量进行调试
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 初始化KFold交叉验证
kf = KFold(n_splits=5)

# 模型参数
input_dim = features.shape[2]
hidden_dim = 256
lstm_layers = 2
embed_size = 256
num_heads = 8
num_transformer_layers = 4
forward_expansion = 4
dropout = 0.5  # 增加Dropout比例
max_length = features.shape[1]  # 假设时间序列长度为96
num_classes = 8  # 假设脉象数据是八分类问题
num_epochs = 30
early_stopping_patience = 5  # 早停法的容忍度

# 进行交叉验证
fold = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_preds = []
all_labels = []
for train_index, test_index in kf.split(dataset):
    print(f"Fold {fold}/{kf.get_n_splits()}")
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化特征提取模型
    feature_extractor = BiLSTMTransformerFeatureExtractor(input_dim, hidden_dim, lstm_layers, embed_size, num_heads,
                                                          num_transformer_layers, forward_expansion, dropout,
                                                          max_length)
    feature_extractor.to(device)

    # 初始化CNN分类器
    classifier = CNNClassifier(hidden_dim * 2, num_classes)
    classifier.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_test_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # 训练特征提取和分类模型
        feature_extractor.eval()
        classifier.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                features = feature_extractor(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix(loss=running_loss / len(train_loader.dataset))

        train_loss = running_loss / len(train_loader.dataset)

        # 测试特征提取和分类模型
        classifier.eval()
        running_loss = 0.0
        all_preds_fold = []
        all_labels_fold = []
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                features = feature_extractor(inputs)
                outputs = classifier(features)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                all_preds_fold.extend(preds.cpu().numpy())
                all_labels_fold.extend(labels.cpu().numpy())
                progress_bar.set_postfix(loss=running_loss / len(test_loader.dataset))

        test_loss = running_loss / len(test_loader.dataset)
        test_accuracy = accuracy_score(all_labels_fold, all_preds_fold)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

        scheduler.step(test_loss)

        # 早停法
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

    all_preds.extend(all_preds_fold)
    all_labels.extend(all_labels_fold)

    # 打印分类报告
    print("Classification Report for Fold:\n", classification_report(all_labels_fold, all_preds_fold))

    fold += 1

# 打印总体分类报告
print("Overall Classification Report:\n", classification_report(all_labels, all_preds))
