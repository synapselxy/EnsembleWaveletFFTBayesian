import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import ptwt
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset

import config_bayesian as cfg
from models.ffc import FFC, FFCSE_block
from models.BBB.BBBLinear import BBBLinear
from models.BBB.BBBConv import BBBConv2d



class waveletTransformModule(nn.Module):
    def __init__(self, wavelet='db1', level=1, conv_channels=8):
        super(waveletTransformModule, self).__init__()
        self.wavelet = wavelet
        self.level = level

        # 低频分支：卷积神经网络，用于处理近似系数
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 通道数从 1 增加到 16
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),  # 通道数从 16 减少到 8
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 4, kernel_size=1)  # 使用 1x1 卷积将通道数进一步减少到 4
        )

        # # 定义卷积和池化层用于高频特征提取
        # self.high_freq_conv = nn.Sequential(
        #     nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

    def forward(self, x):
        # 使用 ptwt.conv_transform_2.wavedec2 进行小波变换
        # batch_features = []
        batch_low_freq_features = []
        batch_high_freq_features = []
        for img in x:  # 处理每个图像
            # features = []
            low_freq_features = []
            high_freq_features = []
            for channel in range(img.shape[0]):  # 对每个通道应用小波变换
                coeffs = ptwt.conv_transform_2.wavedec2(
                    img[channel].unsqueeze(0),  # 将每个通道作为单独的输入
                    wavelet=self.wavelet,
                    level=self.level
                )

                # 近似系数（低频）用于卷积操作
                approx_coeffs = coeffs[0].unsqueeze(0)  # 获取近似系数并增加 batch 维度 torch.Size([1, 1, 16, 16])
                #print("approx_coeffs:", approx_coeffs.shape) #
                approx_features = self.low_freq_conv(approx_coeffs)  # 应用卷积层 torch.Size([1, 32, 8, 8])
                # print("approx_features:", approx_features.shape)
                low_freq_features.append(approx_features.flatten(start_dim=1))

                # 细节系数（高频）用于 FFT 操作
                for detail_coeffs in coeffs[1:]:
                    cH, cV, _ = detail_coeffs  # 水平系数 cH，垂直系数 cV，忽略对角系数 cD

                    # 展平并拼接水平和垂直系数
                    detail_flat = torch.cat([cH.flatten(), cV.flatten()]).unsqueeze(0)
                    high_freq_features.append(detail_flat)


            # 拼接各通道的低频和高频特征
            low_freq_concat = torch.cat(low_freq_features, dim=1) # torch.Size([1, 6144]) -> [1,192]
            high_freq_concat = torch.cat(high_freq_features, dim=1) # torch.Size([1, 4608]) -> [1,1536]
            batch_low_freq_features.append(low_freq_concat)
            batch_high_freq_features.append(high_freq_concat)

        # 将低频和高频特征转换为张量
        batch_low_freq_features = torch.stack(batch_low_freq_features)
        batch_high_freq_features = torch.stack(batch_high_freq_features)

        return batch_low_freq_features, batch_high_freq_features


class GlobalFFTFeatureExtractor(nn.Module):
    def __init__(self):
        super(GlobalFFTFeatureExtractor, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)

        # 计算 FFT，获取频域表示
        fft_features = torch.fft.fft2(x)  # 2D FFT
        # print("fft_feature",fft_features.shape)
        fft_magnitude = torch.abs(fft_features)  # 取幅值，获得频率信息的强度
        # print("fft_magnitude",fft_magnitude.shape)
        # 将频域特征通过全局池化生成全局特征
        global_features = self.global_pool(fft_magnitude)
        return global_features.view(x.size(0), -1)  # 展平为 (batch_size, feature_dim)

class BayesianClassifier(nn.Module):
    def __init__(self, global_input_dim, high_input_dim, num_classes):
        super(BayesianClassifier, self).__init__()
        self.fc1=BBBLinear(global_input_dim, 128)
        self.global_fc = nn.Sequential(
            # BBBLinear(global_input_dim, 128),  # 假设全局特征降维到 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc2=BBBLinear(high_input_dim, 128)
        self.high_freq_fc = nn.Sequential(
            # BBBLinear(high_input_dim, 128),  # 假设高频特征降维到 128
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 将两个特征拼接后的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(64 + 64, 128),  # 将 global_fc 和 high_freq_fc 拼接后的输入维度
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self,  global_feature, high_freq_feature):
        global_out = self.fc1(global_feature)
        global_out = self.global_fc(global_out)
        high_freq_out = self.fc2(high_freq_feature)
        high_freq_out = self.high_freq_fc(high_freq_out)
        # 拼接两个特征
        combined_feature = torch.cat((global_out, high_freq_out), dim=1)

        # 分类
        output = self.classifier(combined_feature)
        return output

    def kl_loss(self):
        return 0.1*(self.fc1.kl_loss() + self.fc2.kl_loss())


class WaveletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset  # 原始 CIFAR-10 数据集

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 获取原始图像和标签
        img, label = self.dataset[idx]

        # 将图像转换为适合 wavelet_module 和 fourier_module 的输入形状
        img = img.unsqueeze(0)  # (3, 32, 32) -> (1, 3, 32, 32)

        # 提取低频和高频特征
        low_freq_features, high_freq_features = wavelet_module(img)
        # print("low_freq_features", low_freq_features.shape)
        # print("high_freq_features", high_freq_features.shape)
        # 提取 Fourier 特征
        fourier_features = fourier_module(img)
        # print("fourier_features", fourier_features.shape)
        fourier_features= fourier_features.unsqueeze(1)
        # 展平和拼接所有特征
        # combined_features = torch.cat([
        #     low_freq_features.view(-1),
        #     high_freq_features.view(-1),
        #     fourier_features.reshape(-1)
        # ], dim=-1)
        # print("combined_features:", combined_features.shape)
        combined_global_feature = torch.cat([low_freq_features.view(-1), fourier_features.view(-1)], dim=0).detach()
        # print("combined_global_feature", combined_global_feature.shape)
        high_freq_features = high_freq_features.view(-1).detach()
        return combined_global_feature, high_freq_features, label


# 数据增强和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

wavelet_module = waveletTransformModule()
fourier_module = GlobalFFTFeatureExtractor()
# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = WaveletDataset(trainset)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)



testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)



# 超参数设置
in_channels = 3        # CIFAR-10 的 RGB 图像
num_classes = 10       # CIFAR-10 类别数量
priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),
    'posterior_rho_initial': (-3, 0.1),
}


# 确定输入特征维度
# print(trainset[0][0].shape)
example_global_feature, example_high_freq_feature, _ = trainset[0]
global_feature_dim = example_global_feature.shape[0]
high_freq_feature_dim = example_high_freq_feature.shape[0]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU!")
# model = BayesianClassifier(input_dim=3072, num_classes=10).to(device)  # 假设小波系数长度为 3072
model = BayesianClassifier(global_feature_dim, high_freq_feature_dim, num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 示例训练过程
for epoch in range(cfg.n_epochs):  # 假设训练 5 个 epoch
    # running_loss = 0.0
    model.train()
    correct = 0
    total = 0
    for global_feature, high_freq_feature, labels in trainloader:
        # 前向传播
        global_feature = global_feature.to(device)
        high_freq_feature = high_freq_feature.to(device)
        outputs = model(global_feature, high_freq_feature).to(device)
        labels = labels.to(device)
        # loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels) + model.kl_loss()
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # running_loss += loss.item()
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{cfg.n_epochs}], Accuracy: {accuracy:.2f}%")

    # print(f"Epoch [{epoch + 1}/5], Loss: {running_loss / len(trainloader)}")
