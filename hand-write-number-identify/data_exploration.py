# 数据探索
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 数据预处理：只转tensor，不做其他增强
transform = transforms.Compose([transforms.ToTensor()])

# 加载数据集，自动下载
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
print(type(train_data))
# ========== 打印基础信息 ==========
print(f"训练集样本总数：{len(train_data)}")
print(f"测试集样本总数：{len(test_data)}")

# 取出第0号样本
img, label = train_data[0]
print("\n==== 单个样本信息 ====")
print(f"图像张量shape [通道,高,宽]：{img.shape}")
print(f"图片真实标签：{label}")
print(f"像素最小值：{img.min().item()}")
print(f"像素最大值：{img.max().item()}")
print(f"图片像素矩阵：\n{img.squeeze().numpy()}") # squeeze去掉通道维度 [28,28]
print(type(img))# 看看img的类型

# ========== 画图展示这个手写数字 ==========
plt.figure()
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"MNIST sample, label={label}")
plt.axis("off")
plt.show()

# ========== 一次性看前10张图片 ==========
plt.figure(figsize=(10, 2))
for i in range(10):
    img_i, label_i = train_data[i]
    plt.subplot(1, 10, i+1)
    plt.imshow(img_i.squeeze(), cmap="gray")
    plt.title(f"{label_i}")
    plt.axis("off")
plt.tight_layout()
plt.show()
