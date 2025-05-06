import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from IPython.display import Image
import os

import timm
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR,StepLR

os.environ['CUDA_LAUNCH_BLOCKING']='1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

#查看数据集目录情况,可以通过查看右侧input下的目录结构,弄清楚哪个是训练集哪个是验证集最终找到图片目录
print(os.listdir('../input/training_set'))

# 数据路径
train_dir = '../input/training_set/training_set'
val_dir = '../input/test_set/test_set'

batch_size = 16
num_epochs = 3
num_classes = 2
lr = 3e-5

# 数据预处理（包括数据增强）思考一下训练和验证的数据预处理有什么不同？
# ==== 数据预处理 ====
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.3)),  # 模拟远拍/近拍，原图可放缩为 70%-130%
    #transforms.RandomResizedCrop(384, scale=(0.7, 1.3)),  # 模拟远拍/近拍，原图可放缩为 70%-130%
    transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),  # 额外放缩一点
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 光照增强
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # 模糊模拟
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    #transforms.Resize((384, 384)), #for cait
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 加载数据集
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# ==== 训练 & 验证函数 ====
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return val_loss / len(loader), acc

# ==== 模型 ====
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=2)
model.to(device)

# ==== 损失函数 ====
criterion = nn.CrossEntropyLoss()

# ==== 优化器（SGD + momentum）====
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01) #cait
# ==== 学习率调度器（Cosine Annealing）====
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

# ==== 训练循环 ====
file_header='ViT'
best_acc = 0.0
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
save_path = 'checkpoints/'+file_header+'_best.pth'

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    val_loss, val_acc = evaluate(model, val_loader)

    scheduler.step()
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    better_model_path='checkpoints/'+file_header+str(val_acc*10000)[:4]+'.pth'
    # 保存最好的模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(),better_model_path)
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to {save_path} with acc {val_acc:.4f}")

print("Training completed.")

#通过这条命令看看是否保存好了训练好的模型？
print(os.listdir('checkpoints'))

import random
import matplotlib.pyplot as plt

checkpoint = torch.load('checkpoints/ViT_best.pth')  # 可改为'cuda'看你的环境
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# 随机选择一张图片
image_idx = random.randint(0, len(val_dataset) - 1)  # 随机选择一个索引
image, label = val_dataset[image_idx]

# 显示图像
plt.imshow(image.permute(1, 2, 0))  # 将tensor维度转换为(H, W, C)
plt.axis('off')  # 关闭坐标轴
plt.show()

# 扩展图像的批次维度并输入到模型中
image = image.unsqueeze(0)  # 将图像从 [C, H, W] 转换为 [1, C, H, W]
image = image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))  # 如果有GPU可用，使用GPU

# 获取模型输出
with torch.no_grad():  # 在推理时禁用梯度计算
    output = model(image)

# 输出分类结果
_, predicted = torch.max(output, 1)  # 获取预测类别
class_names = val_dataset.classes  # 获取类别名称（猫/狗）
predicted_class = class_names[predicted.item()]  # 获取预测的类别名称

print(f"Predicted class: {predicted_class}")