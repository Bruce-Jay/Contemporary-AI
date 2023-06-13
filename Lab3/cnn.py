# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from LeNet import LeNet
from AlexNet import AlexNet
from VGGNet import VGGNet11, VGGNet19
from ResNet import ResNet
from MobileNet import MobileNet
from GoogLeNet import GoogLeNet
import argparse
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torchinfo import summary

# Define relevant variables for the ML task
batch_size = 64
learning_rate = 0.0001
num_epochs = 20

# Create an argparse.ArgumentParser object
parser = argparse.ArgumentParser(description="图像识别深度学习模型")
# Add parameters
parser.add_argument('-model', type=str, default='LeNet5', choices=['LeNet', 'AlexNet', 'VGGNet11', 'VGGNet19', 'ResNet',
                                                                   'MobileNet', 'GoogLeNet'],
                    help='选取模型 (default: LeNet)')
parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
parser.add_argument('-lr', type=float, default=0.0001, help='学习率')
parser.add_argument('-dropout', type=float, default=0.5, help='dropout value')
# Parse command line arguments
args = parser.parse_args()

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the dataset and preprocessing
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.Compose([
                                               # transforms.Resize((32, 32)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.1307,), std=(0.3081,))]),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.Compose([
                                              # transforms.Resize((32, 32)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.1325,), std=(0.3105,))]),
                                          download=True)

# divide the dataset
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=4)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=4)

if args.model == 'LeNet':
    model = LeNet(dropout=args.dropout).to(device)
elif args.model == 'AlexNet':
    model = AlexNet(dropout=args.dropout).to(device)
elif args.model == 'VGGNet11':
    model = VGGNet11(dropout=args.dropout).to(device)
elif args.model == 'VGGNet19':
    model = VGGNet19(dropout=args.dropout).to(device)
elif args.model == 'MobileNet':
    model = MobileNet().to(device)
elif args.model == 'ResNet':
    model = ResNet().to(device)
elif args.model == 'GoogLeNet':
    model = GoogLeNet(aux_logits=False, init_weights=True).to(device)

# Setting the loss function
cost = nn.CrossEntropyLoss()

# Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

# this is defined to print how many steps are remaining when training
total_step = len(train_loader)

summary(model, (64, 1, 28, 28), device='cuda')

starttime = time.time()

num_epochs = args.num_epochs
# 训练集和验证集的 loss 之差
best_loss = 1
epochs_without_improvement = 0
for epoch in range(num_epochs):
    model.train()
    # 记录每一个 epoch 的 loss 和 acc
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate the accuracy
        output = F.softmax(outputs, dim=1)
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(pred == labels)
        train_loss += loss.item()
        train_acc += acc.item()

    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            # calculate the accuracy
            output = F.softmax(outputs, dim=1)
            pred = torch.argmax(output, dim=1)
            acc = torch.sum(pred == labels)
            val_loss += loss.item()
            val_acc += acc.item()

    # 计算每一个 epoch 的训练损失和精度
    train_loss_epoch = train_loss / train_size
    train_acc_epoch = train_acc / train_size
    # 计算验证损失和精度
    val_loss_epoch = val_loss / val_size
    val_acc_epoch = val_acc / val_size

    # 计算验证集的平均损失看是否实现提前停止
    if val_loss_epoch < best_loss:
        best_loss = round(val_loss_epoch, 4)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # 实现提前停止的轮数
    patience = 3
    if (epochs_without_improvement == patience):
        print('Early stooping at epoch {}...'.format(epoch+1))
        break

    print('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'
          .format(epoch + 1, num_epochs, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch))

endtime = time.time()
print('训练时间 %.2f 秒' % (endtime - starttime))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
