## PyTorch从精通到入门04：CNN实现图像分类

前一个例子，全部使用了全连接层来做图像识别，这其实不是常用做法。

图像识别的标准做法，是使用不同的卷积核不停的卷图像里面的每个区域，通过不同的卷积核去提取图像的不同特征。最后再把这些特征通过全连接层输出、进行类别判断。

下面的代码，构建了两个卷积层，然后再把卷积层的输出送到全连接层进行分类。

顺便附带了保存和加载CheckPoint的基本实现。



```python

import torch
import torchvision

class ImageClassificationWithCNN(torch.nn.Module):
    def __init__(self, image_width, image_height, num_classifications) -> None:
        super().__init__()
        channel_cnt_1 = 6
        channel_cnt_2 = 16
        fc_features_1 = 128
        cnn_stride = 1
        cnn_kernel_size = 5
        self.pool_size = 2

        # 两次一样的操作，原始尺寸 + 步长 - 卷积核大小，然后除以池化核大小
        height_after_cnn = (image_height + cnn_stride - cnn_kernel_size)/self.pool_size
        height_after_cnn = int((height_after_cnn + cnn_stride - cnn_kernel_size)/self.pool_size)
        width_after_cnn = (image_width + cnn_stride - cnn_kernel_size)/self.pool_size
        width_after_cnn = int((width_after_cnn + cnn_stride - cnn_kernel_size)/self.pool_size)

        # 全连接层时，数据展平后的大小
        self.size_after_cnn = channel_cnt_2 * height_after_cnn * width_after_cnn

        # 28 + 步长(1) - kernel(5) = 24，池化后
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=channel_cnt_1, kernel_size=cnn_kernel_size)
        self.conv_2 = torch.nn.Conv2d(in_channels=channel_cnt_1, out_channels=channel_cnt_2, 
            kernel_size=cnn_kernel_size)
        # Channel * Height * Width
        self.fc_1 = torch.nn.Linear(in_features=self.size_after_cnn, out_features=fc_features_1)
        self.fc_2 = torch.nn.Linear(in_features=fc_features_1, out_features=num_classifications)

    def forward(self, x):
        # 2 * 2最大化池化
        y = torch.max_pool2d(torch.relu(self.conv_1(x)), self.pool_size)
        y = torch.max_pool2d(torch.relu(self.conv_2(y)), self.pool_size)
        # 将数据展平，从[batch_size, channel, height, width] -> [batch_size, channel * height * width]
        y = y.view(-1, self.size_after_cnn)
        y = torch.relu(self.fc_1(y))
        y = self.fc_2(y)
        return y


def load_train_data(batch_size=32):
    # 下载MNIST训练数据集
    # 然后转为Tensor、标准化操作(均值mean为0.1，方差std为0.5)
    train_dataset = torchvision.datasets.MNIST('./Data/', train=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1,), (0.5,))
        ]))
    
    # 用Dataloader加载数据
    # 接收来自用户的Dataset实例，并使用采样器策略将数据采样为小批次。
    # 每批次32个、随机打乱、并发线程为2、使用pin_memory
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=2, **{'pin_memory': True})
    
    return train_dataloader


def load_test_data(batch_size=32):
    # 下载MNIST测试数据集
    # 然后转为Tensor、标准化操作(均值mean为0.1，方差std为0.5)
    test_dataset = torchvision.datasets.MNIST('./Data/', train=False, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1,), (0.5,))
        ]))
    
    # 用Dataloader加载数据
    # 接收来自用户的Dataset实例，并使用采样器策略将数据采样为小批次。
    # 每批次32个、并发线程为2、使用pin_memory
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=2, **{'pin_memory': True})
    
    return test_dataloader


def train(model, device, train_dataloader, loss_func, optimizer):
    # 训练模式
    model.train()
    # 所有批次累计损失和
    epoch_train_loss = 0
    # 累计预测正确的样本总数
    epoch_train_correct = 0

    # 循环一次数据的多个批次
    for x, y in train_dataloader:
        # non_blocking=True异步传输数据
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        predicted = model(x)
        loss = loss_func(predicted, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加
        with torch.no_grad():
            epoch_train_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_train_loss += loss.item()

    return (epoch_train_loss, epoch_train_correct)


def test(model, device, test_dataloader, loss_func):
    # 测试模式
    model.eval()
    # 所有批次累计损失和
    epoch_test_loss = 0
    # 累计预测正确的样本总数
    epoch_test_correct = 0

    # 循环一次数据的多个批次
    # 测试模式，不需要梯度计算、反向传播
    with torch.no_grad():
        for x, y in test_dataloader:
            # non_blocking=True异步传输数据
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            predicted = model(x)
            loss = loss_func(predicted, y)

            # 累加
            epoch_test_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_test_loss += loss.item()

    return (epoch_test_loss, epoch_test_correct)


def fit(epoch=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImageClassificationWithCNN(28, 28, 10)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练数据
    train_data = load_train_data()
    # 数据集总量
    total_train_data_cnt = len(train_data.dataset)
    # 数据集批次数目
    num_train_batch = len(train_data)

    # 测试数据
    test_data = load_test_data()
    # 数据集总量
    total_test_data_cnt = len(test_data.dataset)
    # 数据集批次数目
    num_test_batch = len(test_data)

    # 循环全部数据
    for i in range(epoch):
        # 训练模型
        epoch_train_loss, epoch_train_correct = train(model, device, train_data, loss_func, optimizer)
        # 所有批次的统计和/批次数量 = 平均损失率
        avg_train_loss = epoch_train_loss/num_train_batch
        # 预测正确的样本数/总样本数 = 平均正确率
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt

        # 测试模型
        epoch_test_loss, epoch_test_correct = test(model, device, test_data, loss_func)
        # 所有批次的统计和/批次数量 = 平均损失率
        avg_test_loss = epoch_test_loss/num_test_batch
        # 预测争取的样本数/总样本数 = 平均正确率
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

        # CheckPoint用法
        if (i + 1)%5 == 0:
            # 每5个epoch保存模型信息
            ckpt_path = f'./ImageClassificationWithCNN_{i+1}.ckpt'
            save_checkpoint(model, optimizer, i, ckpt_path)


def save_checkpoint(model, optimizer, epoch, file_path):
    # 构造CheckPoint内容
    ckpt = {
        'model': model.state_dict(), 
        'optimizier': optimizer.state_dict(),
        'epoch': epoch, 
        #'lr_schedule': lr_schedule.state_dict()
    }
    # 保存文件
    torch.save(ckpt, file_path)


def load_checkpoint(model, optimizer, file_path):
    # 加载文件
    ckpt = torch.load(file_path)
    # 加载模型参数
    model.load_state_dict(ckpt['model'])
    # 加载优化器参数
    optimizer.load_state_dict(ckpt['optimizer']) 
    # 设置开始的epoch
    epoch = ckpt['epoch']
    # 加载lr_scheduler
    #lr_schedule.load_state_dict(ckpt['lr_schedule'])
    return epoch


if __name__ == '__main__':
    fit(10)



```

经过10次迭代之后，可以得到：

```shell
Epoch 10 - Train accuracy: 99.56%, Train loss: 0.013176; Test accuracy: 99.20%, Test loss: 0.032355
```



整体效果比前一个例子的全部使用全连接层来的好。如果图像复杂的话，效果会有更明显提升。
