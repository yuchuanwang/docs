## PyTorch从精通到入门03：全连接层实现图像分类

这个例子使用全连接层，来识别PyTorch自带的手写数字图片，每个数字包含0~9。

相比后面的卷积层，全连接层比较简单，但精度比不上CNN卷积神经网络，仅供娱乐。

这个模型把28*28的输入，逐层输出为256、64、10。最后判断输出是0~9这十个数字中的哪一个。



```python
# 用全连接层来做图像分类

import torch
import torchvision

class ImageClassification(torch.nn.Module):
    def __init__(self, image_width, image_height, num_classifications) -> None:
        super(ImageClassification, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        image_size = image_width * image_height
        medium_features_1 = 256
        medium_features_2 = 64

        # 模型结构：输入28*28 -> 256 -> 64 -> 输出10
        self.input = torch.nn.Linear(in_features=image_size, out_features=medium_features_1)
        self.hidden = torch.nn.Linear(in_features=medium_features_1, out_features=medium_features_2)
        self.output = torch.nn.Linear(in_features=medium_features_2, out_features=num_classifications)

    def forward(self, x):
        # 将输入展平
        x_flatten = x.reshape(-1, self.image_width * self.image_height)
        y = self.input(x_flatten)
        y = torch.relu(y)

        y = self.hidden(y)
        y = torch.relu(y)

        y = self.output(y)

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
    # 每批次32个、随机打乱、使用pin_memory
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
        shuffle=True, **{'pin_memory': True})
    
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
    # 每批次32个、使用pin_memory
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
        shuffle=False, **{'pin_memory': True})
    
    return test_dataloader

def train(epoch=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImageClassification(28, 28, 10)
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
        #############################################################################################
        # 训练模式
        model.train()
        # 所有批次累计损失和
        epoch_train_loss = 0
        # 累计预测正确的样本总数
        epoch_train_correct = 0

        # 循环一次数据的多个批次
        for x, y in train_data:
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

        # 所有批次的统计和/批次数量 = 平均损失率
        avg_train_loss = epoch_train_loss/num_train_batch
        # 预测正确的样本数/总样本数 = 平均正确率
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt
        #############################################################################################

        #############################################################################################
        # 测试模式
        model.eval()
        # 所有批次累计损失和
        epoch_test_loss = 0
        # 累计预测正确的样本总数
        epoch_test_correct = 0

        # 循环一次数据的多个批次
        # 测试模式，不需要梯度计算、反向传播
        with torch.no_grad():
            for x, y in test_data:
                # non_blocking=True异步传输数据
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                
                predicted = model(x)
                loss = loss_func(predicted, y)

                # 累加
                epoch_test_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_test_loss += loss.item()

        # 所有批次的统计和/批次数量 = 平均损失率
        avg_test_loss = epoch_test_loss/num_test_batch
        # 预测争取的样本数/总样本数 = 平均正确率
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt
        #############################################################################################

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))


if __name__ == '__main__':
    train(10)



```



经过10次迭代之后，可以得到：

```shell
Epoch 10 - Train accuracy: 99.30%, Train loss: 0.020863; Test accuracy: 97.45%, Test loss: 0.110290
```

准确率其实还是可以的，但仅限于简单的图像识别，而且后面有更好的CNN方法来处理图像(计算机视觉)问题。
