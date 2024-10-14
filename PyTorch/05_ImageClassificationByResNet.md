## PyTorch从精通到入门05：基于ResNet迁移学习和微调，实现图像分类

目前PyTorch自带了很多著名的CNN模型，可以用来帮我们提取图像的特征，然后基于提取到的特征，我们再自己设计如何基于特征进行分类。试验下来，可以发现分类的准确率比自己搭一个CNN模型好了不少。

这就是迁移学习(Transfer Learning)的概念。在做迁移学习时，一般的思路是，利用预训练模型的卷积部分提取数据集的特征，重新训练分类器。

等到分类器训练完毕之后，将冻结的卷积基解冻，使得卷积基适应当前数据集，更好的提取特征。这个就是所谓的微调(Fine tune)了。

下面的例子，数据部分，用了kaggle上的一个图像数据集，里面有15种不同蔬菜的照片，累计照片总数为21000张。下载地址：https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/data

下载结束后，请把文件解压缩到与代码同级的Data目录。

模型部分，使用了ResNet50模型，用它来提取图像的特征。



```python

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class VegetableDataset:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.train_dataset_dir = r'./Data/Vegetable/train'
        self.test_dataset_dir = r'./Data/Vegetable/test'
        self.validation_dataset_dir = r'./Data/Vegetable/validation'

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.validation_dataset = None
        self.validation_dataloader = None

        self.id_to_class = dict()

    def load_train_data(self):
        self.train_dataset = torchvision.datasets.ImageFolder(self.train_dataset_dir, transform=self.transform)
        print(self.train_dataset.classes)
        print(self.train_dataset.class_to_idx)
        print(f'Train dataset size: {len(self.train_dataset)}')

        # Reverse from: label -> id, to: id -> label
        self.id_to_class = dict((val, key) for key, val in self.train_dataset.class_to_idx.items())
        print(self.id_to_class)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, 
            shuffle=True, **{'pin_memory': True})
        return self.train_dataloader

    def load_test_data(self):
        self.test_dataset = torchvision.datasets.ImageFolder(self.test_dataset_dir, transform=self.transform)
        print(f'Test dataset size: {len(self.test_dataset)}')

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, 
            **{'pin_memory': True})
        return self.test_dataloader
    
    def load_validation_data(self):
        self.validation_dataset = torchvision.datasets.ImageFolder(self.validation_dataset_dir, transform=self.transform)
        print(f'Validation dataset size: {len(self.validation_dataset)}')

        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size, 
            **{'pin_memory': True})
        return self.validation_dataloader
    
    def show_sample_images(self):
        images_to_show = 6
        imgs, labels = next(iter(self.train_dataloader))
        plt.figure(figsize=(56, 56))
        for i, (img, label) in enumerate(zip(imgs[:images_to_show], labels[:images_to_show])):
            # permute交换张量维度，把原来在0维的channel移到最后一维
            img = (img.permute(1, 2, 0).numpy() + 1)/2
            # rows * cols
            plt.subplot(2, 3, i+1)
            plt.title(self.id_to_class.get(label.item()))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
        
        # Show all images
        plt.show()


class VegetableResnet(torch.nn.Module):
    def __init__(self, image_width=224, image_height=224, num_classifications=15, 
                 enable_dropout=False, enable_bn=False):
        super().__init__()

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        print(self.resnet)

        fc_features = 128
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(resnet_features, fc_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(fc_features, num_classifications)
        )

    def forward(self, x):
        y = self.resnet(x)
        return y
    
    def get_name(self):
        return 'VegetableResnet50'
    
    def transfer_learning_mode(self):
        # 冻结卷积基
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 解冻全连接层
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def fine_tune_mode(self):
        # 解冻卷积基
        for param in self.resnet.parameters():
            param.requires_grad = True


class ModelTrainer():
    def __init__(self, model, loss_func, optimizer, lr_scheduler=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.model = self.model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self, dataloader):
        # 训练模式
        self.model.train()
        # 所有批次累计损失和
        epoch_loss = 0
        # 累计预测正确的样本总数
        epoch_correct = 0

        # 循环一次数据的多个批次
        for x, y in dataloader:
            # non_blocking=True异步传输数据
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            predicted = self.model(x)
            loss = self.loss_func(predicted, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录已经训练了多少个epoch并触发学习速率的衰减
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 累加
            with torch.no_grad():
                epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_loss += loss.item()

        return (epoch_loss, epoch_correct)

    def test(self, dataloader):
        # 测试模式
        self.model.eval()
        # 所有批次累计损失和
        epoch_loss = 0
        # 累计预测正确的样本总数
        epoch_correct = 0

        # 循环一次数据的多个批次
        # 测试模式，不需要梯度计算、反向传播
        with torch.no_grad():
            for x, y in dataloader:
                # non_blocking=True异步传输数据
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                
                predicted = self.model(x)
                loss = self.loss_func(predicted, y)

                # 累加
                epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_loss += loss.item()

        return (epoch_loss, epoch_correct)
    
    def validate(self, dataloader):
        total_val_data_cnt = len(dataloader.dataset)
        num_val_batch = len(dataloader)
        val_loss, val_correct = self.test(dataloader)
        # 所有批次的统计和/批次数量 = 平均损失率
        avg_val_loss = val_loss/num_val_batch
        # 预测正确的样本数/总样本数 = 平均正确率
        avg_val_accuracy = val_correct/total_val_data_cnt

        return (avg_val_loss, avg_val_accuracy)

    def fit(self, train_dataloader, test_dataloader, epoch):
        # 数据集总量
        total_train_data_cnt = len(train_dataloader.dataset)
        # 数据集批次数目
        num_train_batch = len(train_dataloader)
        # 数据集总量
        total_test_data_cnt = len(test_dataloader.dataset)
        # 数据集批次数目
        num_test_batch = len(test_dataloader)

        best_accuracy = 0.0

        # 循环全部数据
        for i in range(epoch):
            # 训练模型
            epoch_train_loss, epoch_train_correct = self.train(train_dataloader)
            # 所有批次的统计和/批次数量 = 平均损失率
            avg_train_loss = epoch_train_loss/num_train_batch
            # 预测正确的样本数/总样本数 = 平均正确率
            avg_train_accuracy = epoch_train_correct/total_train_data_cnt

            # 测试模型
            epoch_test_loss, epoch_test_correct = self.test(test_dataloader)
            # 所有批次的统计和/批次数量 = 平均损失率
            avg_test_loss = epoch_test_loss/num_test_batch
            # 预测争取的样本数/总样本数 = 平均正确率
            avg_test_accuracy = epoch_test_correct/total_test_data_cnt

            msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
            print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

            # CheckPoint
            if avg_test_accuracy > best_accuracy:
                # 保存最佳测试模型
                best_accuracy = avg_test_accuracy
                ckpt_path = f'./{self.model.get_name()}.ckpt'
                self.save_checkpoint(i, ckpt_path)
                print(f'Save model to {ckpt_path}')

    def predict(self, x):
        # Prediction
        prediction = self.model(x.to(self.device))
        # Predicted class value using argmax
        #predicted_class = np.argmax(prediction)
        return prediction

    def save_checkpoint(self, epoch, file_path):
        # 构造CheckPoint内容
        ckpt = {
            'model': self.model.state_dict(), 
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch, 
            #'lr_schedule': self.lr_schedule.state_dict()
        }
        # 保存文件
        torch.save(ckpt, file_path)

    def load_checkpoint(self, file_path):
        # 加载文件
        ckpt = torch.load(file_path)
        # 加载模型参数
        self.model.load_state_dict(ckpt['model'])
        # 加载优化器参数
        self.optimizer.load_state_dict(ckpt['optimizer']) 
        # 设置开始的epoch
        epoch = ckpt['epoch']
        # 加载lr_scheduler
        #self.lr_schedule.load_state_dict(ckpt['lr_schedule'])
        return epoch


def train_with_resnet(including_finetune=True):
    model = VegetableResnet()
    model.transfer_learning_mode()
    loss_func = torch.nn.CrossEntropyLoss()
    # 仅优化分类器参数
    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=0.0001)

    veg = VegetableDataset(batch_size=16)
    # 训练数据
    train_dataloader = veg.load_train_data()
    #veg.show_sample_images()
    # 训练数据
    test_dataloader = veg.load_test_data()
    # 验证数据
    validation_dataloader = veg.load_validation_data()

    # Train model and save best one
    print('Begin transfer learning...')
    trainer = ModelTrainer(model, loss_func, optimizer)
    trainer.fit(train_dataloader, test_dataloader, 5)

    if including_finetune:
        # 微调
        model.fine_tune_mode()
        # 较小的lr
        optimizer_finetune = torch.optim.Adam(model.parameters(), lr=0.00001)
        print('Begin fine tune...')
        trainer = ModelTrainer(model, loss_func, optimizer_finetune)
        trainer.fit(train_dataloader, test_dataloader, 2)

    # Load best model
    #trainer.load_checkpoint('./VegetableResnet50.ckpt')
    avg_val_loss, avg_val_accuracy = trainer.validate(validation_dataloader)
    print(f'Validation: {avg_val_accuracy * 100}%, {avg_val_loss}')
    
    # Try to predict single image
    images = [
            './Data/Vegetable/validation/Bean/0192.jpg', 
            './Data/Vegetable/validation/Cabbage/1202.jpg',
            './Data/Vegetable/validation/Carrot/1202.jpg',
            './Data/Vegetable/validation/Cauliflower/1258.jpg',
            './Data/Vegetable/validation/Papaya/1004.jpg',
            './Data/Vegetable/validation/Potato/1202.jpg',
            './Data/Vegetable/validation/Pumpkin/1202.jpg',
            './Data/Vegetable/validation/Tomato/1202.jpg'
        ]
    for path in images:
        img = Image.open(path)
        img_tensor = veg.transform(img)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(trainer.device)
        prediction = trainer.predict(img_tensor)
        # numpy需要到CPU上操作
        index = prediction.to('cpu').data.numpy().argmax()
        label = veg.id_to_class[index]
        print(label)

if __name__ == '__main__':    
    train_with_resnet(True)
    


```

经过5次迭代之后，就可以得到99.80%的准确率：

```shell
Begin transfer learning...
Epoch  1 - Train accuracy: 89.63%, Train loss: 0.936581; Test accuracy: 98.37%, Test loss: 0.195718
Save model to ./VegetableResnet50.ckpt
Epoch  2 - Train accuracy: 98.05%, Train loss: 0.162241; Test accuracy: 99.37%, Test loss: 0.061107
Save model to ./VegetableResnet50.ckpt
Epoch  3 - Train accuracy: 99.10%, Train loss: 0.077781; Test accuracy: 99.67%, Test loss: 0.035570
Save model to ./VegetableResnet50.ckpt
Epoch  4 - Train accuracy: 99.39%, Train loss: 0.048853; Test accuracy: 99.63%, Test loss: 0.025435
Epoch  5 - Train accuracy: 99.58%, Train loss: 0.033993; Test accuracy: 99.80%, Test loss: 0.016173
Save model to ./VegetableResnet50.ckpt
Validation: 99.83333333333333%, 0.014029651822366236
```

效果是相当的好，比我自己设计CNN模型的准确率高了不少。

如果再加上微调的话，效果更好了，微调2次，准确率达到99.90%：

```shell
Begin fine tune...
Epoch  1 - Train accuracy: 99.70%, Train loss: 0.013348; Test accuracy: 99.90%, Test loss: 0.005066
Save model to ./VegetableResnet50.ckpt
Epoch  2 - Train accuracy: 99.93%, Train loss: 0.003862; Test accuracy: 99.90%, Test loss: 0.003799
Validation: 99.96666666666667%, 0.003069976973252546
```

只不过微调跑起来太慢了，每一轮需要耗不少的时间。

所以在实际的应用中，基于现有的模型，然后去做迁移学习、微调是个比较靠谱的方法。
