## PyTorch从精通到入门06：基于LSTM实现文本分类

在处理序列的数据时，基本都会使用RNN循环卷积模型。LSTM长短期记忆也是RNN的一种。本例子用PyTorch自带的LSTM模型，对中文的微博评论进行分类。

微博数据来自：[weibo_senti_100k | Kaggle](https://www.kaggle.com/datasets/fstcap/weibo-senti-100k)

一共有10万条评论，因为是中文，使用了jieba库来进行分词。分词之后，建立词表把词映射成数字。

然后建立一个模型，指定每个词的embedding维度为100，再经过LSTM处理后，送入全连接层进行分类。

代码如下：

```python

import torch
import torchtext.data.functional as textF

import torch.nn.functional as F
import numpy as np
import pandas as pd
import jieba
# pip install scikit-learn
from sklearn.model_selection import train_test_split

class TextProcessor():
    def __init__(self, min_occurrences=2, padding_len=50):
        self.min_occurrences = min_occurrences
        self.padding_len = padding_len
        self.words_cnt = 0
        self.vocab = None

    @classmethod
    def tokenize(self, str):
        txt = str.replace('！','').replace('，','').replace('。','').replace('@','').replace('/','')
        # 分词、返回列表
        return jieba.lcut(txt)

    def build_vocab(self, csv):
        # 统计每个词的次数
        word_cnt = pd.value_counts(np.concatenate(csv.review.values))
        # 删除次数较少的词
        word_cnt = word_cnt[word_cnt > self.min_occurrences]
        # 对每个词进行编码
        word_list = list(word_cnt.index)
        self.word_index = dict((word, word_list.index(word) + 1) for word in word_list)
        vocab = csv.review.apply(lambda t : [self.word_index.get(word, 0) for word in t])

        return (len(self.word_index) + 1, vocab)

    def load_file(self, csv_path):
        # 读取文件
        csv = pd.read_csv(csv_path)
        # 对review列分词
        csv['review'] = csv.review.apply(TextProcessor.tokenize)
        # 建立词汇表，将词语映射为数字
        self.words_cnt, self.vocab = self.build_vocab(csv)
        # Padding
        padding_text =  [v + (self.padding_len - len(v)) * [0] if len(v)<=self.padding_len else v[:self.padding_len] for v in self.vocab]
        padding_text = np.array(padding_text)

        # 标签
        labels = csv.label.values

        return (padding_text, labels)
 

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, vocabs, labels):
        self.vocabs = vocabs
        self.labels= labels
    
    def __getitem__(self,index):
        vocab = torch.LongTensor(self.vocabs[index])
        label= self.labels[index]
        return (vocab, label)
        
    def __len__(self):
        return len(self.vocabs)
    

class CommentClassification(torch.nn.Module):
    def __init__(self, num_classes, words_cnt, embedding_dim=100, hidden_size=200, rnn_layers=3, bidirectional=True) :
        super(CommentClassification, self).__init__()
        # 将单词编码成embedding_dim维的向量
        self.embedding = torch.nn.Embedding(words_cnt, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim,
            hidden_size,
            num_layers=rnn_layers,
            dropout=0.5, 
            # 双向RNN
            bidirectional=bidirectional
        )

        rnn_out = hidden_size
        if bidirectional:
            # 双向RNN时，隐藏层翻倍
            rnn_out = hidden_size * 2

        self.fc1 = torch.nn.Linear(rnn_out, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.output = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        y = self.embedding(x)
        r_o, _ = self.lstm(y)
        r_o = r_o[-1]
        y = F.dropout(F.relu(self.fc1(r_o)))
        y = F.dropout(F.relu(self.fc2(y)))
        y = self.output(y)
        return y


def build_dataloader(path, batch_size=32, padding_len=50):
    txt_processor = TextProcessor(2, padding_len)
    padding_text, labels = txt_processor.load_file(path)
    x_train, x_test, y_train, y_test = train_test_split(padding_text, labels)

    train_ds = TextDataset(x_train, y_train)
    test_ds = TextDataset(x_test, y_test)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return (txt_processor, train_dl, test_dl)

def train(model, device, dataloader):
    model.train()

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 所有批次累计损失和
    epoch_loss = 0
    # 累计预测正确的样本总数
    epoch_correct = 0
    
    for x, y in dataloader:
        x = x.permute(1, 0)
        x = x.to(device)
        y = y.to(device)

        predicted = model(x)
        loss = loss_func(predicted, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加
        with torch.no_grad():
            epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_loss += loss.item()

    return (epoch_loss, epoch_correct)


def test(model, device, dataloader):
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
     # 所有批次累计损失和
    epoch_loss = 0
    # 累计预测正确的样本总数
    epoch_correct = 0

    # 循环一次数据的多个批次
    # 测试模式，不需要梯度计算、反向传播
    with torch.no_grad():
        for x, y in dataloader:
            x = x.permute(1, 0)
            x = x.to(device)
            y = y.to(device)
            
            predicted = model(x)
            loss = loss_func(predicted, y)

            # 累加
            epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
            epoch_loss += loss.item()

    return (epoch_loss, epoch_correct)

def fit(epoch=20):
    padding_len = 50
    txt_processor, train_dl, test_dl = build_dataloader('./Data/weibo_senti_100k.csv', padding_len=padding_len)
    model = CommentClassification(2, txt_processor.words_cnt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 数据集总量
    total_train_data_cnt = len(train_dl.dataset)
    # 数据集批次数目
    num_train_batch = len(train_dl)
    # 数据集总量
    total_test_data_cnt = len(test_dl.dataset)
    # 数据集批次数目
    num_test_batch = len(test_dl)

    best_accuracy = 0.0

    for i in range(epoch):
        epoch_train_loss, epoch_train_correct = train(model, device, train_dl)
        # 所有批次的统计和/批次数量 = 平均损失率
        avg_train_loss = epoch_train_loss/num_train_batch
        # 预测正确的样本数/总样本数 = 平均正确率
        avg_train_accuracy = epoch_train_correct/total_train_data_cnt

        epoch_test_loss, epoch_test_correct = test(model, device, test_dl)
        # 所有批次的统计和/批次数量 = 平均损失率
        avg_test_loss = epoch_test_loss/num_test_batch
        # 预测争取的样本数/总样本数 = 平均正确率
        avg_test_accuracy = epoch_test_correct/total_test_data_cnt

        msg_template = ("Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
        print(msg_template.format(i+1, avg_train_accuracy*100, avg_train_loss, avg_test_accuracy*100, avg_test_loss))

        if avg_test_accuracy > best_accuracy:
            best_accuracy = avg_test_accuracy
            torch.save(model.state_dict(), 'lstm_comments.model')


if __name__ == '__main__':
    fit(5)



```

经过5次迭代之后，可以得到95%的准确率，效果还可以：

```shell
Epoch  1 - Train accuracy: 81.37%, Train loss: 0.360749; Test accuracy: 94.47%, Test loss: 0.130716
Epoch  2 - Train accuracy: 94.96%, Train loss: 0.123227; Test accuracy: 95.16%, Test loss: 0.121589
Epoch  3 - Train accuracy: 95.84%, Train loss: 0.109692; Test accuracy: 95.27%, Test loss: 0.127981
Epoch  4 - Train accuracy: 96.54%, Train loss: 0.097101; Test accuracy: 95.22%, Test loss: 0.125326
Epoch  5 - Train accuracy: 97.13%, Train loss: 0.081660; Test accuracy: 95.14%, Test loss: 0.138659
```
