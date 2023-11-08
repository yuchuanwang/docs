## PyTorch从精通到入门01：线性回归

没看错，确实是从精通到入门。

一开始看的时候，觉得很容易就掌握、精通了。但随着学习、实践的深入，发现已经掌握的知识点越来越赶不上还未掌握的知识点，所以自觉地把自己降低到入门的程度、低到尘埃里那种(虽然不知道能否从尘埃里开出花来)……



第一篇总结，是关于线性回归的的实现。

简单地说，有一坨x和y的数据，看起来它们的关系应该是线性的，但里面又有些噪音在干扰：

y = ax + b


那么，如何用PyTorch无脑的求出a和b的值呢？

```python
import torch 
import numpy as np
import random

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        # 一元一次方程，输入输出都为1维
        self.linear = torch.nn.modules.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)
    
def generate_data(batch_size=32):
    # 产生正态分布的输入
    x = torch.randn(batch_size)
    # 目标函数为：y = 3x + 4
    y = 3.0 * x + 4.0 + random.randint(-1, 1)

    return (x, y)

def train(epoch=500):
    # 有CUDA则用之
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LinearRegression().to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(epoch):
        x, y = generate_data()
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        # 把张量都移到合适的设备上
        x = x.to(device)
        y = y.to(device)

        # PyTorch标准套路
        predicted = model(x)
        loss = loss_func(predicted, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {i+1}, loss {loss.item()}')

    # 最终结果
    print(f'Weight: {model.linear.weight.data}')
    print(f'Bias: {model.linear.bias.data}')


if __name__ == '__main__':
    train(1000)


```



假设输入是这样的：

y = 3x + 4

其中还带了些随机的误差。

构造一个简单的模型，经过迭代之后，可以得到：

```shell
Weight: tensor([[2.9913]], device='cuda:0')
Bias: tensor([3.9066], device='cuda:0')
```

基本上已经接近3和4这两个预期的结果了。

这样，就解决了线性回归的问题。


