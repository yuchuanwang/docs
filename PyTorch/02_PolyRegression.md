## PyTorch从精通到入门02：多项式回归

线性拟合的函数是一个一次函数，用一元一次函数即可表达。

如果需要更高阶的多项式，如二次、三次等等时，计算的思路基本相同。

比如如下一元三次方程，从数学角度来看，输入量即自变量仍然只有一个，即x：

y = ax^3 + bx^2 + cx + d

但是从神经网络的角度来看，可以将其看成三个输入量，即将x的不同次方均当作一个单独的输入量。

由此可知，需要构建的神经网络具有三个输入，但是输出仍然只有一个。



```python
import torch
import numpy as np
import random

class PolyRegression(torch.nn.Module):
    def __init__(self, input_dim):
        super(PolyRegression, self).__init__()
        self.linear = torch.nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)

def generate_data(batch_size=32):
    # 产生正态分布随机数、添加一个维度
    # 32行1列
    x = torch.randn(batch_size).unsqueeze(1)
    # f(x) = 2 * x^3 + 3 * x^2 + 4 * x + 5
    # 32行1列
    y = 2.0 * x**3 + 3.0 * x**2 + 4.0 * x + 5.0 + random.random()/10

    # 将x, x平方、x三次方作为输入
    # 按维度1拼接（横着拼）
    # 32行3列
    x_data = torch.cat([x**i for i in range(1, 4)], 1)
    return (x_data, y)

def train(epoch=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PolyRegression(3).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(epoch):
        x, y = generate_data(32)
        x = x.to(device)
        y = y.to(device)

        # PyTorch标准流程
        predicted = model(x)
        loss = loss_func(predicted, y)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 优化参数
        optimizer.step()

        print(f'Epoch {i+1}, loss {loss.item()}')

    # 最终结果
    print(f'Weight: {model.linear.weight.data}')
    print(f'Bias: {model.linear.bias.data}')


if __name__ == '__main__':
    train(2000)


```



经过迭代之后，可以得到：

```shell
Weight: tensor([[3.9090, 3.0290, 2.0208]], device='cuda:0')
Bias: tensor([4.9937], device='cuda:0')
```

接近2、3、4、5的预期结果。

如此解决了多项式回归的问题。


