## 用PyTorch, Profiler和TensorBoard优化AI训练性能

在AI模型的训练过程中，每一步训练基本上会包括如下的过程：

CPU: DataLoader Load Data --> CPU: Compile Operators -- Host2Device (Ops and Data) --> Device: Forward --> NIC: Collective Communication --> Device: BP --Device2Host--> CPU: Save Checkpoint

在这种异构、异步、集群的环境中，性能问题可能发生在图中的每个环节。为了定位、优化性能问题，需要进行采集、定位、优化，整体是比较复杂的过程。

按照经典的性能优化理论：如果你不能度量它，你就无法优化它。我们在对一个任务进行优化之前，首先需要采集到相关的性能数据，然后定位性能瓶颈，再进行性能优化，最后重新运行任务、对比前后的性能数据。

本文采用PyTorch + Profiler + TensorBoard为例，对AI训练任务的性能优化进行了一些实践和总结。

其中，基于PyTorch实现的AI模型训练做为被优化的任务；Profiler做为性能数据采集的工具；TensorBoard做为可视化性能数据、分析的工具。

#### 1. 环境搭建

跳过安装PyTorch的步骤不提，需要采集性能数据的话，我们需要安装对应的工具：

```shell
pip install torch_tb_profiler
```

通过此命令行，我们可以安装所需的Pytorch Profiler TensorBoard Plugin

安装完成之后，可以通过命令行：

```shell
tensorboard --logdir=<Your Logs Directory>
```

或者添加bind_all参数，使得可以从外部访问：

```shell
tensorboard --logdir=<Your Logs Directory> --bind_all
```

运行TensorBoard，然后根据该程序输出的结果，在浏览器里面访问可视化的性能数据分析(端口可能不一定是6006)，支持Chrome、Firefox和Edge浏览器，不支持Safari：

http://localhost:6006/#pytorch_profiler

如果在VS Code里面，可以在菜单View -> Command Palette输入：Launch TensorBoard，然后在VS Code里面直接查看性能数据。

---

*坑1：该插件在Windows下有bug，采集到的GPU数据无法正确显示。该问题至少在2022年就被发现，不过一直没修复：*

*https://discuss.pytorch.org/t/pytorch-profiler-not-profiling-gpu-on-windows/146685*

*所以，请在Linux下使用此插件和试验。*

---

#### 2. 待优化的AI模型训练任务

首先创建一个常见的CNN模型，并实现对它进行训练的代码。

```python
import torch
import torchvision

# Simple CNN classification model
class SimpleModel(torch.nn.Module):
    def __init__(self, img_size, hidden_channels=128, num_hidden_layers=3, num_classes=10):
        super().__init__()
        # Conv params
        self.strike = 1
        self.kernel_size = 3
        self.pool_size = 2
        self.num_classes = num_classes

        self.in_layer = torch.nn.Conv2d(in_channels=3, 
            out_channels=hidden_channels, 
            kernel_size=self.kernel_size, 
            padding='same')
        # padding=same means input size = output size, and max pool by 2
        self.img_size_after_conv = int(img_size/self.pool_size)

        # Build hidden layers
        hidden = []
        for i in range(num_hidden_layers):
            hidden.append(torch.nn.Conv2d(in_channels=hidden_channels, 
                out_channels=hidden_channels, 
                kernel_size=self.kernel_size, 
                padding='same'))
            hidden.append(torch.nn.ReLU())
            hidden.append(torch.nn.MaxPool2d(2))

            # Update size
            # padding=same means input size = output size, and max pool by 2
            self.img_size_after_conv = int(self.img_size_after_conv/self.pool_size)
        self.hidden_layers = torch.nn.Sequential(*hidden)

        flatten_size = hidden_channels * self.img_size_after_conv * self.img_size_after_conv
        self.fc1 = torch.nn.Linear(flatten_size, 256)
        self.out_layer = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        y = torch.max_pool2d(torch.relu(self.in_layer(x)), self.pool_size)
        y = self.hidden_layers(y)
        y = torch.flatten(y, 1)

        assert torch.numel(y) > self.num_classes

        y = torch.relu(self.fc1(y))
        y = self.out_layer(y)
        return y


def train(data, model, device, loss_fn, optimizer):
    x = data[0].to(device)
    y = data[1].to(device)
    predicted = model(x)

    loss = loss_fn(predicted, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def main():
    img_size = 224
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=img_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Will run on device: {device}')

    model = SimpleModel(img_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    num_batch = len(train_dl)
    for i in range(epochs):
        epoch_train_loss = 0.0
        for step, data in enumerate(train_dl):
            epoch_train_loss += train(data, model, device, loss_fn, optimizer)
        print(f'Loss of epoch {i + 1}: {epoch_train_loss/num_batch}')

if __name__ == '__main__':
    main()
```

运行此代码，可以看到该训练过程正常运行：

```shell
Will run on device: cuda
Loss of epoch 1: 1.3591355803450635
Loss of epoch 2: 0.8879912892596049
Loss of epoch 3: 0.6307942265417052
Loss of epoch 4: 0.391661481322841
Loss of epoch 5: 0.2135638974866784
```

#### 3. 记录性能数据

修改前面的main函数，加入Profiler的部分，进行性能数据采集：

```python
def main():
    img_size = 224
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=img_transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Will run on device: {device}')

    model = SimpleModel(img_size).to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create profiler
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='./logs/simple'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True)
    # Start profiling
    prof.start()
    epochs = 5
    num_batch = len(train_dl)
    abort = False
    for i in range(epochs):
        epoch_train_loss = 0.0
        for step, data in enumerate(train_dl):
            if step >= (1 + 2 + 1):
                # Break after profiling
                abort = True
                break

            epoch_train_loss += train(data, model, device, loss_fn, optimizer, prof)

            # Step profiling
            prof.step()

        if abort:
            break

        print(f'Loss of epoch {i + 1}: {epoch_train_loss/num_batch}')

    # Stop profiling
    prof.stop()Profiler的主要参数代表的意义如下所示：
```

- activities：需要记录哪些设备的性能活动，包括ProfilerActivity.CPU、ProfilerActivity.CUDA和ProfilerActivity.XPU。

- schedule：通过定义wait、warmup、active、repeat、skip_first这些数值，决定在哪些步骤记录性能数据。

- on_trace_ready：定义在记录完成时，所需要回调的的操作。

- record_shapes：是否记录算子输入形状的信息。

- profile_memory：是否跟踪张量的内存分配与释放。

- with_stack：是否记录算子的源代码信息(文件和行号)。

- with_flops：是否估计算子的FLOPs。

- with_modules：是否记录算子的调用栈所对应的模块层次结构(包括函数名)。

通过创建profiler，进行start/step/stop操作，就可以在指定的目录里面得到对应的JSON文件。后面可以用TensorBoard来打开并分析该性能数据。

---

*坑2：所产生的JSON文件，在Windows下用TensorBoard打开的话，会报错、失败：*

*json.decoder.JSONDecodeError: Invalid \escape*

*这是因为插件在Windows下用斜线"\"来分隔路径，这个符号在JSON解析时会失败。需要手动打开该JSON文件，将全部的"\"替换成"/"或者"\\\"。*

*Linux下没有这个问题，也不用替换。*

---

#### 4. 性能优化过程 - 消除Device2Host

我们先运行TensorBoard来打开前面一步采集到的性能数据，对该工具建立初步的概念：

```shell
tensorboard --logdir=./logs/
```

根据工具输出的结果，通过浏览器访问：[TensorBoard](http://localhost:6006/#pytorch_profiler)，可以得到这样的Overview结果：

![Overview.png](https://github.com/yuchuanwang/docs/blob/main/Assets/Overview.png)

<center>图1. 未优化前的Overview</center>

图中展示了系统的配置、GPU的使用率、每一步的整体执行时间等宏观信息。

对于性能分析与优化，更有用的信息在如下的Trace页面：

![Trace.png](https://github.com/yuchuanwang/docs/blob/main/Assets/Trace.png)

<center>图2. 未优化前的Trace</center>

图中，横轴X反映了代码的执行顺序、所消耗的时间；纵轴Y反映了代码的调用栈。

醒醒，到了需要仔细观察、寻找问题的阶段了。

从Trace的视图里面，可以看到：

1. 整个ProfilerStep#3，大概花了80ms的时间；

2. 横轴X上，耗费时间最多、接近一半的居然是aten::item这个函数

查一下PyTorch的文档：

[torch.Tensor.item &mdash; PyTorch 2.4 documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html#torch.Tensor.item)

可以看到：

> Tensor.item() → number[](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html#torch.Tensor.item)
> 
> Returns the value of this tensor as a standard Python number. This only works for tensors with one element. For other cases, see [`tolist()`](https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html#torch.Tensor.tolist "torch.Tensor.tolist").

继续看tolist的文档：

[torch.Tensor.tolist &mdash; PyTorch 2.4 documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html#torch.Tensor.tolist)

> Tensor.tolist() → list or number[](https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html#torch.Tensor.tolist)
> 
> Returns the tensor as a (nested) list. For scalars, a standard Python number is returned, just like with [`item()`](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html#torch.Tensor.item "torch.Tensor.item"). Tensors are automatically moved to the CPU first if necessary.

最后一句话解释了原因：

**Tensors are automatically moved to the CPU first if necessary.**

item()、tolist()函数会自动把数据从GPU/NPU搬到CPU上去处理。正是因为这个Device2Host的操作，导致了这个接近一半的耗时。

知道问题来源之后，那就先把它注释掉，不在每次训练时获取、打印loss信息：

```python
#print(f'Loss: {loss.item()}')
```

然后，再次运行、采集性能数据。可以得到新的结果：

![Overview_No_item.png](https://github.com/yuchuanwang/docs/blob/main/Assets/Overview_No_item.png)

<center>图3. 优化item()后的Overview</center>

可以看到，GPU的使用率从原来的60.99%上升到96.1%。简单的一行代码修改，基本就把GPU的计算能力压榨出来。

再看看新的Trace信息。

![Trace_No_item.png](https://github.com/yuchuanwang/docs/blob/main/Assets/Trace_No_item.png)

<center>图4. 优化item()后的Trace</center>

可以看到，原来最碍眼的aten::item()调用，已经消失了。而单步训练的时间，从原来的80ms，降到现在的57ms。效果非常明显。

#### 5. 性能优化过程 - 优化Host2Device

还没完。

在性能优化的过程中，我们一般都是先把第一个瓶颈的地方优化掉；然后寻找下一个瓶颈，直到满足性能要求。

随着item()的优化结束，在新的Trace视图中，可以看到第一个和第二个长长的函数调用，分别是dataloader的__next__，和aten::to。结合代码来看，这两个函数分别对应的是DataLoader加载训练数据集，然后把数据集从CPU侧搬到GPU/NPU侧，即Host2Device的过程。而这两个函数占用了整个步骤80%左右的时间。

我们试着通过num_workers、pin_memory、non_blocking这些技巧来优化它们的性能。

先了解一下这几个参数的意义：

关于num_workers：

[torch.utils.data &mdash; PyTorch 2.4 documentation](https://pytorch.org/docs/stable/data.html)

> ## Single- and Multi-process Data Loading[](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)
> 
> A [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") uses single-process data loading by default.
> 
> Within a Python process, the [Global Interpreter Lock (GIL)](https://wiki.python.org/moin/GlobalInterpreterLock) prevents true fully parallelizing Python code across threads. To avoid blocking computation code with data loading, PyTorch provides an easy switch to perform multi-process data loading by simply setting the argument `num_workers` to a positive integer.

num_workers默认是0，只使用主进程来加载数据。这种情况下，后续的操作会被加载数据的操作所阻塞。

通过将num_workers改为正整数，可以启动多进程来进行数据的加载、提升性能。

具体的num_workers数值是一个超参数，需要进行试验并选择最佳数值。

一般的说法是，num_workers = (节点上CPU的核数)/(节点上GPU的个数)，可以做为参考数值。

关于pin_memory：

[torch.utils.data &mdash; PyTorch 2.4 documentation](https://pytorch.org/docs/stable/data.html)

> ## Memory Pinning[](https://pytorch.org/docs/stable/data.html#memory-pinning)
> 
> Host to GPU copies are much faster when they originate from pinned (page-locked) memory. See [Use pinned memory buffers](https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-pinning) for more details on when and how to use pinned memory generally.
> 
> For data loading, passing `pin_memory=True` to a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") will automatically put the fetched data Tensors in pinned memory, and thus enables faster data transfer to CUDA-enabled GPUs.

pin_memory默认是False。通过将其改为True，可以避免内存页交换，加快从CPU内存搬到GPU内存的速度。当然也会有坏处，它会增加对系统内存的占用，否则PyTorch默认就可以把它设为True了。

当把DataLoader的pin_memory设为True之后，对应的Host2Device搬运操作：to()，一般会把non_blocking参数也设为True，以支持异步的搬运。

如下修改代码中的这三个参数：

```python
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
......

x = data[0].to(device, non_blocking=True)
y = data[1].to(device, non_blocking=True)
```

然后重新运行、采集性能数据。可以得到下面的结果。

![Overview_Workers.png](https://github.com/yuchuanwang/docs/blob/main/Assets/Overview_Workers.png)

图5. 优化Host2Device后的Overview

GPU的使用率现在是96.36%

![Trace_Workers.png](https://github.com/yuchuanwang/docs/blob/main/Assets/Trace_Workers.png)

图6. 优化Host2Device后的Trace

整体的单步训练耗时，也进一步从52ms降低到了10ms级别。

之前那俩显眼包：数据加载和to函数，从原来的80%时间占用，变成了基本看不见。主要的时间都花在了forward、backward和Optimizer.step()上面，这才是符合预期的。

*注：使用了多进程、异步传输之后，这个耗时变得不精确了。如果需要准确的时间对比，需要在每轮训练结束之前强制让cuda进行同步。*

#### 6. 常用优化思路

以上是性能分析优化的思考过程，以及两个经典的优化点。在实际项目中，有很多可以进行优化的方法，根据优化的发力点做如下分类。

##### 6.1 CPU侧的优化

**使用CPU的性能模式**：现在的CPU一般会分为高性能模式和各种省电模式。为了达到极致的性能要求，可以在BIOS里面把CPU的模式设置为Performance，使其固定运行在高主频。或者通过如下的脚本修改模式：

```shell
# For example, you have 8 cores to use
for i in `seq 0 7`
do
  echo performance > /sys/devices/system/cpu/cpu${i}/cpufreq/scaling_governor
done
```

**多进程加载数据**：修改DataLoader的num_workers为正整数，用多个进程来加载数据集。比如使用8个进程：

```python
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
```

**固定内存页**：修改DataLoader的pin_memory为True，并且修改CPU到GPU的to函数参数non_blocking为True，使得数据可以异步传输：

```python
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True,)
...
x = x.cuda(non_blocking=True)
```

**增大batch_size**：当发现GPU/NPU的显存还有比较大的空间，算力使用率也没用满的情况下，可以适当增大每次训练的batch size：

```python
# Increase batch size from 32 to 64
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True,)
```

**预加载数据到内存**：事先把数据集复制到Linux的内存中，然后DataLoader从内存读取数据而非从硬盘读取，从而加速数据加载的过程；而且在分布式训练中，同一节点上的多个训练进程都可以复用该内存，而不需要每个进程复制一份：

```shell
cp -r /dataset/path/on/disk /dev/shm/data
```

```python
dataset = CustomDataset(data_root="/dev/shm/data")
```

**用as_tensor()代替tensor()**：代码里面有时会使用as_tensor()、from_array()、tensor()、Tensor()这些方法来基于numpy数据创建张量。前两者在创建的时候，会直接复用numpy的内存；而后两者会进行内存的复制。最好使用as_tensor()、from_array()减少内存复制的开销。

**Host2Device问题的信号**：当在Profiler中发现to, copy_, cudaStreamSynchronize这些函数调用的时候，那么就需要去留意这个H2D操作是否是必不可少的。如果能去除这个操作，就尽量把它去掉。



##### 6.2 Device侧的优化

**直接在Device上创建张量**：在创建张量时，如果你已经确定这个张量会在Device上被使用，那么在创建时，直接指定device参数；而不是创建之后再去to(device)。比如下列的函数，和其他包含device参数的函数：

```python
# Bad
torch.rand(size).cuda()
# Good
torch.rand(size, device='cuda')


# Bad
torch.zeros(size).cuda()
# Good
torch.zeros(size, device='cuda')

# Bad
torch.full(size).cuda()
# Good
torch.full(size, device='cuda')


# Bad
torch.tensor(...).cuda()
# Good
torch.tensor(..., device='cuda')
```

**留意代码中的这些函数**：print(), item(), assert(), cpu(), numpy()。这些函数需要在CPU上执行。当这些函数处理的数据本来在GPU/DPU上，PyTorch会默默的把它们搬运到CPU，然后再执行。而这个操作会带来明显的性能问题。比如说，前面的代码里面就依然存在类似的问题。

**避免在前向计算、反向传播的代码中创建对象、或者复制数据**：在实现代码的时候，我们有时候会不经意的在前向计算，或者反向传播的相关代码中，创建一些对象，或者执行一些数据的复制。虽然从代码上来看，它们只出现一次，但在训练过程中，它们会被反复的执行，从而带来很严重的性能问题。所以，尽可能把这些对象、数据的的操作移到到别的地方，比如说模型的构造函数，让它们只在开始的时刻执行一次。

**Device2Host问题的信号**：当在Profiler中发现item, cudaMemCpyAsync这些函数调用的时候，那么就需要去留意这个D2G操作是否是必不可少的。如果能去除这个操作，就尽量把它去掉。

**使用torch.compile**：通过它，将模型从Eager模式编译成Graph模式，提高模型的运行效率：

```python
model = torch.compile(model)
```

除了模型之外，还可以把损失函数、激活函数都先编译，然后再去使用：

```python
loss_func = torch.compile(torch.nn.CrossEntropyLoss().cuda(device))
```

也可以把一些算子融合，然后通过编译修饰符进行加速：

```python
@torch.compile
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
```

*注：这个功能对GPU有要求，比如我实验的P100就不支持……*

**使用混合精度**：Automatic Mixed Precision(AMP)，即自动混合精度。详细的官方介绍在这里：https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
通过将计算精度从fp32降低到fp16或者bf16，可以加速计算速度、减少内存使用；尤其在具有TensorCore的设备上，会有更突出的效果：

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    outputs = model(inputs)
    loss = loss_func(outputs, labels)
```

##### 6.3 集合通信的优化

集合通信的优化，包括集合通信算法、拓扑感知、计算通信重叠/掩盖等等，这些有点复杂。
暂时还没整理完毕。留着以后再单独写。



#### 7. 性能优化总结

性能优化是很苦恼而有趣的过程，中间会有很多谜一样的问题，需要去不停的思考、假设、修改、验证。

而对于AI集群训练的这个场景，我觉得在优化之前，有几个问题需要很清楚的回答，才能做好性能优化的工作：

- 这个workload的原理是什么？

- 这个代码是在什么地方执行？

- 这个代码所操作的数据是在什么地方？

虽然，能回答这些问题不一定能把性能问题解决掉，但回答不出来的话，肯定是解决不掉的。


