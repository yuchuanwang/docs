## MPI高性能计算和集合通信编程

MPI (Message Passing Interface)，是高性能计算(High Performance Computing - HPC)领域中广泛使用的并行编程模式。

本来MPI跟我的工作没啥紧密的关系，不过随着近年来大模型和分布式训练的发展，各家大大小小的芯片公司纷纷模仿NVIDIA的NCCL集合通信库，推出了各种各样的xCCL。而各种集合通信库的实现里面，都需要实现或定制MPI的集合通信原语。

因此，在了解了一些集合通信库的实现后，特意回头来看看这项技术的源头MPI。理解了这些基础的通信原语，有利于更加深入的理解各种各样的集合通信库。

本文将基于Ubuntu和OpenMP，来展示如何进行MPI的编程；并逐个调用集合通信原语，展示这些原语如何使用、能得到什么样的结果。

#### 环境准备

为了进行MPI的编程，需要先安装这些软件：

```shell
sudo apt install mpich libopenmpi-dev
```

mpich是MPI代码的编译器；libopenmpi-dev是OpenMP的开发包。

#### Hello MPI

既然学习一个新语言的时候，大家都用Hello World作为起点，那么我们也用Hello MPI来作为MPI学习的起点吧。

先创建一个hello_mpi.cpp，输入以下代码、编译、执行。看看效果之后，再来介绍MPI编程的基础概念。

代码如下：

```cpp
#include <iostream>
#include <mpi.h>

void MpiWithRank(int current_rank, int world_size)
{
    std::cout << "Hello MPI from rank: " << current_rank << "/" << world_size << std::endl;
}

int main(int argc, char **argv)
{
    // Begin
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int current_rank = 0;
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    MpiWithRank(current_rank, world_size);

    // End
    MPI_Finalize();

    return 0;
}
```

然后，编译这个文件：

```shell
mpic++ hello_mpi.cpp -o hello_mpi
```

编译成功之后，执行命令：

```shell
mpirun -np 4 hello_mpi
```

一切正常的话，将会得到类似下面的输出：

```shell
Hello MPI from rank: 3/4
Hello MPI from rank: 0/4
Hello MPI from rank: 1/4
Hello MPI from rank: 2/4
```

现在，我们来介绍一下这些代码和步骤都做了些啥。

MPI编程里面，有几个最基础的函数：

int MPI_Init(&argc, &argv)：它作为MPI程序的第一个函数调用，负责初始化MPI并行环境。

int MPI_Finalize()：它作为MPI程序的最后一个函数调用，退出MPI并行环境。

int MPI_Comm_size(MPI_COMM_WORLD, &world_size)：获得默认通信域内的进程数目world_size。也可以理解为，在默认的进程组里面，存在多少个进程在并行工作。

int MPI_Comm_rank(MPI_COMM_WORLD, &current_rank)：获得本进程在默认通信域的编号，该编号从0开始，一直增加到world_size-1。比如我们前面的例子里，运行了四个进程做并行工作，那么world_size就是4，而每个进程的rank分别是0、1、2、3。

两个命令行里面，mpic++用来编译MPI + C++代码，并得到可执行文件。如果编译MPI + C代码的话，则换成mpicc即可。

mpirun则用来运行编译好的可执行文件，-np 4代表需要启动4个进程，这四个进程构成通信域/进程组，一起并行工作。

4个进程启动后，它们在各自的MpiWithRank函数里，分别输出了各自的rank序号和通信域的world_size。

这个就是MPI程序的最基础代码了。后面所有的例子，都是基于这个框架和步骤，逐渐增加功能的。

#### Send/Recv

Send和Recv是基础的通信操作，在MPI的概念里，意味着从某个rank/进程发出一些数据，然后在另一个rank/进程里接收这些数据。

在大模型分布式训练中，执行流水线并行时，不同的stage之间，就可以通过Send/Recv来收发前向和反向的数据。

涉及到的函数如下所示：

int MPI_Send(void *buff, int count, MPI_Datatype datatype, int recipient, int tag, MPI_Comm communicator)：buff是打算发送的数据；count是数据的个数；datatype是数据的类型；recipient表示数据要发送到哪个rank；tag是收发双方用来校验消息的，本文中没用它；communicator是使用哪个通信域/进程组。

int MPI_Recv(void *buff, int count, MPI_Datatype datatype, int sender, int tag, MPI_Comm communicator, MPI_Status *status)：参数和MPI_Send很类似，主要的区别在于，buff用来保存收到的数据；sender表示数据是从哪个rank发送过来的。

接下来，新建一个send_recv.cpp，并输入以下内容：

```cpp
#include <iostream>
#include <mpi.h>

void SendRecv(int current_rank, int world_size)
{
    // Recv data from previous rank
    int src_rank = current_rank - 1;
    if (src_rank < 0)
    {
        // For rank 0, recv from the last rank
        src_rank = world_size - 1;
    }

    int dst_rank = current_rank + 1;
    if (dst_rank >= world_size)
    {
        // For last rank, send to the first rank
        dst_rank = 0;
    }

    int data_send = 11;
    int tag = 121;
    MPI_Status status;
    if (current_rank == 0)
    {
        // Send to next rank
        std::cout << "Rank " << current_rank << ": Send data " << data_send 
            << " to rank " << dst_rank << std::endl;
        MPI_Send(&data_send, 1, MPI_INT, dst_rank, tag, MPI_COMM_WORLD);

        // Recv from last rank
        MPI_Recv(&data_send, 1, MPI_INT, src_rank, tag, MPI_COMM_WORLD, &status);
        std::cout << "Rank " << current_rank << ": Recv data " << data_send 
            << " from rank " << src_rank << std::endl;
    }
    else
    {
        // Recv from previous rank
        MPI_Recv(&data_send, 1, MPI_INT, src_rank, tag, MPI_COMM_WORLD, &status);
        std::cout << "Rank " << current_rank << ": Recv data " << data_send 
            << " from rank " << src_rank << std::endl;

        // Modify the data before sending to next rank
        data_send += 1;

        // Send to next rank
        std::cout << "Rank " << current_rank << ": Send data " << data_send 
            << " to rank " << dst_rank << std::endl;
        MPI_Send(&data_send, 1, MPI_INT, dst_rank, tag, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    // Begin
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int current_rank = 0;
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    SendRecv(current_rank, world_size);

    // End
    MPI_Finalize();

    return 0;
}
```

编译：

```shell
mpic++ send_recv.cpp -o send_recv
```

运行：

```shell
mpirun -np 4 send_recv
```

可以得到输出：

```shell
Rank 0: Send data 11 to rank 1
Rank 1: Recv data 11 from rank 0
Rank 1: Send data 12 to rank 2
Rank 2: Recv data 12 from rank 1
Rank 2: Send data 13 to rank 3
Rank 3: Recv data 13 from rank 2
Rank 3: Send data 14 to rank 0
Rank 0: Recv data 14 from rank 3
```

用下面这张图来解释一下这段代码做了些什么：

![mpi_send_recv.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_send_recv.png)

一共运行了4个进程并行工作，进程rank 0把11发送给下一个进程rank 1；rank 1收到后，将数据增加1之后，按顺序发给下一个进程rank 2；以此类推，最后一个进程再把最终的数据发送回rank 0。由此构成了一个环形的Send/Recv操作。

了解完Send/Recv之后，我们接下来看第一个集合通信操作：广播。

#### Broadcast

广播操作执行的时候，会把指定的数据，从源rank复制到通信域里面的其他所有的ranks。

在大模型的分布式训练中，执行数据并行时，需要把模型/参数从一个GPU复制到DP域内的其它GPU时，用的就是广播操作。

对应的函数是：

int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int emitter_rank, MPI_Comm communicator)：具体参数的含义和前面的Send/Recv基本类似，不一样的是这里用emitter_rank来表示从哪个rank广播数据到其他所有的ranks。

在调用MPI_Bcast的时候，虽然接收方的那些ranks不执行广播操作，但一样需要调用这个函数。具体MPI_Bcast的实现我还没去看过，但可以猜到的是，如果不调用的话，这些被广播的进程们，它们怎么知道要接收数据、要把收到的数据放到什么地方呢？其他的集合通信操作也是同样的道理。

看看具体怎么用的。

新建一个broadcast.cpp文件：

```cpp
#include <iostream>
#include <mpi.h>

void Broadcast(int current_rank, int world_size)
{
    int data_bcast = 0;
    int root_rank = 0;
    if (current_rank == root_rank)
    {
        // Root rank to broadcast data
        data_bcast = 10;
    }

    std::cout << "Rank " << current_rank << ": original data is " << data_bcast << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&data_bcast, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << current_rank << ": new data is " << data_bcast << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int current_rank = 0;
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    Broadcast(current_rank, world_size);

    MPI_Finalize();
}
```

然后编译这个文件：

```shell
mpic++ ./broadcast.cpp -o broadcast
```

接下来执行它：

```shell
mpirun -np 4 broadcast
```

可以得到如下的输出：

```shell
Rank 0: original data is 10
Rank 1: original data is 0
Rank 2: original data is 0
Rank 3: original data is 0
Rank 1: new data is 10
Rank 0: new data is 10
Rank 3: new data is 10
Rank 2: new data is 10
```

这个代码执行的操作如下图所示：

![mpi_bcast.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_bcast.png)

rank 0把数据10，广播到了通信域内的其他所有rank。

另外，例子里为了数据打印整齐，用了MPI_Barrier让各个进程进行同步，这里不详述。

接下来，看看收集操作。

#### Gather

收集操作会把通信域内所有进程/ranks的数据，收集到指定的rank中。收集后的数据顺序，跟各个rank的序号一致。

对应的函数是：

int MPI_Gather(const void* buffer_send, int count_send, MPI_Datatype datatype_send, void* buffer_recv, int count_recv, MPI_Datatype datatype_recv, int root, MPI_Comm communicator)：从参数可以看出，执行这个函数时，会把各个rank上面的buffer_send，发送到root rank上面的buffer_recv；并且，这些发送来的数据，会按照rank的次序，在root rank依次排列。

下面通过代码来看看怎么使用这个函数。

新建gather.cpp文件：

```cpp
#include <iostream>
#include <sstream>
#include <mpi.h>

void PrintArray(int* array, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void GatherNumbers(int current_rank, int world_size)
{
    int root_rank = 0;
    int data_sent = (current_rank + 1) * 100;
    std::cout << "Rank " << current_rank << ": Send data " << data_sent 
        << " to root for gathering" << std::endl;

    if (current_rank == root_rank)
    {
        // Alloc buffer to gather data
        int buffer[world_size];
        for (int i = 0; i < world_size; i++)
        {
            buffer[i] = 0;
        }
        std::cout << "Before gather, the data in root rank is: " << std::endl;
        PrintArray(buffer, world_size);

        // Gather data
        MPI_Gather(&data_sent, 1, MPI_INT, 
            buffer, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

        std::cout << "After gather, the data in root rank is: " << std::endl;
        PrintArray(buffer, world_size);
    }
    else
    {
        // Send data to root
        MPI_Gather(&data_sent, 1, MPI_INT, 
            NULL, 0, MPI_INT, 
            root_rank, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int current_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    GatherNumbers(current_rank, world_size);
    //GatherString(current_rank, world_size);

    MPI_Finalize();
}
```

编译并执行这个文件：

```shell
mpic++ ./gather.cpp -o gather
mpirun -np 4 gather
Rank 1: Send data 200 to root for gathering
Rank 2: Send data 300 to root for gathering
Rank 3: Send data 400 to root for gathering
Rank 0: Send data 100 to root for gathering
Before gather, the data in root rank is:
0 0 0 0
After gather, the data in root rank is:
100 200 300 400
```

可以看到rank 0把所有ranks的数据，按照次序收集到了数组里。

具体过程如下图所示：

![mpi_gather.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_gather.png)

MPI_Gather收集数据时，是做了一个假设：每个rank所发过来的的数据所占空间一致。针对每个rank的数据长度不一致的情况，可以使用MPI_Gatherv函数。

接下来看看收集的反向操作：分散。

#### Scatter

分散操作与收集操作相反，它会把指定rank上的数据，按照各个rank的序号，逐个分给通信域内的所有ranks。

对应的函数是：

int MPI_Scatter(const void* buffer_send, int count_send, MPI_Datatype datatype_send, void* buffer_recv, int count_recv, MPI_Datatype datatype_recv, int root, MPI_Comm communicator)：函数很长，看起来有点吓人。它的意思是，把root rank上的buffer_send数据，依次分散到communicator通信域内各个rank的buffer_recv上。

这里还有些细节没展开，比如原始数据的长度不够、太长、不能被rank个数整除等等，这些特殊情况下怎么处理。

看看怎么使用这个函数来分散数据。

新建scatter.cpp文件：

```cpp
#include <iostream>
#include <mpi.h>

void PrintArray(int* array, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void ScatterNumbers(int current_rank, int world_size)
{
    int root_rank = 0;
    int data_recv = 0; 
    if (current_rank == root_rank)
    {
        // Create an array whose size is the same as the world size
        int buffer[world_size];
        // Set the values of array
        for (int i = 0; i < world_size; i++)
        {
            // buffer containing all values
            buffer[i] = (i + 1) * 100;
        }

        // Before scatter, the data is
        std::cout << "Rank 0: Array values to be scattered: " << std::endl; 
        PrintArray(buffer, world_size);

        // Dispatch buffer values to all the processes in the same communicator
        // If the array length is bigger than world size, it will only is the previous world_size items
        // If it is smaller than world size, the extra ranks will receive random number
        MPI_Scatter(buffer, 1, MPI_INT, &data_recv, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

        std::cout << "Rank " << current_rank << ": scatter value " << data_recv << std::endl; 
    }
    else
    {
        // Receive the dispatched data
        MPI_Scatter(NULL, 1, MPI_INT, &data_recv, 1, MPI_INT, root_rank, MPI_COMM_WORLD);

        // After scatter
        std::cout << "Rank " << current_rank << ": scatter value " << data_recv << std::endl; 
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int current_rank = 0;
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    ScatterNumbers(current_rank, world_size);
    //ScatterString(current_rank, world_size);

    MPI_Finalize();
}
```

编译并执行该文件，可以得到如下结果：

```shell
mpic++ ./scatter.cpp -o scatter
mpirun -np 4 ./scatter
Rank 0: Array values to be scattered:
100 200 300 400
Rank 0: scatter value 100
Rank 1: scatter value 200
Rank 2: scatter value 300
Rank 3: scatter value 400
```

该代码将rank 0上的数组，按照各个rank的次序，依次分散到不同的rank中。在此过程中，rank 0上的原始数组是不会发生变化的。

具体过程如下图所示：

![mpi_scatter.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_scatter.png)

和MPI_Gatherv类似，如果需要给每个rank分散不同长度的数据，可以使用MPI_Scatterv函数。

下一个操作是归约。

#### Reduce

归约操作在大模型训练和大数据里面是很重要的操作。比如说，通过Reduce操作，可以把通信域内所有ranks的数据，发送到指定的root rank；然后在root rank上，对所有的数据进行计算，比如求和、求最大值、求最小值、求乘积等等。

对应的函数如下：

int MPI_Reduce(const void* send_buffer, void* receive_buffer, int count, MPI_Datatype datatype, MPI_Op operation, int root, MPI_Comm communicator)：又是一个很长的函数，它的意思是，把communicator通信域内各个rank上的send_buffer，发送到root rank的receive_buffer，然后由root进行operation类型的操作。

看看具体怎么使用这个函数。

新建文件reduce.cpp：

```cpp
#include <iostream>
#include <sstream>
#include <mpi.h>

void PrintArray(int* array, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void ReduceAverage(int current_rank, int world_size)
{
    int root_rank = 0;
    int global_data = 0;

    // Assign data for each rank
    int local_data = (current_rank + 1) * 100;
    std::cout << "Rank " << current_rank << ": Send " << local_data 
        << " to root rank for reduce average" << std::endl;

    MPI_Reduce(&local_data, &global_data, 1, MPI_INT,
        MPI_SUM, root_rank, MPI_COMM_WORLD);

    if (current_rank == root_rank)
    {
        // Show the reduce result at root rank
        std::cout << "Rank " << current_rank << ": Reduce sum " 
            << global_data << std::endl;
        std::cout << "Rank " << current_rank << ": Reduce average " 
            << (float)global_data/world_size << std::endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int current_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    ReduceAverage(current_rank, world_size);

    MPI_Finalize();
}
```

编译并执行该文件，可以得到如下结果：

```shell
mpic++ ./reduce.cpp -o reduce
mpirun -np 4 reduce
Rank 1: Send 200 to root rank for reduce average
Rank 2: Send 300 to root rank for reduce average
Rank 3: Send 400 to root rank for reduce average
Rank 0: Send 100 to root rank for reduce average
Rank 0: Reduce sum 1000
Rank 0: Reduce average 250
```

该代码执行的操作如下图所示：

![mpi_reduce.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_reduce.png)

每个rank将各自的数据发送到rank 0，然后rank 0进行求和操作 (代码里面还带了求平均值的操作，MPI本身不带求均值的操作符)。

Reduce和Gather看起来有点像，都是各个rank把数据发到rank root。主要的区别在于root收到后怎么处理，Gather做的是把各个数据按照rank次序，拼接成更长的数据；而Reduce做的是对各个数据进行数学操作，得到另一个数据。

接下来，看看一个比较复杂的操作。

#### Reduce-Scatter

顾名思义，这个操作是Reduce和Scatter的组合。

通过这个操作，可以把通信域内的所有数据先做一次归约reduce，再把归约结果按照rank次序，分散scatter到各个rank去。

对应的函数为：

int MPI_Reduce_scatter(const void* send_buffer, void* receive_buffer, const int* counts, MPI_Datatype datatype, MPI_Op operation, MPI_Comm communicator)：它的意思是，将communicator通信域里面每个进程的send_buffer数据做归约，然后再把归约后的数据，按照进程的rank次序，依次发到每个进程的receive_buffer里面。counts是个数组，用来约定每个进程分到几个数据。

在下面的例子，我们在每个进程里新建一个数组，数组的长度和world_size一致。然后执行MPI_Reduce_scatter，就可以将这些数组先归约再分散了。最后，每个进程会收到一个数字。如果需要不同的进程收到不一样个数的数字，可以通过调整scatter_cnts实现。

新建如下的reduce_scatter.cpp文件：

```cpp
#include <iostream>
#include <sstream>
#include <mpi.h>

void PrintArray(int* array, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void RecuceScatter(int current_rank, int world_size)
{
    // Each rank has an array whose size is world_size
    // Then reduce the array, and scatter the results to each rank

    // Create array for each rank
    int buffer[world_size];
    for (int i = 0; i < world_size; i++)
    {
        buffer[i] = current_rank + i + 1;
    }
    std::cout << "Rank " << current_rank << ": Original array is: " << std::endl;
    PrintArray(buffer, world_size);

    // Create the count to scatter to each rank
    int scatter_cnts[world_size];
    // Each rank will receive one element
    for (int i = 0; i < world_size; i++)
    {
        scatter_cnts[i] = 1;
    }

    int results = 0;
    MPI_Reduce_scatter(buffer, &results, scatter_cnts, 
        MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    std::cout << "Rank " << current_rank << ": Receive data: " 
        << results << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int current_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    RecuceScatter(current_rank, world_size);

    MPI_Finalize();
}
```

编译并执行该文件，可以得到如下结果：

```shell
mpic++ ./reduce_scatter.cpp -o reduce_scatter

mpirun -np 4 reduce_scatter

Rank 0: Original array is:
1 2 3 4
Rank 1: Original array is:
2 3 4 5
Rank 2: Original array is:
3 4 5 6
Rank 3: Original array is:
4 5 6 7
Rank 0: Receive data: 10
Rank 1: Receive data: 14
Rank 2: Receive data: 18
Rank 3: Receive data: 22
```

具体过程如下图所示，不同进程里，相同颜色的数字进行加法操作(Reduce)。然后每个进程收到一种颜色的结果(Scatter)。

![mpi_reduce_scatter.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_reduce_scatter.png)

下一个操作是All Gather。

#### All Gather

这个操作和之前的Gather操作之间，唯一的区别在于，在完成Gather操作之后，会把收集到的结果复制到所有的ranks。可以用Gather + Broadcast来理解这个操作。

对应的函数为：

int MPI_Allgather(const void* buffer_send, int count_send, MPI_Datatype datatype_send, void* buffer_recv, int count_recv, MPI_Datatype datatype_recv, MPI_Comm communicator)

对比一下之前的Gather函数：

int MPI_Gather(const void* buffer_send, int count_send, MPI_Datatype datatype_send, void* buffer_recv, int count_recv, MPI_Datatype datatype_recv, int root, MPI_Comm communicator)

可以看到All Gather少了个root参数，其它的全部一样。功能类似，在此不赘述，直接看例子。

新建一个allgather.cpp文件：

```cpp
#include <iostream>
#include <sstream>
#include <mpi.h>

void PrintArray(int* array, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void AllGatherNumbers(int current_rank, int world_size)
{
    int root_rank = 0;
    int data_sent = (current_rank + 1) * 100;
    std::cout << "Rank " << current_rank << ": Send data " << data_sent 
        << " for allgather" << std::endl;

    // Alloc buffer to gather data
    int buffer[world_size];
    MPI_Allgather(&data_sent, 1, MPI_INT,
        buffer, 1, MPI_INT,
        MPI_COMM_WORLD);

    std::cout << "After allgather, the data in rank " << current_rank << std::endl;
    PrintArray(buffer, world_size);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int current_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    AllGatherNumbers(current_rank, world_size);
    //AllGatherString(current_rank, world_size);

    MPI_Finalize();
}
```

编译并执行该文件，可以得到如下结果：

```shell
mpic++ ./allgather.cpp -o allgather

mpirun -np 4 allgather

Rank 0: Send data 100 for allgather
Rank 1: Send data 200 for allgather
Rank 3: Send data 400 for allgather
Rank 2: Send data 300 for allgather
After allgather, the data in rank 0
100 200 300 400
After allgather, the data in rank 3
100 200 300 400
After allgather, the data in rank 1
100 200 300 400
After allgather, the data in rank 2
100 200 300 400
```



具体过程如下图所示：



![mpi_allgather.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_allgather.png)



胜利在望，只剩下两个通信原语了。

我们将All Reduce放到最后，先讲All to All操作。

#### All to All

All to All是通信域内所有的进程全交换。执行的操作可以用线性代数里面学的矩阵转置来理解。在大模型的分布式训练中，专家并行MoE就使用了All to All的操作。

对应的函数为：

int MPI_Alltoall(const void* buffer_send, int count_send, MPI_Datatype datatype_send, void* buffer_recv, int count_recv, MPI_Datatype datatype_recv, MPI_Comm communicator)：它的意思是，communicator通信域内的每个进程，都把自己的buffer_send发送到其它所有的进程，同时，从其它的所有进程接收数据并存放到buffer_recv里面。

看一个具体的例子来加深理解。

新建alltoall.cpp文件：

```cpp
#include <iostream>
#include <sstream>
#include <mpi.h>

void PrintArray(int* array, int len)
{
    for (int i = 0; i < len; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void All2AllTranpose(int current_rank, int world_size)
{
    // Initialize send buffer for each rank
    int send_buffer[world_size];
    for (int i = 0; i < world_size; i++) 
    {
        // Process rank sends data like: 0 sends {0,1,2,3}, 1 sends {10,11,12,13}, etc.
        send_buffer[i] = current_rank * 10 + i;
    }
    std::cout << "Rank " << current_rank << ": Send data:" << std::endl;
    PrintArray(send_buffer, world_size);

    // Buffer to receive data from all other ranks
    int recv_buffer[world_size];
    // All-to-all communication
    MPI_Alltoall(send_buffer, 1, MPI_INT, recv_buffer, 1, MPI_INT, MPI_COMM_WORLD);

    // Print the received data
    std::cout << "Rank " << current_rank << ": Recv data:" << std::endl;
    PrintArray(recv_buffer, world_size);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int current_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    All2AllTranpose(current_rank, world_size);

    MPI_Finalize();
}

```



编译并执行该文件，可以得到如下结果：

```shell
mpic++ ./alltoall.cpp -o alltoall

mpirun -np 4 alltoall

Rank 0: Send data:
0 1 2 3
Rank 1: Send data:
10 11 12 13
Rank 2: Send data:
20 21 22 23
Rank 3: Send data:
30 31 32 33
Rank 0: Recv data:
0 10 20 30
Rank 1: Recv data:
1 11 21 31
Rank 2: Recv data:
2 12 22 32
Rank 3: Recv data:
3 13 23 33
```



如果将每个进程的原始数组看作矩阵不同的行、数组里的每个元素看作不同的列的话，可以看到，经过All to All之后，矩阵的行与列发生了交换，即执行了数学里面的矩阵转置。

具体过程如下图所示：



![mpi_alltoall.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_alltoall.png)



最后我们来看看All Reduce操作。

#### All Reduce

这个应该是目前变种最多的一个集合通信原语了。在大模型的分布式训练中，这个操作广泛应用于数据并行、张量并行。为了提升All Reduce的效率，各大公司、高校提出了很多不同的做法。我们这里只介绍最基础的原理。

复习一下前面讲的Reduce归约操作，它是把通信域内所有进程的数据，都发到指定root rank上面进行计算，比如求和、求最大值、求最小值、求乘积等等。

All Reduce也是做了类似的事情，可以理解为，执行完Reduce之后，root rank把结果广播到所有的进程；或者这样理解，依次指定通信域内的每个rank作为root，然后分别执行一次Reduce操作。

对应的函数为：

int MPI_Allreduce(const void* send_buffer, void* receive_buffer, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm communicator)

对比一下之前Reduce函数：

int MPI_Reduce(const void* send_buffer, void* receive_buffer, int count, MPI_Datatype datatype, MPI_Op operation, int root, MPI_Comm communicator)

可以看到，All Reduce少了个root参数，它不用指定最后由哪个rank来保存归约结果，而是每个rank都会得到归约结果。

我们通过一个例子来看看怎么用。

新建allreduce.cpp文件：

```cpp
#include <iostream>
#include <sstream>
#include <mpi.h>

void AllReduceAverage(int current_rank, int world_size)
{
    int root_rank = 0;
    int global_data = 0;

    // Assign data for each rank
    int local_data = (current_rank + 1) * 100;
    std::cout << "Rank " << current_rank << ": Send " << local_data << " for allreduce average" << std::endl;

    MPI_Allreduce(&local_data, &global_data, 1, MPI_INT,
        MPI_SUM, MPI_COMM_WORLD);

    // Show the reduce result at each rank
    std::cout << "Rank " << current_rank << ": Reduce sum " << global_data << std::endl;
    std::cout << "Rank " << current_rank << ": Reduce average " << (float)global_data/world_size << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int current_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

    AllReduceAverage(current_rank, world_size);

    MPI_Finalize();
}

```



编译并执行该文件，可以得到如下结果：

```shell
mpic++ ./allreduce.cpp -o allreduce

mpirun -np 4 allreduce

Rank 0: Send 100 for allreduce average
Rank 3: Send 400 for allreduce average
Rank 2: Send 300 for allreduce average
Rank 1: Send 200 for allreduce average
Rank 2: Reduce sum 1000
Rank 2: Reduce average 250
Rank 3: Reduce sum 1000
Rank 3: Reduce average 250
Rank 0: Reduce sum 1000
Rank 0: Reduce average 250
Rank 1: Reduce sum 1000
Rank 1: Reduce average 250
```

具体过程如下图所示：



![mpi_allreduce.png](https://github.com/yuchuanwang/docs/blob/main/Assets/mpi_allreduce.png)



通过这个操作，每个rank都得到了所有ranks规约后的累积和、平均值。



----

至此，MPI的编程框架，以及集合通信的10个原语：MPI_Send、MPI_Recv、MPI_Bcast、MPI_Gather、MPI_Scatter、MPI_Reduce、MPI_Reduce_scatter、MPI_Allgather、MPI_Alltoall、MPI_Allreduce都讲完了。

洋洋洒洒写了这么长的文字和示意图，成功的把自己的知识梳理了一遍，也希望对读者有所帮助。




