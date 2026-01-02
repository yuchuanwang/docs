## 从零实现一个集合通信库：Mini CCL 设计与实现

### 1 前言

随着AI分布式训练和推理的需求暴涨，集合通信的应用也随之越来越多了。

在NVIDIA开源了NCCL之后，各芯片公司、互联网公司也推出了各种各样以自己公司名字命名的xCCL。比如华为的HCCL、腾讯的TCCL、阿里的ACCL、百度的BCCL、微软的MSCCL…… 加上众多芯片创业公司的通信库，26个字母很快就不够用了(或者已经不够用了 :D)

本文不是要重新开发一个集合通信库，而是抛开特定的芯片结构和功能、工程优化细节，尝试开发一个最基本的版本，从而加深自己对集合通信的理解。



### 2 整体架构设计

##### 2.1 双平面架构

Mini CCL 采用经典的**控制面 + 数据面**分离设计：

```
─────────────────────────────────────────────────────────────────────────
                                控制面
                        初始化阶段：获得全局节点信息
            Bootstrap：使用 TCP 进行节点发现、信息交换、集群拓扑构建

─────────────────────────────────────────────────────────────────────────
                                数据面
                       运行阶段：根据拓扑进行数据传输
                             Application
					              |
                     Communicator：统一的集合通信 API
					              |
                 Topology：TopologyRing或者TopologyFullMesh
					              |
                  Transport：TransportTCP或者TransportRdma

─────────────────────────────────────────────────────────────────────────
```



##### 2.2 核心组件

| 组件           | 职责                             | 关键文件                                     |
| ------------ | ------------------------------ | ---------------------------------------- |
| Bootstrap    | 控制面：节点发现、信息交换（仅初始化阶段）          | bootstrap.h/cpp                          |
| Communicator | 集合通信入口，协调 Topology 和 Transport | communicator.h/cpp                       |
| Topology     | 定义节点连接关系，实现具体的集合通信算法。          | topology_ring.h/cpp, topology_fullmesh.h/cpp |
| Transport    | 数据面：实现可靠传输                     | transport_tcp.h/cpp, transport_rdma.h/cpp    |



目前只支持了Full Mesh和Ring两种拓扑。如果需要支持更多的拓扑结构，只需要对应的实现一个具体的topology子类即可。



##### 2.3 初始化流程

```
──────────────────────────────────────────────────────────────────────────
                        Communicator::Init() 流程

  1. 创建 Transport 监听端口
  transport->Listen(0)  // 指定端口或者系统分配端口
  data_port = transport->GetListenPort()

  2. Bootstrap 阶段（控制面）
  Bootstrap::Run(config, data_port, all_nodes)
  - Rank 0: 收集所有节点的 IP:Port；广播给所有节点
  - 其他 Rank: 发送自己的信息；接收完整节点列表

  3. 根据 Topology 建立数据面连接
  for each peer in topology->GetNeighbors(rank):
      if ShouldConnect(rank, peer): // 主动连接
          transport->Connect(peer_ip, peer_port)
      else: // 被动接受
          transport->Accept()

──────────────────────────────────────────────────────────────────────────
```



### 3 主要实现

##### 3.1 Bootstrap：分布式系统的起点

在分布式系统中，各节点启动时需要相互发现。Mini CCL 采用了 **Master-Worker** 模式：

```
                    ┌─────────────────┐
                    │  Rank 0 (Master) │
                    │   监听端口 12321  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
  ┌────────────┐      ┌────────────┐      ┌────────────┐
  │   Rank 1   │      │   Rank 2   │      │   Rank 3   │
  │  (Worker)  │      │  (Worker)  │      │  (Worker)  │
  │ IP:Port A  │      │ IP:Port B  │      │ IP:Port C  │
  └────────────┘      └────────────┘      └────────────┘
```

**Bootstrap 协议时序：**

```
  Rank 0 (Master)              Rank 1               Rank 2               Rank 3
       │                          │                    │                    │
       │◄─────── TCP Connect ─────│                    │                    │
       │◄────────────────────── TCP Connect ───────────│                    │
       │◄────────────────────────────────── TCP Connect ─────────────────── │
       │                          │                    │                    │
       │◄──── NodeInfo{1,A} ──────│                    │                    │
       │◄──────────── NodeInfo{2,B} ───────────────────│                    │
       │◄──────────────────── NodeInfo{3,C} ─────────────────────────────── │
       │                          │                    │                    │
       │───── AllNodes[0,1,2,3] ──►                    │                    │
       │──────────── AllNodes[0,1,2,3] ────────────────►                    │
       │──────────────────── AllNodes[0,1,2,3] ────────────────────────────►
       │                          │                    │                    │
```

通过这个过程，所有节点都获取了完整的集群拓扑。



##### 3.2 Transport 抽象层

Transport 层提供统一的点对点通信接口，屏蔽 TCP/RDMA 的差异：

```cpp
class Transport {
public:
    // 连接管理
    virtual bool Listen(uint16_t port) = 0; // 监听端口 (0 = 系统分配)
    virtual uint16_t GetListenPort() const = 0; // 获取实际监听端口
    virtual bool Accept() = 0; // 接受连接
    virtual bool Connect(const std::string& addr, uint16_t port) = 0;

    // 数据传输
    virtual bool Send(const void* data, size_t size) = 0;
    virtual bool Recv(void* data, size_t size) = 0;

    // 状态管理
    virtual void Close() = 0;
    virtual bool IsConnected() const = 0;
};
```



##### 3.3 TransportTCP 实现

TCP 实现相对简单，使用标准 Socket API：

```cpp
bool TransportTCP::Send(const void* data, size_t size) 
{
    const char* ptr = static_cast<const char*>(data);
    size_t remaining = size;

    while (remaining > 0) 
    {
        ssize_t sent = ::send(sock_fd, ptr, remaining, 0);
        if (sent <= 0) 
        {
            if (errno == EINTR) continue;
            return false;
        }
        ptr += sent;
        remaining -= sent;
    }
    return true;
}

bool TransportTCP::Recv(void* data, size_t size)
{
    char* ptr = static_cast<char*>(data);
    size_t remaining = size;

    while (remaining > 0)
    {
        ssize_t received = ::recv(sock_fd, ptr, remaining, 0);
        if (received <= 0) 
        {
            if (errno == EINTR) continue;
            return false;
        }
        ptr += received;
        remaining -= received;
    }
    return true;
}
```



##### 3.4 TransportRdma 实现

RDMA 实现复杂得多，涉及多个 libibverbs 概念：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RDMA 核心概念                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐                                                │
│  │    ibv_context      │  RDMA 设备上下文，通过 ibv_open_device() 获取    │
│  └──────────┬──────────┘                                                │
│             │                                                           │
│             ▼                                                           │
│  ┌─────────────────────┐                                                │
│  │  Protection Domain  │  保护域，隔离不同应用的 RDMA 资源                 │
│  │        (PD)         │  通过 ibv_alloc_pd() 创建                       │
│  └──────────┬──────────┘                                                │
│             │                                                           │
│      ┌──────┴──────┐                                                    │
│      ▼             ▼                                                    │
│  ┌────────┐   ┌────────┐                                                │
│  │ Memory │   │ Queue  │                                                │
│  │ Region │   │  Pair  │                                                │
│  │  (MR)  │   │  (QP)  │                                                │
│  └────────┘   └───┬────┘                                                │
│                   │                                                     │
│           ┌───────┴───────┐                                             │
│           ▼               ▼                                             │
│      ┌─────────┐    ┌─────────┐                                         │
│      │  Send   │    │ Receive │                                         │
│      │  Queue  │    │  Queue  │                                         │
│      │  (SQ)   │    │   (RQ)  │                                         │
│      └────┬────┘    └────┬────┘                                         │
│           │              │                                              │
│           └──────┬───────┘                                              │
│                  ▼                                                      │
│           ┌─────────────┐                                               │
│           │     CQ      │  Completion Queue                             │
│           │             │  发送/接收完成后产生 Work Completion            │
│           └─────────────┘                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

具体的代码挺多的，这里不展开了，请查看TransportRdma类的实现。



##### 3.5 RDMA Buffer Pool 机制

RDMA 要求数据缓冲区必须注册为 Memory Region (MR)。由于 MR 注册涉及页表锁定，开销较大，因此采用 Buffer Pool 预分配策略：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Buffer Pool 结构                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  buffers[0]  ┌────────────────────────────────────────┐                 │
│              │  data: 16 MB buffer                    │  <── ibv_mr     │
│              │  mr: registered memory region          │                 │
│              │  id: 0, in_use: false                  │                 │
│              └────────────────────────────────────────┘                 │
│                                                                         │
│  buffers[1]  ┌────────────────────────────────────────┐                 │
│              │  data: 16 MB buffer                    │  <── ibv_mr     │
│              │  mr: registered memory region          │                 │
│              │  id: 1, in_use: true  <── 正在使用      │                 │
│              └────────────────────────────────────────┘                 │
│                                                                         │
│     ...      (默认 16 个 buffer，每个 16 MB)                              │
│                                                                         │
│  buffers[15] ┌────────────────────────────────────────┐                 │
│              │  data: 16 MB buffer                    │  <── ibv_mr     │
│              │  mr: registered memory region          │                 │
│              │  id: 15, in_use: false                 │                 │
│              └────────────────────────────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```



##### 3.6 Zero Copy 模式

对于大数据传输，Buffer Pool 模式的 memcpy 开销不可忽视。

Zero Copy 模式直接注册用户缓冲区为 MR，以消除memcpy、提升性能：

```cpp
bool TransportRdma::SendZeroCopyInternal(const void* data, size_t size)
{
    // 1. 构建size header的MR
    uint64_t net_size = htobe64(size);
    AutoMR size_mr(rdma_res->pd, &net_size, sizeof(uint64_t));

    // 2. 直接注册用户缓冲区为 MR
    AutoMR data_mr(rdma_res->pd, const_cast<void*>(data), size);

    // 3. 构造双 SGE 的 Send WR
    struct ibv_sge sge[2];
    sge[0].addr = reinterpret_cast<uint64_t>(&net_size);
    sge[0].length = sizeof(uint64_t);
    sge[0].lkey = size_mr.get()->lkey;
    sge[1].addr = reinterpret_cast<uint64_t>(data);
    sge[1].length = size;
    sge[1].lkey = data_mr.get()->lkey;

    struct ibv_send_wr wr =
    {
        .sg_list = sges,
        .num_sge = 2,
        .opcode = IBV_WR_SEND,
        .send_flags = IBV_SEND_SIGNALED
    };

    ... ...

    ibv_post_send(qp, &wr, &bad_wr);
    PollCompletion(nullptr);

    // MR 自动注销（RAII）
    return true;
}
```



**Buffer Pool vs Zero Copy 对比：**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Buffer Pool 模式                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  用户数据 ──memcpy──> Buffer[i] ──RDMA Send──> 远端 Buffer ──memcpy──> 用户│
│                         |                           |                   │
│                    (预注册 MR)                  (预注册 MR)               │
│                                                                         │
│  优点: MR 注册开销分摊，小数据延迟低                                         │
│  缺点: 需要两次 memcpy，大数据带宽受限                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      Zero Copy 模式                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  用户数据 ──注册MR──> RDMA Send ──> 远端 ──注销MR──> 用户数据                │
│      |                                           |                      │
│  (动态注册)                                   (动态注销)                   │
│                                                                         │
│  优点: 无 memcpy，大数据带宽高                                             │
│  缺点: 每次传输都要注册/注销 MR，小数据延迟高                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

根据我在个人电脑上，使用Soft RoCE的测试结果：基本上数据量在1 MB以上的时候，Zero Copy的性能就超过了Buffer Pool了。



### 4 支持的集合通信操作

Mini CCL 实现了完整的 MPI 风格集合通信原语：

| 操作                | 描述                         | 通信模式 | 数据流向               |
| ----------------- | -------------------------- | ---- | ------------------ |
| **Broadcast**     | Root 广播数据到所有节点             | 一对多  | Root -> All        |
| **Scatter**       | Root 分发不同数据到各节点            | 一对多  | Root -> All (不同数据) |
| **Gather**        | 各节点数据汇聚到 Root              | 多对一  | All -> Root        |
| **Reduce**        | 规约到 Root (SUM/MAX/MIN/AVG) | 多对一  | All -> Root (规约)   |
| **AllGather**     | 所有节点收集所有数据                 | 多对多  | All <-> All        |
| **ReduceScatter** | 规约后分散到各节点                  | 多对多  | All <-> All        |
| **AllReduce**     | 所有节点持有规约结果                 | 多对多  | All <-> All (最常用)  |
| **AllToAll**      | 全交换                        | 多对多  | All <-> All (置换)   |
| **Barrier**       | 同步屏障                       | 多对多  | 同步                 |

---

具体的通信原语介绍，可以参考之前的文章：

https://github.com/yuchuanwang/docs/blob/main/Network/MPI_HPC.md



### 5 快速开始

##### 5.1 环境准备

```bash
# Ubuntu/Debian
sudo apt install -y libibverbs-dev ibverbs-utils rdma-core libfmt-dev mpich

# 如果没有物理 RDMA 网卡，使用软件模拟
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
```

##### 5.2 编译

```bash
git clone https://github.com/yuchuanwang/mini_ccl.git
cd mini_ccl
mkdir build && cd build
cmake ..
make -j4
```

##### 5.3 测试

使用 MPI 启动：

```bash
# TCP + Ring (default)
mpirun -np 4 ./test_ccl
# RDMA + Ring
mpirun -np 4 ./test_ccl rdma
# TCP + FullMesh
mpirun -np 4 ./test_ccl tcp fullmesh
```

或者手动多终端启动：

```bash
# Terminal 1
./tests/test_ccl 0 4 rdma ring

# Terminal 2-4
./tests/test_ccl 1 4 rdma ring
./tests/test_ccl 2 4 rdma ring
./tests/test_ccl 3 4 rdma ring
```



### 6 其他

目前的代码里面，还有一些TODO没完成，包括如何让Python调用这个通信库、如何支持更多的拓扑、如何优化RDMA传输等等，以后有空再逐渐补充……

这个代码陆陆续续写了不少时间了，也尝试让AI来辅助我实现具体功能、然后我来review、修改。切身体会到AI coding的发展速度飞快，从一年前的低质量输出，到现在已经可以代替很多的编码工作了。

AI对很多行业，包括软件开发行业和相关的就业，将会带来巨大的冲击。我们准备好了吗……






