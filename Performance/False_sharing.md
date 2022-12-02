## 性能优化 - 伪共享

#### 背景

上一篇关于[局部性的文章](./Locality_of_reference.md)，讲了因为CPU缓存、Cache Line、局部性而导致的性能差异。我们接着分析因为缓存、Cache Line和缓存一致性，在多线程并发编程中所带来的另一个问题：伪共享。

伪共享，False Sharing，没找到中文的标准翻译。参考一下英文版的[wiki]([False sharing - Wikipedia](https://en.wikipedia.org/wiki/False_sharing))：

> In computer science, false sharing is a performance-degrading usage pattern that can arise in systems with distributed, coherent caches at the size of the smallest resource block managed by the caching mechanism. When a system participant attempts to periodically access data that is not being altered by another party, but that data shares a cache block with data that is being altered, the caching protocol may force the first participant to reload the whole cache block despite a lack of logical necessity. The caching system is unaware of activity within this block and forces the first participant to bear the caching system overhead required by true shared access of a resource.
> 
> By far the most common usage of this term is in modern multiprocessor CPU caches, where memory is cached in lines of some small power of two word size (e.g., 64 aligned, contiguous bytes). If two processors operate on independent data in the same memory address region storable in a single line, the cache coherency mechanisms in the system may force the whole line across the bus or interconnect with every data write, forcing memory stalls in addition to wasting system bandwidth. In some cases, the elimination of false sharing can result in order-of-magnitude performance improvements. False sharing is an inherent artifact of automatically synchronized cache protocols and can also exist in environments such as distributed file systems or databases, but current prevalence is limited to RAM caches.

简单地说，CPU每次会读取一个Cache Line的数据(64字节)进入缓存。如果两个线程分别运行在Core0、Core1，这两个Core读取了同一块64字节的内存进入各自的L1缓存。Core0去修改前4个字节的内容；Core1去修改接下去的4个字节的内容。这两个看似不冲突的操作，因为CPU的[缓存一致性]([MESI协议 - 维基百科，自由的百科全书](https://zh.m.wikipedia.org/zh-hans/MESI%E5%8D%8F%E8%AE%AE))，会带来对性能的冲击。

#### 实验

我们可以通过两个简单的代码来进行对比。两个代码都是一样的目的，创建两个线程，分别对同一个结构体内的不同数据进行操作。线程A操作数据a，线程B操作数据b，看起来河水不犯井水。

**第一个C文件 - False_sharing_hit.c：**

```c
// False_sharing_hit.c

#include <stdio.h>
#include <pthread.h>

#define Loops 1000000000

struct
{
    int data_thread_a;
    int data_thread_b;
} data;

void* thread_a_routine()
{
    for (int i = 0; i < Loops; i++)
    {
        data.data_thread_a = 1;
    }
}

void* thread_b_routine()
{
    for (int i = 0; i < Loops; i++)
    {
        data.data_thread_b = 2;
    }
}

int main(int argc, char *argv[])
{
    pthread_t tid_a, tid_b;
    pthread_create(&tid_a, NULL, (void*)thread_a_routine, NULL);
    pthread_create(&tid_b, NULL, (void*)thread_b_routine, NULL);

    return 0;
}
```

**第二个C文件 - False_sharing_avoid.c：**

```c
// False_sharing_avoid.c

#include <stdio.h>
#include <pthread.h>

#define Loops 1000000000
#define CacheLine 64

struct
{
    int data_thread_a;
    // Add this to avoid false sharing
    char padding[CacheLine];
    int data_thread_b;
} data;


void* thread_a_routine()
{
    for (int i = 0; i < Loops; i++)
    {
        data.data_thread_a = 1;
    }
}

void* thread_b_routine()
{
    for (int i = 0; i < Loops; i++)
    {
        data.data_thread_b = 2;
    }
}

int main(int argc, char *argv[])
{
    pthread_t tid_a, tid_b;
    pthread_create(&tid_a, NULL, (void*)thread_a_routine, NULL);
    pthread_create(&tid_b, NULL, (void*)thread_b_routine, NULL);

    return 0;
}
```

分别编译这两个代码：

```shell
$ gcc False_sharing_hit.c -o False_sharing_hit
$ gcc False_sharing_avoid.c -o False_sharing_avoid
```

接着分别执行编译出来的程序：

```shell
$ time ./False_sharing_hit

real    0m0.006s
user    0m0.000s
sys    0m0.005s
```

```shell
$ time ./False_sharing_avoid

real    0m0.002s
user    0m0.002s
sys     0m0.000s
```

可以发现，第二个程序只花了第一个程序1/3的时间。

#### 结论

第一个代码触发了伪共享的问题。线程A和线程B分别运行在两个Core上，假设Core0运行线程A，Core1运行线程B。data_thread_a和data_thread_b紧紧挨着，会一起被读入两个Core的缓存。当Core0上的线程A修改了data_thread_a的数值之后，Core0的Cache Line变成了dirty状态。需要写回内存后，Core1上的线程B才能继续修改该Cache Line。反之亦然，当Core1的线程B修改完data_thread_b之后，Cache Line又变成dirty状态，需要写回内存后，Core0上的线程A才能修改data_thread_a。所以，为了保证缓存一致性，不管哪个线程发生了修改操作，都会触发缓存回写内存的操作。
像不像A、B两人住在一起，A阳性了，B作为密接被关起来；等到A、B都放出来后，B又阳性了，A作为密接又被关起来了？子子孙孙无穷尽也……

而第二个代码data_thread_a和data_thread_b之间加入了补齐的元素，使得data_thread_a和data_thread_b不会进入同一个Cache Line。每个Core上的Cache Line修改不需要立即写回内存。从而带来了3倍的性能收益。
承接上面的比喻，就是A、B不住在一起，不是时空伴随者，所以A阳性了，B不用被关…… 

所以在程序开发中，需要理解代码背后发生了什么，才能写出高效的代码。
