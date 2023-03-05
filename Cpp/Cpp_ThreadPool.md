## C++ 线程池

在做C++开发的时候，线程池是一个经常需要使用的技术。通过它，可以充分利用多线程带来的好处，同时避免频繁创建和、销毁线程所带来的消耗。

借用一下[Wiki](https://zh.wikipedia.org/zh-cn/%E7%BA%BF%E7%A8%8B%E6%B1%A0)的定义：

> **线程池**（英语：thread pool）：一种[线程](https://zh.wikipedia.org/wiki/%E7%BA%BF%E7%A8%8B "线程")使用模式。线程过多会带来调度开销，进而影响缓存局部性和整体性能。而线程池维护着多个线程，等待着监督管理者分配可并发执行的任务。这避免了在处理短时间任务时创建与销毁线程的代价。线程池不仅能够保证内核的充分利用，还能防止过分调度。可用线程数量应该取决于可用的并发处理器、处理器内核、内存、网络sockets等的数量。 例如，对于计算密集型任务，线程数一般取cpu数量+2比较合适，线程数过多会导致额外的线程切换开销。



但是，在标准C++库里，并未提供线程池的实现，于是，这世上就有了n种实现方式。

更糟糕的是，我又贡献了第n+1种。

期待以后的C++ 2x能统一实现一个版本。



Linux老大Linus Torvalds 说："Talk is cheap. Show me the code." 

所以，先贴代码为敬。

```cpp
// ThreadPool with C++ std::thread
// Author: Yuchuan Wang
// yuchuan.wang@gmail.com
// 

#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <queue>

// Function will be running inside thread pool
using ThreadTask = std::function<void()>;

class ThreadPool
{
public:
    // If threads_num is 0, it will use the same number of CPU cores
    // If tasks_num is -1, the number of tasks will be unlimited
    ThreadPool(int threads_num = 0, int tasks_num = -1)
    {
        if(threads_num == 0)
        {
            max_threads = std::thread::hardware_concurrency();
        }
        else
        {
            max_threads = threads_num;
        }
        max_tasks = tasks_num;
        is_running = false;
    }

    ~ThreadPool()
    {
        WaitForStop();
    }

    // Add task to queue
    bool AddTask(ThreadTask task)
    {
        // Scope for lock
        {
            std::unique_lock<std::mutex> lock(tasks_guard);
            if(max_tasks == -1)
            {
                // Unlimited
                tasks.push(task);
            }
            else
            {
                if(tasks.size() >= max_tasks)
                {
                    return false;
                }
                else
                {
                    tasks.push(task);
                }
            }
        }

        // Notify thread
        tasks_event.notify_one();

        return true; 
    }

    // Start threads
    bool Start()
    {
        if(is_running)
        {
            // Running already
            return false;
        }

        is_running = true;
        if(threads.empty())
        {
            CreateThreads();
        }

        return true;
    }

    void WaitForStop()
    {
        if(!is_running)
        {
            // I am not running
            return;
        }

        is_running = false; 
        tasks_event.notify_all();
        for(auto &t : threads)
        {
            // Wait for all threads to exit
            t.join();
        }
        threads.clear();
    }

private:
    void CreateThreads()
    {
        for(int i = 0; i < max_threads; i++)
        {
            threads.push_back(std::thread(&ThreadPool::ThreadRoutine, this));
        }
    }

    // Thread worker function
    // Take task from queue, and run it
    static void ThreadRoutine(ThreadPool* ptr)
    {
        if(ptr == nullptr)
        {
            return; 
        }

        while(ptr->is_running || !ptr->tasks.empty())
        {
            ThreadTask task; 
            // Scope for lock
            {
                // Get task to run
                std::unique_lock<std::mutex> lock(ptr->tasks_guard);
                while(ptr->tasks.empty())
                {
                    // Wait until task is ready
                    ptr->tasks_event.wait(lock);
                }
                
                // OK, now there is a task ready to run
                task = ptr->tasks.front();
                ptr->tasks.pop();
            }
            // Run it
            task();
        }
    }

private:
    // Max threads allowed
    int max_threads;
    // Max tasks inside queue
    int max_tasks; 
    // Vector of threads
    std::vector<std::thread> threads; 
    // Queue of tasks
    std::queue<ThreadTask> tasks; 
    // Flag of runnin status
    bool is_running;
    // Mutex to protect the tasks queue
    std::mutex tasks_guard; 
    // Condition of tasks event
    std::condition_variable tasks_event; 
};

```



大概解释几个点：

1. using ThreadTask = std::function<void()>;

类似于以前的typedef用法，定义了一个ThreadTask的类型，代表C++的函数对象。它会被放进线程池的任务队列里面。后续的各个线程，会从任务队列里面取出具体的任务(函数对象)，然后执行这个任务。

2. ThreadPool(int threads_num = 0, int tasks_num = -1)

初始化线程池。默认情况下，创建的线程数跟你系统的CPU核数一样，任务队列里的任务个数无限。

3. bool AddTask(ThreadTask task)

把需要线程执行的任务，加入到任务队列里面。

4. bool Start()

创建threads_num个线程，并开始运行。

5. void WaitForStop()

等待所有的线程完成当前正在处理的任务后，退出。



具体的使用方法，可以参考如下代码：

```cpp
#include "ThreadPool.h"

int product_sell = 0;
void ProductCounter(std::mutex* task_protect)
{
    //std::this_thread::sleep_for(std::chrono::seconds(1));
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    std::lock_guard<std::mutex> lock(*task_protect);
    std::cout <<"How many products sell: " << product_sell++ << std::endl;
}

int main()
{
    std::mutex protect_task;
    ThreadPool pool(0, -1);
    for(int i = 0; i < 100; i++)
    {
        pool.AddTask(std::bind(ProductCounter, &protect_task));
    }
    pool.Start();
    // Do more stuff...
    for(int i = 0; i < 50; i++)
    {
        pool.AddTask(std::bind(ProductCounter, &protect_task));
    }
    pool.WaitForStop();
    return 0;
}

```



在这个小例子中，创建了一个线程池。先添加了100个任务，然后启动线程池。接着又添加了50个任务。线程池会以同样的线程个数(在我的电脑，有8个core，所以默认是8个线程)，依次把任务完成，最后退出。

输出是这样的：

```shell
How many products sell: 0
How many products sell: 1
How many products sell: 2
How many products sell: 3
How many products sell: 4
How many products sell: 5
How many products sell: 6
How many products sell: 7
How many products sell: 8
How many products sell: 9
How many products sell: 10
How many products sell: 11
How many products sell: 12
How many products sell: 13
How many products sell: 14
How many products sell: 15
How many products sell: 16
How many products sell: 17
How many products sell: 18
How many products sell: 19
How many products sell: 20
How many products sell: 21
How many products sell: 22
How many products sell: 23
How many products sell: 24
How many products sell: 25
How many products sell: 26
How many products sell: 27
How many products sell: 28
How many products sell: 29
How many products sell: 30
How many products sell: 31
How many products sell: 32
How many products sell: 33
How many products sell: 34
How many products sell: 35
How many products sell: 36
How many products sell: 37
How many products sell: 38
How many products sell: 39
How many products sell: 40
How many products sell: 41
How many products sell: 42
How many products sell: 43
How many products sell: 44
How many products sell: 45
How many products sell: 46
How many products sell: 47
How many products sell: 48
How many products sell: 49
How many products sell: 50
How many products sell: 51
How many products sell: 52
How many products sell: 53
How many products sell: 54
How many products sell: 55
How many products sell: 56
How many products sell: 57
How many products sell: 58
How many products sell: 59
How many products sell: 60
How many products sell: 61
How many products sell: 62
How many products sell: 63
How many products sell: 64
How many products sell: 65
How many products sell: 66
How many products sell: 67
How many products sell: 68
How many products sell: 69
How many products sell: 70
How many products sell: 71
How many products sell: 72
How many products sell: 73
How many products sell: 74
How many products sell: 75
How many products sell: 76
How many products sell: 77
How many products sell: 78
How many products sell: 79
How many products sell: 80
How many products sell: 81
How many products sell: 82
How many products sell: 83
How many products sell: 84
How many products sell: 85
How many products sell: 86
How many products sell: 87
How many products sell: 88
How many products sell: 89
How many products sell: 90
How many products sell: 91
How many products sell: 92
How many products sell: 93
How many products sell: 94
How many products sell: 95
How many products sell: 96
How many products sell: 97
How many products sell: 98
How many products sell: 99
How many products sell: 100
How many products sell: 101
How many products sell: 102
How many products sell: 103
How many products sell: 104
How many products sell: 105
How many products sell: 106
How many products sell: 107
How many products sell: 108
How many products sell: 109
How many products sell: 110
How many products sell: 111
How many products sell: 112
How many products sell: 113
How many products sell: 114
How many products sell: 115
How many products sell: 116
How many products sell: 117
How many products sell: 118
How many products sell: 119
How many products sell: 120
How many products sell: 121
How many products sell: 122
How many products sell: 123
How many products sell: 124
How many products sell: 125
How many products sell: 126
How many products sell: 127
How many products sell: 128
How many products sell: 129
How many products sell: 130
How many products sell: 131
How many products sell: 132
How many products sell: 133
How many products sell: 134
How many products sell: 135
How many products sell: 136
How many products sell: 137
How many products sell: 138
How many products sell: 139
How many products sell: 140
How many products sell: 141
How many products sell: 142
How many products sell: 143
How many products sell: 144
How many products sell: 145
How many products sell: 146
How many products sell: 147
How many products sell: 148
How many products sell: 149
```

相关的代码，上传在[Github](https://github.com/yuchuanwang/ThreadPool)，欢迎使用并提出意见。




