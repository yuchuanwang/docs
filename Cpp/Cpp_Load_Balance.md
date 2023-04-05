## 负载均衡算法的实现

> **负载平衡**（英语：load balancing）是一种[电子计算机](https://zh.wikipedia.org/wiki/%E7%94%B5%E5%AD%90%E8%AE%A1%E7%AE%97%E6%9C%BA "电子计算机")技术，用来在多个计算机（[计算机集群](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E9%9B%86%E7%BE%A4 "计算机集群")）、网络连接、CPU、磁盘驱动器或其他资源中分配负载，以达到优化资源使用、最大化吞吐率、最小化响应时间、同时避免过载的目的。 使用带有负载平衡的多个服务器组件，取代单一的组件，可以通过[冗余](https://zh.wikipedia.org/wiki/%E5%86%97%E4%BD%99 "冗余")提高可靠性。负载平衡服务通常是由专用软件和硬件来完成。 主要作用是将大量作业合理地分摊到多个操作单元上进行执行，用于解决互联网架构中的[高并发](https://zh.wikipedia.org/wiki/%E5%B9%B6%E5%8F%91%E6%80%A7 "并发性")和[高可用](https://zh.wikipedia.org/wiki/%E9%AB%98%E5%8F%AF%E7%94%A8%E6%80%A7 "高可用性")的问题。

这是[维基百科]([负载均衡 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E8%B4%9F%E8%BD%BD%E5%9D%87%E8%A1%A1))对负载均衡的定义。

按照我的理解，负载均衡主要就是为了两个目的：并行处理(A忙不过来，B一起上)、防止单点失败(A忙废了，B顶上)。分别对应了高并发、高可用的核心诉求。

负载均衡的算法有很多种，我尝试把我的理解表达出来。

陆游老先生曾经曰过：纸上得来终觉浅，绝知此事要躬行。所以，用C++把它们实现一下。

#### 1. 轮询 Round Robin

轮询，就是将请求逐个分发给后端的服务器，每个服务器都被平等对待。

具体实现，就是把所有的后端服务器放到一个数组里，并用一个变量来保存当前索引。每次过来一个新的请求，把当前索引前进一步，并对数组长度取模。

参考代码如下：

```cpp
// Load Balance with Round Robin

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

class LoadBalanceRoundRobin
{
public: 
    LoadBalanceRoundRobin()
    {
        current = -1;
    }

    ~LoadBalanceRoundRobin()
    {
    }

    bool AddServer(const std::string& srv)
    {
        servers.push_back(srv);
        return true;
    }

    // Simulate a new request
    bool NextRequest()
    {
        if(servers.empty())
        {
            std::cout << "Please add servers first. " << std::endl;
            return false; 
        }

        // Move to next server
        current = (current + 1) % servers.size();
        // Update stats
        stats[servers[current]]++;

        return true; 
    }

    void PrintStats() const
    {
        std::cout << "Server Hit stats with round robin: ";
        for(auto x : stats)
        {
            std::cout << std::endl;
            std::cout << x.first << ": " << x.second;
        }
        std::cout << std::endl;
    }

private:
    // The servers to be balanced
    std::vector<std::string> servers; 
    // Index of server hit
    int current;
    // Stats, key is the server, value is the hit count
    std::unordered_map<std::string, int> stats; 
};
```

往这个LB添加5个服务器，然后发起一百万次请求：

```cpp
void TestRoundRobin()
{
    LoadBalanceRoundRobin lb;
    lb.AddServer("192.168.1.10");
    lb.AddServer("192.168.1.11");
    lb.AddServer("192.168.1.12");
    lb.AddServer("192.168.1.13");
    lb.AddServer("192.168.1.14");

    for(int i = 0; i < 1000000; i++)
    {
        lb.NextRequest();
    }

    lb.PrintStats();
}
```

可以看到，这些请求被均匀的分发给了每个服务器：

```shell
Server Hit stats with round robin:
192.168.1.14: 200000
192.168.1.13: 200000
192.168.1.12: 200000
192.168.1.11: 200000
192.168.1.10: 200000
```

***<u>Note一下，本次内容重点在于试验负载均衡算法，不在于类的设计。所以，后面还会出现好几个相似的类，但我并没有去做任何的继承。实际工程应用的时候，需要考虑抽象、继承问题，以减少代码重复。</u>***

#### 2. 加权轮询 Weighted Round Robin

后端的服务器，有些强劲、有些比较弱。可以给强劲的服务器分配较大的权重，给它分发更多的请求。

具体实现跟前面的轮询很像。主要的差别在于，添加后端服务器时，根据权重的数值N，把对应的服务器添加N次到数组里面，这样使得该服务器被轮询到的次数比例，跟它的权重比例一样。

参考代码如下：

```cpp
// Load Balance with Weighted Round Robin

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <algorithm>

class LoadBalanceWeightedRoundRobin
{
public: 
    LoadBalanceWeightedRoundRobin()
    {
        current = -1;
    }

    ~LoadBalanceWeightedRoundRobin()
    {
    }

    bool AddServer(const std::string& srv, int weight)
    {
        if(weight < 1)
        {
            std::cout << "Weight should be equal or greater than 1." << std::endl;
            return false;
        }

        for(int i = 0; i < weight; i++)
        {
            // Add weight times to vector
            servers.push_back(srv);
        }

        // Shuffle the vector so that servers will be out of order
        std::random_shuffle(servers.begin(), servers.end());
        return true;
    }

    // Simulate a new request
    bool NextRequest()
    {
        if(servers.empty())
        {
            std::cout << "Please add servers first. " << std::endl;
            return false; 
        }

        // Move to next server
        current = (current + 1) % servers.size();
        // Update stats
        stats[servers[current]]++;

        return true; 
    }

    void PrintStats() const
    {
        std::cout << "Server Hit stats with weighted round robin: ";
        for(auto x : stats)
        {
            std::cout << std::endl;
            std::cout << x.first << ": " << x.second;
        }
        std::cout << std::endl;
    }

private:
    // The servers to be balanced
    std::vector<std::string> servers; 
    // Index of server hit
    int current;
    // Stats, key is the server, value is the hit count
    std::unordered_map<std::string, int> stats; 
};
```

往这个LB添加5个服务器，并给与不同的权重，然后发起一百万次请求：

```cpp
void TestWeightedRoundRobin()
{
    LoadBalanceWeightedRoundRobin lb;
    lb.AddServer("192.168.1.10", 1);
    lb.AddServer("192.168.1.11", 2);
    lb.AddServer("192.168.1.12", 3);
    lb.AddServer("192.168.1.13", 4);
    lb.AddServer("192.168.1.14", 10);

    for(int i = 0; i < 1000000; i++)
    {
        lb.NextRequest();
    }

    lb.PrintStats();
}
```

可以看到，每个服务器被分发的请求比例，跟它的权重比例是一样的：

```shell
Server Hit stats with weighted round robin: 
192.168.1.13: 200000
192.168.1.10: 50000
192.168.1.12: 150000
192.168.1.11: 100000
192.168.1.14: 500000
```

#### 3. 随机 Random

随机算法也很好理解，每次过来一个请求，随机分发给某一台后端服务器即可。随着请求量的增加，每个后端服务器的请求总数会趋向一致。

参考代码如下：

```cpp
// Load Balance with Random

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

class LoadBalanceRandom
{
public: 
    LoadBalanceRandom()
    {
    }

    ~LoadBalanceRandom()
    {
    }

    bool AddServer(const std::string& srv)
    {
        servers.push_back(srv);
        return true;
    }

    // Simulate a new request
    bool NextRequest()
    {
        if(servers.empty())
        {
            std::cout << "Please add servers first. " << std::endl;
            return false; 
        }

        // Pickup a random server
        int current = rand() % servers.size();
        // Update stats
        stats[servers[current]]++;

        return true; 
    }

    void PrintStats() const
    {
        std::cout << "Server Hit stats with random: ";
        for(auto x : stats)
        {
            std::cout << std::endl;
            std::cout << x.first << ": " << x.second;
        }
        std::cout << std::endl;
    }

private:
    // The servers to be balanced
    std::vector<std::string> servers; 
    // Stats, key is the server, value is the hit count
    std::unordered_map<std::string, int> stats; 
};
```

往这个LB添加5个服务器，然后发起一百万次请求：

```cpp
void TestRandom()
{
    LoadBalanceRandom lb;
    lb.AddServer("192.168.1.10");
    lb.AddServer("192.168.1.11");
    lb.AddServer("192.168.1.12");
    lb.AddServer("192.168.1.13");
    lb.AddServer("192.168.1.14");

    for(int i = 0; i < 1000000; i++)
    {
        lb.NextRequest();
    }

    lb.PrintStats();
}
```

可以看到，这些请求基本上被均匀的分发给了每个服务器：

```shell
Server Hit stats with random: 
192.168.1.13: 200448
192.168.1.10: 199584
192.168.1.14: 199799
192.168.1.11: 199495
192.168.1.12: 200674
```

#### 4. 加权随机 Weighted Random

加权随机与加权轮询的思路类似，给强劲的后端服务器更大的权重(出现次数更多)，让它更容易被随机选中。

具体代码跟随机相比，差别只是在添加后端服务器时，根据它的权重值，添加对应的次数而已：

```cpp
// Load Balance with Weight Random

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <algorithm>

class LoadBalanceWeightedRandom
{
public: 
    LoadBalanceWeightedRandom()
    {
    }

    ~LoadBalanceWeightedRandom()
    {
    }

    bool AddServer(const std::string& srv, int weight)
    {
        if(weight < 1)
        {
            std::cout << "Weight should be equal or greater than 1." << std::endl;
            return false;
        }

        for(int i = 0; i < weight; i++)
        {
            // Add weight times to vector
            servers.push_back(srv);
        }

        return true;
    }

    // Simulate a new request
    bool NextRequest()
    {
        if(servers.empty())
        {
            std::cout << "Please add servers first. " << std::endl;
            return false; 
        }

        // Pickup a random server
        int current = rand() % servers.size();
        // Update stats
        stats[servers[current]]++;

        return true; 
    }

    void PrintStats() const
    {
        std::cout << "Server Hit stats with weighted random: ";
        for(auto x : stats)
        {
            std::cout << std::endl;
            std::cout << x.first << ": " << x.second;
        }
        std::cout << std::endl;
    }

private:
    // The servers to be balanced
    std::vector<std::string> servers; 
    // Stats, key is the server, value is the hit count
    std::unordered_map<std::string, int> stats; 
};
```

往这个LB添加5个服务器，并给与不同的权重，然后发起一百万次请求：

```cpp
void TestWeightedRandom()
{
    LoadBalanceWeightedRandom lb;
    lb.AddServer("192.168.1.10", 1);
    lb.AddServer("192.168.1.11", 2);
    lb.AddServer("192.168.1.12", 3);
    lb.AddServer("192.168.1.13", 4);
    lb.AddServer("192.168.1.14", 10);

    for(int i = 0; i < 1000000; i++)
    {
        lb.NextRequest();
    }

    lb.PrintStats();
}
```

可以看到，每个服务器被分发的请求比例，跟它的权重比例是基本一样的：

```shell
Server Hit stats with weighted random: 
192.168.1.13: 200290
192.168.1.12: 149612
192.168.1.11: 100060
192.168.1.10: 50291
192.168.1.14: 499747
```

#### 5. 源地址哈希 Source IP Hash

根据客户端的IP地址，通过Hash算出个数值后，对后端服务器的总数取模，然后把请求分发给取模得到的服务器。

这个代码也很简单，就不实现了。

---

前面的这五种算法，在选择分发到哪个服务器时，都依赖于服务器的总数。

而服务器总会有挂掉的时候。一旦某个服务器挂了，意味着可用的服务器总数发生了变化(虽然前面的例子，都没有实现RemoveServer的接口)，那么被选中的服务器都会发生变化。

这就带来了不一致的问题、每个请求都要重新计算。所以，后面的一致性哈希算法应运而生了。

题外话，技术也好、算法也好，都是为了解决某些具体的问题、场景而被发明出来的。

**理解了问题，有助于更好的理解为什么会有这样的算法、解决方案。**

---

#### 6. 一致性哈希 Consistent Hashing

一致性哈希是现在用的比较广泛的算法，具体就不解释了，网上资料非常多。[维基百科]([一致哈希 - 维基百科，自由的百科全书](https://zh.wikipedia.org/wiki/%E4%B8%80%E8%87%B4%E5%93%88%E5%B8%8C))的描述也很清楚：

> 一致哈希将每个对象映射到圆环边上的一个点，系统再将可用的节点机器映射到圆环的不同位置。查找某个对象对应的机器时，需要用一致哈希算法计算得到对象对应圆环边上位置，沿着圆环边上查找直到遇到某个节点机器，这台机器即为对象应该保存的位置。 当删除一台节点机器时，这台机器上保存的所有对象都要移动到下一台机器。添加一台机器到圆环边上某个点时，这个点的下一台机器需要将这个节点前对应的对象移动到新机器上。 更改对象在节点机器上的分布可以通过调整节点机器的位置来实现。

参考代码如下：

```cpp
// Load Balance with Consistent Hashing

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <unordered_map>
#include <iostream>

class LoadBalanceConsistentHashing
{
public: 
    LoadBalanceConsistentHashing(int vNum = 32)
    {
        virtualNum = vNum; 
    }

    ~LoadBalanceConsistentHashing()
    {
    }

    bool AddServer(const std::string& srv)
    {
        servers.push_back(srv);

        // Insert virtual nodes for each real server
        for(int i = 0; i < virtualNum; i++)
        {
            // Compose name like: 192.168.1.10#1
            std::stringstream srvName;
            srvName << srv << "#" << i; 
            unsigned int hashKey = std::hash<std::string>{}(srvName.str());
            nodes.insert({hashKey, srv});
        }

        return true;
    }

    bool DeleteServer(const std::string& srv)
    {
        auto server = std::find(servers.begin(), servers.end(), srv);
        if(server == servers.end())
        {
            std::cout << "Invalid server to delete. " << std::endl;
            return false; 
        }

        // Delete from real servers
        servers.erase(server);

        // Delete virtual nodes for this real server
        for(int i = 0; i < virtualNum; i++)
        {
            // Compose name like: 192.168.1.10#1
            std::stringstream srvName;
            srvName << srv << "#" << i; 
            unsigned int hashKey = std::hash<std::string>{}(srvName.str());

            // Find and delete
            auto it = nodes.find(hashKey);
            if(it != nodes.end()) 
            {
                nodes.erase(it);
            }
        }

        return true; 
    }

    // Simulate a new request
    bool NextRequest()
    {
        if(servers.empty())
        {
            std::cout << "Please add servers first. " << std::endl;
            return false; 
        }

        // Find the node for this request
        int val = rand(); 
        unsigned int hashKey = std::hash<std::string>{}(std::to_string(val));
        auto node = nodes.lower_bound(hashKey);
        if(node == nodes.end())
        {
            // Use the first node if not found
            node = nodes.begin();
        }

        // Update stats
        stats[node->second]++;

        return true; 
    }

    void ResetStats()
    {
        stats.clear();
    }

    void PrintStats() const
    {
        std::cout << "Server Hit stats with Consistent Hashing: ";
        for(auto x : stats)
        {
            std::cout << std::endl;
            std::cout << x.first << ": " << x.second;
        }
        std::cout << std::endl;
    }

private:
    // Virtual nodes number for each real server
    int virtualNum; 
    // The real servers
    std::vector<std::string> servers; 
    // The virtual servers. Key is hash, value is the real server
    std::map<unsigned int, std::string> nodes; 
    // Stats, key is the real server, value is the hit count
    std::unordered_map<std::string, int> stats; 
};
```

往这个LB添加5个服务器，每个服务器默认的虚拟节点数为32个，然后发起一百万次请求：

```cpp
    LoadBalanceConsistentHashing lb;
    lb.AddServer("192.168.1.10");
    lb.AddServer("192.168.1.11");
    lb.AddServer("192.168.1.12");
    lb.AddServer("192.168.1.13");
    lb.AddServer("192.168.1.14");

    for(int i = 0; i < 1000000; i++)
    {
        lb.NextRequest();
    }

    lb.PrintStats();
```

可以看到，每个服务器被分发的请求总数比较均匀：

```shell
Server Hit stats with Consistent Hashing: 
192.168.1.13: 200693
192.168.1.14: 140848
192.168.1.11: 180680
192.168.1.12: 265031
192.168.1.10: 212748
```

我在代码里，用的是C++自带的哈希函数。如果使用别的哈希算法，比如Fowler-Noll-Vo，还能得到更加均匀的分布。

接下来，尝试把其中一个服务器删除，它附属的32个虚拟节点也会被删除。代码如下：

```cpp
    // Delete one server, and try again
    lb.DeleteServer("192.168.1.12");
    lb.ResetStats();
    for(int i = 0; i < 1000000; i++)
    {
        lb.NextRequest();
    }
    lb.PrintStats();
```

得到的输出，依然比较均匀：

```shell
Server Hit stats with Consistent Hashing: 
192.168.1.14: 205757
192.168.1.10: 292805
192.168.1.11: 241764
192.168.1.13: 259674
```

#### 7. 最小连接数法 Least Connection

检测所有后端服务器中，连接数最少的一个，然后把请求分发给它。连接数少，可以认为它处理的快，那么能者多劳，再多处理一点。

这个需要去统计、获取后端服务器的连接数，然后才能判断。就不实现了。

文章中的代码，全部上传在GitHub，欢迎访问。[GitHub - yuchuanwang/LoadBalance](https://github.com/yuchuanwang/LoadBalance)
