## 介绍一个简单清晰的Redis C++客户端



Redis的客户端里面，对于C语言，一般的选择就是使用官方的hiredis；

而对于C++语言，要么是选择hiredis，然后自己去创建、释放资源，构建各种Redis命令；要么是使用Github上面各种C++的实现。官方并没有推荐的C++实现。

我曾经在Github上搜了不少时间，发现多数的C++实现要么是依赖太多的第三方代码、要么是设计复杂。没找到满足我自己需要的实现：简单、够用。



因此，我自己实现了一个开源的**MiniRedisClient**代码，放在[GitHub - yuchuanwang/RedisClient: Clean Redis Client with simple interfaces, and pipeline/publish/subscribe supported](https://github.com/yuchuanwang/RedisClient)，欢迎大家使用、提出问题、Star。

这个代码实现了如下的功能：

1. 主要的Redis命令，对于未封装的命令，也提供了原生的命令接口去调用；

2. 流水线功能。可以以批处理的形式，发送多条Redis命令；

3. 发布功能。可以把消息发布给指定的Channel，订阅该Channel的客户端能正常收到所发布的消息；

4. 订阅功能。可以订阅指定的Channel，当有消息被发布到该Channel时，可以收到该消息。

在这些功能里面，除了订阅功能额外引入了libevent的依赖之外；其他的功能都只需要依赖官方的hiredis，直接yum install或apt install libhiredis-dev即可。



以下是功能演示。

**第一部分，调用主要的Redis命令实现、以及流水线功能。**

```cpp
void TestClient()
{
    std::string repliedStr;
    MiniRedisClient client;
    client.Connect("127.0.0.1", 6379);
    client.client_setname("MiniCppClient", repliedStr);
    client.client_getname(repliedStr); 
    std::cout << repliedStr << std::endl;

    std::cout << client.ping() << std::endl;
    std::cout << client.ping("Hello Redis") << std::endl;

    std::cout << client.auth("password123", repliedStr) << std::endl; 
    std::cout << client.select(1, repliedStr) << std::endl; 
    std::cout << client.select(0, repliedStr) << std::endl; 

    client.set("key 1", "value 1", 3600, repliedStr);
    client.set("key 2", "value 2", 0, repliedStr);

    client.set("key 3", 1234, 3600, repliedStr);
    client.set("key 4", 1001, 0, repliedStr);

    long long int repliedInt = 0;
    client.expire("key 1", 360, repliedInt); 
    client.expire("key 2", 60, repliedInt); 

    client.ttl("key 2", repliedInt);
    client.ttl("key 4", repliedInt);
    client.ttl("invalid", repliedInt);

    client.strlen("key 4", repliedInt);
    client.strlen("invalid", repliedInt);

    client.append("key 4", "2345678", repliedInt);
    client.append("invalid", "ACBDEDF", repliedInt);
    client.strlen("key 4", repliedInt);
    client.strlen("invalid", repliedInt);
    client.del("invalid", repliedInt);

    client.get("key 1", repliedStr);

    client.exists("key 1", repliedInt);
    client.exists("invalid", repliedInt);

    std::string keyDel = "key 2";
    client.del(keyDel, repliedInt);
    keyDel = "key 3";
    client.del(keyDel, repliedInt);
    keyDel = "key 4";
    client.del(keyDel, repliedInt);
    keyDel = "key 5";
    client.del(keyDel, repliedInt);

    client.hset("domains", "example", "example.com", repliedInt); 
    client.hset("domains", "abc", "abc.com", repliedInt); 
    client.hget("domains", "example", repliedStr);

    client.hset("newHash", "me", "1234567890", repliedInt); 
    client.hget("newHash", "you", repliedStr);
    client.hdel("newHash", "me", repliedInt);

    std::map<std::string, std::string> repliedMap; 
    client.hgetall("domains", repliedMap);
    std::vector<std::string> repliedArray; 
    client.hkeys("domains", repliedArray);
    client.hvals("domains", repliedArray);

    client.hexists("domains", "abc", repliedInt);
    client.hexists("domains", "invalid", repliedInt);
    client.hlen("domains", repliedInt);

    client.keys("*", repliedArray);
    client.keys("user*", repliedArray);

    client.rename("users", "friends", repliedStr);
    client.rename("friends", "users", repliedStr);

    client.type("domains", repliedStr);
    client.type("users", repliedStr);
    client.type("username", repliedStr);
    client.type("Null", repliedStr);

    client.decr("counter", repliedInt); 
    client.incr("counter", repliedInt); 

    client.lpush("List123", "item 1", repliedInt);
    client.lpush("List123", "item 2", repliedInt);
    client.lpush("List123", "item 3", repliedInt);
    client.lpush("List123", "item 4", repliedInt);
    client.llen("List123", repliedInt);
    client.lpop("List123", repliedStr); 
    client.llen("List123", repliedInt);
    client.linsert_before("List123", "item 3", "item 2+", repliedInt);
    client.linsert_after("List123", "item 3", "item 3+", repliedInt);
    client.lset("List123", 2, "item set 2", repliedStr); 
    client.lrem("List123", 0, "item 2+", repliedInt);
    client.lindex("List123", 0, repliedStr);
    client.lindex("List123", -1, repliedStr);
    //client.del("List123", repliedInt);

    client.sadd("set123", "ele 1", repliedInt);
    client.sadd("set123", "ele 2", repliedInt);
    client.sadd("set123", "ele 3", repliedInt);
    client.sadd("set123", "ele 4", repliedInt);
    client.sadd("set123", "ele 5", repliedInt);
    client.sadd("set123", "ele 5", repliedInt);
    client.sadd("set123", "ele 4", repliedInt);
    client.scard("set123", repliedInt);
    client.sismember("set123", "ele 1", repliedInt);
    client.sismember("set123", "ele 8", repliedInt);
    client.smembers("set123", repliedArray);
    client.srem("set123", "ele 1", repliedInt);
    client.srem("set123", "ele 8", repliedInt);
    //client.del("set123", repliedInt);

    client.set("Blank space", "value", 0, repliedStr);
    std::vector<std::string> keysDel;
    keysDel.push_back("List123");
    keysDel.push_back("set123");
    keysDel.push_back("Blank space");
    client.del(keysDel, repliedInt);

    // Test pipeline
    client.set("pipeline", 168, 0, repliedStr);
    std::vector<std::string> commands;
    for (int i = 0; i < 1000; i++)
    {
        std::string cmd = "INCR pipeline";
        commands.push_back(cmd);
    }
    client.pipeline(commands, repliedArray);

    commands.clear();
    for (int i = 0; i < 1000; i++)
    {
        std::string cmd = "PING";
        commands.push_back(cmd);
    }
    client.pipeline(commands, repliedArray);

    commands.clear();
    for (int i = 0; i < 1000; i++)
    {
        std::string cmd = "PING Redis";
        commands.push_back(cmd);
    }
    client.pipeline(commands, repliedArray);

    std::cout << repliedInt << std::endl; 
}
```

在这些Demo代码里，包括了常见的连接、ping、get/set命令、各种key-value操作、Hash操作、List操作、Set操作、以及Pipeline。



**第二部分，调用发布功能。**

```cpp
void TestPub()
{
    MiniRedisPubSub pub;
    pub.Connect("127.0.0.1", 6379);

    int interval = 60;
    int i = 0;
    std::string content("Content from C++ client"); 
    while (i < interval)
    {
        pub.Publish("channelFromCpp", content); 
        std::this_thread::sleep_for(std::chrono::seconds(1));
        i++; 
    }

    std::cout << "Publishing done" << std::endl; 
}
```



使用很简单。连接之后，每隔一秒钟，往Channel发布一条消息，连续发布60秒。



**第三部分，调用订阅功能。**

```cpp
void SubscribeCb(const std::string& channel, const std::string& content)
{
    std::cout << "Subscriber CB receives channel: " << channel << ", data: " << content << std::endl; 
}

void TestSub()
{
    MiniRedisPubSub sub;
    sub.SetSubscribeCb(SubscribeCb);
    sub.Connect("127.0.0.1", 6379);
    sub.Subscribe("testChannel1");
    sub.Subscribe("testChannel2");

    int interval = 60;
    int i = 0;
    while (i < interval)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        i++; 
    }
    
    std::cout << "Subscribing done" << std::endl; 
}
```

在使用时，订阅方可以提供一个回调函数，当收到订阅的消息时该函数被触发。



代码基本都是自解释的。

欢迎使用并指出问题，谢谢。


