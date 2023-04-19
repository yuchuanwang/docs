## Docker容器网络的七种武器

知识，学过了之后，把它总结、分享出来，能让自己对它的理解更加的深入。

因此，把以前学的Docker容器网络模型归纳总结、并进行试验。

后续再继续对Kubernetes、CNI进行总结试验。

---

Docker对网络的支持，可以用如下的思维导图来表示：

![DockerNetwork](https://github.com/yuchuanwang/docs/blob/main/Assets/DockerNetwork.png | width=400)

下面，针对每种网络模型进行介绍与试验。


#### 一. 拔网线 - None模型

None，啥都没有，类似于把网线给拔掉了。所以，这种模式下的容器，是一个封闭的环境。

适用于安全性高、又不需要网络访问的情景。

运行容器时，指定：--network=none即可。

```shell
$ docker run -it --rm --name=bbox --network=none busybox sh
```

运行一个BusyBox的容器，然后在容器内可以看到：

```shell
/ # ip link
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
```

该容器除了一个localhost的网卡，并没有对外进行网络通信的设备。


#### 二. 寄生 - Host模型

使用该模式的容器，共享Host宿主机(运行Docker的机器)的网络栈、网卡设备。

这种情况下，容器的网络性能很好。但是不灵活，容器的端口容易与Host的端口冲突。Host A上能正常运行，换到了Host B未必就能正常运行。根据我的经验，这种模式很少有实际应用。

运行容器时，指定：--network=host即可。

```shell
$ docker run -it --rm --name=bbox --network=host busybox sh
```

在容器内看到的网卡信息：

```shell
/ # ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host
       valid_lft forever preferred_lft forever
2: ens33: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel qlen 1000
    link/ether 00:0c:29:49:39:91 brd ff:ff:ff:ff:ff:ff
    inet **192.168.111.128**/24 brd 192.168.111.255 scope global dynamic noprefixroute ens33
       valid_lft 1242sec preferred_lft 1242sec
    inet6 fe80::72bf:3960:42cd:13cb/64 scope link noprefixroute
       valid_lft forever preferred_lft forever
3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue
    link/ether 02:42:65:3a:0c:37 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft forever
    inet6 fe80::42:65ff:fe3a:c37/64 scope link
       valid_lft forever preferred_lft forever
```

与在Host宿主机看到的信息是一致的：

```shell
$ ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host
       valid_lft forever preferred_lft forever
2: ens33: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP group default qlen 1000
    link/ether 00:0c:29:49:39:91 brd ff:ff:ff:ff:ff:ff
    altname enp2s1
    inet **192.168.111.128**/24 brd 192.168.111.255 scope global dynamic noprefixroute ens33
       valid_lft 1148sec preferred_lft 1148sec
    inet6 fe80::72bf:3960:42cd:13cb/64 scope link noprefixroute
       valid_lft forever preferred_lft forever
3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default
    link/ether 02:42:65:3a:0c:37 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft forever
    inet6 fe80::42:65ff:fe3a:c37/64 scope link
       valid_lft forever preferred_lft forever
```

从容器内访问网络，一切正常：

```shell
/ # ping 8.8.8.8
PING 8.8.8.8 (8.8.8.8): 56 data bytes
64 bytes from 8.8.8.8: seq=0 ttl=53 time=34.735 ms
64 bytes from 8.8.8.8: seq=1 ttl=53 time=35.659 ms
64 bytes from 8.8.8.8: seq=2 ttl=53 time=35.603 ms
64 bytes from 8.8.8.8: seq=3 ttl=53 time=35.723 ms
```


#### 三. 搭桥 - Bridge模型

这是Docker在运行容器时，默认的网络模型。

Docker在安装时，会自动在系统里面创建一个叫做docker0的网桥：

```shell
$ ip addr
......
3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default
    link/ether 02:42:65:3a:0c:37 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0
       valid_lft forever preferred_lft forever
    inet6 fe80::42:65ff:fe3a:c37/64 scope link
       valid_lft forever preferred_lft forever
```

继续查看该网桥的详细信息：

```shell
$ docker network inspect bridge
[
    {
        "Name": "bridge",
        "Id": "d92abe90e2bc79d8a4cd5ae73138d8da7aa0684a6a170fe7fc0ade4518057440",
        "Created": "2023-04-18T11:02:11.199720084+08:00",
        "Scope": "local",
        "Driver": "bridge",
        "EnableIPv6": false,
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    **"Subnet": "172.17.0.0/16",
                    "Gateway": "172.17.0.1"**
                }
            ]
        },
        "Internal": false,
        "Attachable": false,
        "Ingress": false,
        "ConfigFrom": {
            "Network": ""
        },
        "ConfigOnly": false,
        "Containers": {},
        "Options": {
            "com.docker.network.bridge.default_bridge": "true",
            "com.docker.network.bridge.enable_icc": "true",
            "com.docker.network.bridge.enable_ip_masquerade": "true",
            "com.docker.network.bridge.host_binding_ipv4": "0.0.0.0",
            "com.docker.network.bridge.name": "docker0",
            "com.docker.network.driver.mtu": "1500"
        },
        "Labels": {}
    }
]
```

根据上面的信息，Docker在运行容器时，会在172.17.0.0/16网段，为容器分配IP地址，并把Gateway指向172.17.0.1，即docker0这个虚拟设备。

而且，Docker会为运行的容器创建一对veth。该veth pair，一端接在容器内部，另一端接在docker0上。使得容器可以通过docker0与外界通信。

运行一个容器，可以看到容器里面的网络设备：

```shell
$ docker run -it --rm --name=bbox busybox sh
/ # ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
6: eth0@if7: <BROADCAST,MULTICAST,UP,LOWER_UP,M-DOWN> mtu 1500 qdisc noqueue
    link/ether 02:42:ac:11:00:02 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.2/16 brd 172.17.255.255 scope global eth0
       valid_lft forever preferred_lft forever
```

容器内部有eth0网卡，它是veth的一端。在Host上，可以看到veth的另一端：

```shell
$ brctl show
bridge name        bridge id            STP enabled        interfaces
docker0            8000.0242653a0c37    no                vethe22e54f
```

网络拓扑参考官方原图：

![DockerBridge](https://github.com/yuchuanwang/docs/blob/main/Assets/Docker_Bridge.png)


###### 1. 相同Host上的容器间网络通信

在这种模式下，同一个Host上的不同容器，可以通过docker0直接通信。比如运行一个Nginx的容器：

```shell
$ docker run -it --rm --name=web nginx
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
/docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
/docker-entrypoint.sh: Configuration complete; ready for start up
2023/04/18 03:50:39 [notice] 1#1: using the "epoll" event method
2023/04/18 03:50:39 [notice] 1#1: nginx/1.23.4
2023/04/18 03:50:39 [notice] 1#1: built by gcc 10.2.1 20210110 (Debian 10.2.1-6)
2023/04/18 03:50:39 [notice] 1#1: OS: Linux 5.19.0-38-generic
2023/04/18 03:50:39 [notice] 1#1: getrlimit(RLIMIT_NOFILE): 1048576:1048576
2023/04/18 03:50:39 [notice] 1#1: start worker processes
2023/04/18 03:50:39 [notice] 1#1: start worker process 28
2023/04/18 03:50:39 [notice] 1#1: start worker process 29
```

在另一个shell窗口里，先查看该容器的IP地址，确定为172.17.0.2：

```shell
$ docker network inspect bridge
[
    {
        "Name": "bridge",
        ......
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.17.0.0/16",
                    "Gateway": "172.17.0.1"
                }
            ]
        },
        ......
        "Containers": {
            "d3025a31c47c80bdcf711f329ff3c70677c28bfaed08d700743d91ad1bc33f15": {
                "Name": "web",
                "EndpointID": "d6f319d19fac52189c70fddb3a732f5b4081ff16fbe21e859c647ff8ff8ae7e6",
                "MacAddress": "02:42:ac:11:00:02",
                **"IPv4Address": "172.17.0.2/16",**
                "IPv6Address": ""
            }
        },
        ......
    }
]
```

然后，运行一个BusyBox的容器：

```shell
$ docker run -it --rm --name=bbox busybox sh
/ # wget 172.17.0.2:80
Connecting to 172.17.0.2:80 (172.17.0.2:80)
saving to 'index.html'
index.html           100% |********************************************************************************************************************************************************************************|   615  0:00:00 ETA
'index.html' saved
```

可以看到，BusyBox容器成功的访问了Nginx容器。


###### 2. 容器与外部网络通信

**2.1. 从内到外**：

Bridge模式下的容器，默认就可以访问外部网络。它依靠Host上的iptables，做了NAT地址转换。

启动一个BusyBox的容器，得到的IP是172.17.0.2。它的Host IP是：192.168.111.128。在容器内可以直接访问外部的另一台机器：192.168.111.129。

```shell
$ docker run -it --rm --name=bbox busybox sh
/ # ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
12: eth0@if13: <BROADCAST,MULTICAST,UP,LOWER_UP,M-DOWN> mtu 1500 qdisc noqueue
    link/ether 02:42:ac:11:00:02 brd ff:ff:ff:ff:ff:ff
    inet 172.17.0.2/16 brd 172.17.255.255 scope global eth0
       valid_lft forever preferred_lft forever
/ # ping 192.168.111.129
PING 192.168.111.129 (192.168.111.129): 56 data bytes
64 bytes from 192.168.111.129: seq=0 ttl=63 time=0.521 ms
64 bytes from 192.168.111.129: seq=1 ttl=63 time=0.465 ms
```

**2.2. 从外到内**：

如果需要外部网络访问Bridge模式下的容器，可以通过端口映射功能。在运行容器时，指定Host端口A与容器端口B的映射。然后，通过访问：Host-IP:Host端口A，即可映射到：容器:容器端口B。

我们做个试验，把前面的Nginx容器和BusyBox容器全部退出。然后重新运行一个新的Nginx容器，并通过-p参数指定端口映射：

```shell
$ docker run -it --rm --name=web -p 8080:80 nginx
```

从另外一台机器发起访问，192.168.111.128是Host的IP地址：

```shell
$ wget 192.168.111.128:8080
Connecting to 192.168.111.128:8080... connected.
HTTP request sent, awaiting response... 200 OK
Length: 615 [text/html]
Saving to: ‘index.html’

index.html                                              100%[===============================================================================================================================>]     615  --.-KB/s    in 0s      

‘index.html’ saved [615/615]
```


#### 四. 如影随形 - Container模型

这个模式我没看到官方的名字，名字我瞎取的，但是在Kubernetes的Pod里面经常用。

具体的做法，是在容器B运行时，指定：--network=container:容器A的名字或者ID。

这样，容器A、B处于同一个网络空间。它们的MAC地址、IP地址都一样，共享网络栈、网卡和配置信息。它们可以通过127.0.0.1直接通信。

在Kubernetes部署Pod的时候，就会用到这个模式。针对每个Pod，Kubernetes先启动Pause容器，然后再启动其它容器并使用Pause容器的网络。这样，同一个Pod之内的容器，共享了同一个网络空间，可以高效的通信。

试验看看，先启动一个Nginx容器：

```shell
$ docker run -it --rm --name=web nginx
```

看看Docker网络情况：

```shell
$ docker network inspect bridge
[
    {
        "Name": "bridge",
        ......
        "IPAM": {
            "Driver": "default",
            "Options": null,
            "Config": [
                {
                    "Subnet": "172.17.0.0/16",
                    "Gateway": "172.17.0.1"
                }
            ]
        },
        ......
        "Containers": {
            "da67b03b02c2e99fdaaa2fd75b7829c4005eba80a50f39404db4da8d8defa0e3": {
                "Name": "web",
                "EndpointID": "0e29b7473a8446100dc45a711536b0277ae0f911cb4c2decc7245511fd2dbb02",
                **"MacAddress": "02:42:ac:11:00:02",**
                **"IPv4Address": "172.17.0.2/16",**
                "IPv6Address": ""
            }
        },
    }
]
```

Nginx容器的IP地址是：172.17.0.2，MAC地址是：02:42:ac:11:00:02。

再启动一个BusyBox容器，并使用Nginx的网络：

```shell
$ docker run -it --rm --name=bbox --network=container:web busybox sh
/ # ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
16: eth0@if17: <BROADCAST,MULTICAST,UP,LOWER_UP,M-DOWN> mtu 1500 qdisc noqueue
    link/ether **02:42:ac:11:00:02** brd ff:ff:ff:ff:ff:ff
    inet **172.17.0.2**/16 brd 172.17.255.255 scope global eth0
       valid_lft forever preferred_lft forever
```

可以看到，BusyBox的IP地址、MAC地址，和Nginx的一模一样。

直接访问127.0.0.1可以得到：

```shell
/ # wget 127.0.0.1
Connecting to 127.0.0.1 (127.0.0.1:80)
saving to 'index.html'
index.html           100% |********************************************************************************************************************************************************************************|   615  0:00:00 ETA
'index.html' saved
```

这种模式除了K8S的Pod之外，还可以用在简易版的Web Server + App Server情景。


#### 五. 套娃 - Overlay模型

Docker通过Overlay模式，实现了对VXLAN的支持。这个模式的环境搭建比别的模式稍显复杂，主要是因为需要有一个地方来保存各个节点在overlay网络中的配置信息。一般是在另一个机器安装etcd或者Consul这种key-value数据库。

偷懒起见，我直接使用了Docker自带的Swarm来搭建环境。准备了两台机器A、B。A身兼两职，既保存数据库，又运行容器。

(悲剧的是，在实验之前，我手欠把Docker从23.0.3升级到23.0。4，然后Docker Swarm的集群就挂了，无法创建Service。相当不靠谱，难怪被K8S打趴下……)

- 首先，在机器A，初始化swarm：
 
  ```shell
  ycwang@ycwang-ubuntu:~$ docker swarm init
  Swarm initialized: current node (qygp7ymrfh5g0lgky10teck4r) is now a manager.
 
  To add a worker to this swarm, run the following command:
 
      docker swarm join --token SWMTKN-1-3rjaah348iir9pkrmssd4hrbtr5gkfpgw70m9l3v25mhqyll8d-33k4mggwe86kuidejitiprowo 192.168.111.128:2377
 
  To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
  ```

- 换到机器B，Copy上面的join命令，加入集群：
 
  ```shell
  ycwang@ycwang-ubuntu-slave:~$ docker swarm join --token SWMTKN-1-3rjaah348iir9pkrmssd4hrbtr5gkfpgw70m9l3v25mhqyll8d-33k4mggwe86kuidejitiprowo 192.168.111.128:2377
  This node joined a swarm as a worker.
  ```

- 回到机器A，可以看到集群的情况：
 
  ```shell
  ycwang@ycwang-ubuntu:~$ docker node ls
  ID                            HOSTNAME              STATUS    AVAILABILITY   MANAGER STATUS   ENGINE VERSION
  qygp7ymrfh5g0lgky10teck4r *   ycwang-ubuntu         Ready     Active         Leader           23.0.4
  sr637j4g891bsxo56tesv55y8     ycwang-ubuntu-slave   Ready     Active                          23.0.4
  ```

- 在机器A上，可以看到Docker为Overlay模式，创建了两个新的网络，docker_gwbridge和ingress。后面运行的容器，会通过docker_gwbridge与外部网络进行通信(南北向流量)：
 
  ```shell
  ycwang@ycwang-ubuntu:~$ docker network ls
  NETWORK ID     NAME              DRIVER    SCOPE
  51276c2e1741   bridge            bridge    local
  596dbdd24c3a   docker_gwbridge   bridge    local
  fc504698f255   host              host      local
  tmbwbg86eph4   ingress           overlay   swarm
  4829db6948ad   none              null      local
  ```

- 在机器A上，为Docker创建Overlay网络：
 
  ```shell
  ycwang@ycwang-ubuntu:~$ docker network create --driver=overlay vxlanA
  thya4qliq95dh81yndfqpimwn
  ```

- 在机器A上，创建服务，使用vxlanA这个网络，replicas 指定为 2：
 
  ```shell
  ycwang@ycwang-ubuntu:~$ docker service create --network=vxlanA --name bboxes --replicas 2 busybox
  ```
到了这一步，我之前在Docker 23.0.3能成功的在两个Node上运行两个容器，并且通过VXLAN在它们之间发送东西向流量。但Docker升级到23.0.4后，这一步就走不下去了…… 

根据之前版本的记忆，每个容器会带两张网卡。一张接在前面的docker_gwbridge网桥上，负责与外部网络的南北向流量。另一张负责VXLAN的东西向流量。

如果从容器A ping 容器B，并用tcpdump在Host宿主机的网卡上抓包，可以清楚的看到被VXLAN封装过的ICMP数据包。

这部分内容，等我把环境重新配置好了，再来补充。


#### 六. 狡兔三窟 - Macvlan模型

Macvlan是一种网卡虚拟化技术，将一张物理网卡(父接口)虚拟出多张网卡(子接口)。每个子接口有自己独立的 MAC 地址和 IP 地址。

物理网卡(父接口)相当于一个交换机，记录着对应的虚拟网卡(子接口)和 MAC 地址。当物理网卡收到数据包后，会根据目的 MAC 地址判断这个包属于哪一个虚拟网卡，并转发给它。

Macvlan技术有四种模式，Docker支持其中的bridge模式。

接下来，试验看看。

- 首先，需要打开网卡的混杂模式，否则它拒绝接收MAC地址跟它不一样的数据报文。ens33是Host机器的物理网卡：
  
  ```shell
  $ sudo ip link set ens33 promisc on
  ```

- 第二步，为Docker创建一个Macvlan网络。子网是：192.168.111.0/24，跟Host一样；指定父接口为ens33
  
  ```shell
  $ docker network create --driver=macvlan --subnet=192.168.111.0/24 --gateway=192.168.111.1 -o parent=ens33 macvnet
  0283990d6acdc9df87d5b34a999c05266e12a4423aa0041387373d8bc5ee042c
  ```

- 第三步，运行容器，指定其IP地址为：192.168.111.10，并使用上一步创建的Macvlan网络：
  
  ```shell
  $ docker run -it --rm --name=web --ip=192.168.111.10 --network=macvnet nginx
  ```

这样，Nginx这个容器就运行在了192.168.110.10这个地址上，从外部机器可以直接访问它：

```shell
$ wget 192.168.111.10
Connecting to 192.168.111.10:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 615 [text/html]
Saving to: ‘index.html’

index.html                                            100%[===============================================================================================================================>]     615  --.-KB/s    in 0s      

‘index.html’ saved [615/615]
```

可以看到，Macvlan 是一种将容器通过二层，连接到物理网络不错的方案，配置简单、性能好。但它也有一些局限性，比如：
物理网卡所连接的交换机，可能会限制同一个物理端口上的 MAC 地址数量。
许多物理网卡上的 MAC地址数量也有限制。


#### 七. 狡兔三窟Plus - IPvlan模型

IPvlan是一个比较新的特性，Linux内核>= 4.2之后才可以稳定的使用。

与Macvlan类似，都是从一个物理网卡(父接口)虚拟出多张网卡(子接口)。与Macvlan不同的是，这些子接口的MAC地址都是一样的，不一样的只是它们的IP地址。而且，它不像Macvlan那样，要求物理网卡打开混杂模式。

IPvlan有两种模式：L2和L3模式。顾名思义，L2模式跟交换机有关，L3模式则跟路由器有关。


**1. L2模式**

IPvlan的L2模式，跟之前的Macvlan非常类似。容器的子接口与父接口在同一子网，父接口做为交换机来转发子接口的数据。如果是与外部网络通信，则依赖父接口进行路由转发。

首先，为Docker创建一个L2模式的IPvlan网络：

```shell
$ docker network create --driver=ipvlan --subnet=192.168.0.0/24 --gateway=192.168.0.1 -o ipvlan_mode=l2 -o parent=ens33 ipv_l2
7091b861fd44798d21be6d3dcdd03e79c68d65e9149862b9f21bca42678fda19
```

该网络与Host宿主机同在192.168.0.0/24网段，Host的网卡是ens33，IP是192.168.0.105。

使用该网络，运行第一个容器：

```shell
$ docker run -it --rm --network=ipv_l2 --name=bbox1 busybox sh
/ # ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
4: eth0@if2: <BROADCAST,MULTICAST,UP,LOWER_UP,M-DOWN> mtu 1500 qdisc noqueue 
    link/ether 00:0c:29:12:5b:f4 brd ff:ff:ff:ff:ff:ff
    inet 192.168.0.2/24 brd 192.168.0.255 scope global eth0
       valid_lft forever preferred_lft forever
```

该容器的IP地址为192.168.0.2。MAC地址与Host宿主机的ens33一致。

从容器内，访问另外一台机器B：

```shell
/ # ping 192.168.0.112
PING 192.168.0.112 (192.168.0.112): 56 data bytes
64 bytes from 192.168.0.112: seq=0 ttl=64 time=79.549 ms
```

从另外的机器B，访问该容器：

```shell
$ ping 192.168.0.2
PING 192.168.0.2 (192.168.0.2) 56(84) bytes of data.
64 bytes from 192.168.0.2: icmp_seq=1 ttl=64 time=3.94 ms
```

可以看到，容器的网络访问都是没问题的。


**2. L3模式**

这个模式下，容器跟Host宿主机可以不在同一个子网。该模式的配置，网上的资料比较少，Docker官网也是语焉不详的。

假设Host宿主机的父接口是ens33，IP地址是192.168.0.105/24。

现在想要创建两个容器，分别属于不同的子网，例如10.0.1.0/24和10.0.2.0/24，并让它们可以相互通信，也可以访问外部网络。

可以按照如下的步骤来实现。

- 首先，把之前环境下的容器退出，并清理资源，因为一个父接口不能同时支持L2和L3模式：
  
  ```shell
  $ docker system prune
  ```

- 创建一个新的IPvlan网络：
  
  ```shell
  $ docker network create --driver=ipvlan --subnet=10.0.1.0/24 --subnet=10.0.2.0/24 -o parent=ens33 -o ipvlan_mode=l3 ipvlan-l3
  ```

- 在两个Terminal窗口，分布运行一个容器，并连接到刚刚创建的IPvlan网络、使用不同的子网。它们的IP地址分别为10.0.1.10和10.0.2.10：
  
  ```shell
  $ docker run -it --rm --name bbox1 --network ipvlan-l3 --ip 10.0.1.10 busybox sh
  / # ip addr
  1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
      link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
      inet 127.0.0.1/8 scope host lo
         valid_lft forever preferred_lft forever
  4: eth0@if2: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP,M-DOWN> mtu 1500 qdisc noqueue 
      link/ether 00:0c:29:12:5b:f4 brd ff:ff:ff:ff:ff:ff
      inet 10.0.1.10/24 brd 10.0.1.255 scope global eth0
         valid_lft forever preferred_lft forever
  ```
  
  ```shell
  $ docker run -it --rm --name bbox2 --network ipvlan-l3 --ip 10.0.2.10 busybox sh
  / # ip addr
  1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
      link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
      inet 127.0.0.1/8 scope host lo
         valid_lft forever preferred_lft forever
  5: eth0@if2: <BROADCAST,MULTICAST,NOARP,UP,LOWER_UP,M-DOWN> mtu 1500 qdisc noqueue 
      link/ether 00:0c:29:12:5b:f4 brd ff:ff:ff:ff:ff:ff
      inet 10.0.2.10/24 brd 10.0.2.255 scope global eth0
         valid_lft forever preferred_lft forever
  ```

- 容器之间互通性。此时，两个容器之间已经可以互相访问。
  
  从bbox1访问bbox2：
  
  ```shell
  / # ping 10.0.2.10
  PING 10.0.2.10 (10.0.2.10): 56 data bytes
  64 bytes from 10.0.2.10: seq=0 ttl=64 time=0.270 ms
  64 bytes from 10.0.2.10: seq=1 ttl=64 time=0.061 ms
  ```
  
  从bbox2访问bbox1：
  
  ```shell
  / # ping 10.0.1.10
  PING 10.0.1.10 (10.0.1.10): 56 data bytes
  64 bytes from 10.0.1.10: seq=0 ttl=64 time=0.077 ms
  64 bytes from 10.0.1.10: seq=1 ttl=64 time=0.077 ms
  ```

- 但此时，从外部网络访问这两个容器依然是无法到达的。因为外部的网络环境里，没有关于10.0.1.10或者10.0.2.10这两个IP地址的路由信息。
  
  需要在外部路由器上添加相应的路由规则，让它知道如何到达容器网络。
  
  假设外部路由器的接口为eth1，IP地址为192.168.0.1/24。
  
  添加路由规则，将目标地址为10.0.1.0/24或10.0.2.0/24的数据包转发到192.168.0.105，即，转发到容器的父接口ens33。
  
  ```shell
  $ sudo ip route add 10.0.1.0/24 via 192.168.0.105 dev eth1
  $ sudo ip route add 10.0.2.0/24 via 192.168.1.105 dev eth1
  ```

  这样，就可以实现IPvlan L3模式的容器与外部网络的通信。Sorry，我忘了我家里路由器的密码了，暂时没法登录实验……

综合运用下来，感觉IPvlan模式应该比Macvlan模式更加实用，因为Macvlan拥有的功能，IPvlan的L2模式都有，而且还少了混杂模式、MAC地址数目的潜在问题。除此之外，IPvlan还多了L3模式的支持。
