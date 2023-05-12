## Kubernetes CNI之Flannel网络模型分析

前面，我们在[搭建Kubernetes集群](https://github.com/yuchuanwang/docs/blob/main/Network/Kubernetes_Installation.md)时，用了比较常见的Flannel做为集群的CNI，它会负责集群内跨Node的通信。

该CNI的yaml文件定义在这里：https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

其中，跟本文相关的重点内容是：

```yaml
data:
  cni-conf.json: |
    {
      "name": "cbr0",
      "cniVersion": "0.3.1",
      "plugins": [
        {
          "type": "flannel",
          "delegate": {
            "hairpinMode": true,
            "isDefaultGateway": true
          }
        },
        {
          "type": "portmap",
          "capabilities": {
            "portMappings": true
          }
        }
      ]
    }
  net-conf.json: |
    {
      "Network": "10.244.0.0/16",
      "Backend": {
        "Type": "vxlan"
      }
    }
```

在net-conf.json部分，可以看到，集群内Pod的网段是：10.244.0.0/16；默认启用的backend类型是VXLAN。

本文将分析两种不同情况下的Pod之间的网络通信方式：

1. 相同Node，不同Pod；

2. 不同Node，不同Pod。

实验的集群，带了两个Node，其中的Master Node，已经被去除了污点，允许被调度。

为了实验相同Node、不同Node的两种情况，计划建立4个Replicas的Busybox。这样可以保证：每个Node上都有两个Pod，用来做相同Node通信的情况；每个Node上都会有Pod，用来实验不同Node通信的情况。

OK, let's hit the road. 

#### 1. 创建YAML文件

首先创建Deployment的YAML文件：

```shell
$ vi busybox_deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: busybox-deployment
spec:
  selector:
    matchLabels:
      app: busybox
  replicas: 4
  template:
    metadata:
      labels:
        app: busybox
    spec:
      containers:
      - name: busybox
        image: busybox
        command:
          - sleep
          - "3600"
        imagePullPolicy: IfNotPresent
      restartPolicy: Always
```

按:wq保存退出。

#### 2. 创建Deployment

用上一步的YAML文件，创建K8S的Deployment：

```shell
$ kubectl apply -f busybox_deployment.yaml
deployment.apps/busybox-deployment configured
```

可以看到，K8S已经按照预期，在两个Node上，分别创建了两个Pod：

```shell
$ kubectl get pods -o wide
NAME                                 READY   STATUS    RESTARTS   AGE     IP            NODE                   NOMINATED NODE   READINESS GATES
busybox-deployment-6fc48fb64-4c5l2   1/1     Running   0          12s     10.244.0.19   ycwang-ubuntu          <none>           <none>
busybox-deployment-6fc48fb64-4nccz   1/1     Running   0          12s     10.244.1.14   ycwang-ubuntu-worker   <none>           <none>
busybox-deployment-6fc48fb64-jbhxl   1/1     Running   0          12s     10.244.1.13   ycwang-ubuntu-worker   <none>           <none>
busybox-deployment-6fc48fb64-vd89t   1/1     Running   0          12s     10.244.0.18   ycwang-ubuntu          <none>           <none>
```

其中，10.244.0.18、10.244.0.19的两个Pod，运行在Master Node；10.244.1.13、10.244.1.14的两个Pod，运行在Worker Node。

#### 3. 节点上的网卡

先看一下此时在Master Node上的网卡情况：

```shell
$ ip addr
......
2: ens33: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UP group default qlen 1000
    link/ether 00:0c:29:49:39:91 brd ff:ff:ff:ff:ff:ff
    altname enp2s1
    inet 192.168.111.128/24 brd 192.168.111.255 scope global dynamic noprefixroute ens33
       valid_lft 1628sec preferred_lft 1628sec
    inet6 fe80::72bf:3960:42cd:13cb/64 scope link noprefixroute
       valid_lft forever preferred_lft forever
......
4: flannel.1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UNKNOWN group default
    link/ether ce:5f:f6:73:cf:b9 brd ff:ff:ff:ff:ff:ff
    inet 10.244.0.0/32 scope global flannel.1
       valid_lft forever preferred_lft forever
    inet6 fe80::cc5f:f6ff:fe73:cfb9/64 scope link
       valid_lft forever preferred_lft forever
5: cni0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UP group default qlen 1000
    link/ether e6:62:4d:8e:51:21 brd ff:ff:ff:ff:ff:ff
    inet 10.244.0.1/24 brd 10.244.0.255 scope global cni0
       valid_lft forever preferred_lft forever
    inet6 fe80::e462:4dff:fe8e:5121/64 scope link
       valid_lft forever preferred_lft forever
.......
```

我们只要观察上面的三个网卡即可。

**ens33**：是Node的物理网卡，所有跨Node的通信，不管用了什么技术，最终都是要通过它进出的。

**flannel.1**：是Flannel安装的网卡/VTEP，用来实现VXLAN的支持。跨Node的Pod之间通信，都需要经过它来进行Overlay网络的封装和解封装。

可以看一下它的具体信息：

```shell
$ ip -details link show flannel.1
4: flannel.1: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1450 qdisc noqueue state UNKNOWN mode DEFAULT group default 
    link/ether 36:a8:31:9e:3e:5d brd ff:ff:ff:ff:ff:ff promiscuity 0 minmtu 68 maxmtu 65535 
    vxlan id 1 local 192.168.111.128 dev ens33 srcport 0 0 dstport 8472 nolearning ttl auto ageing 300 udpcsum noudp6zerocsumtx noudp6zerocsumrx addrgenmode eui64 numtxqueues 1 numrxqueues 1 gso_max_size 65536 gso_max_segs 65535
```

可以看到，这个VXLAN设备，VNI是1、本地IP是192.168.111.128、目标端口是8472(不是标准VXLAN的4789)。

**cni0**：这个跟Docker里面带的Docker0感觉没什么区别，就是一个Linux Bridge。每个Pod会产生一个veth pair，一端接在Pod里面；另一端接在cni0里面。Pod之间的通信，都需要经过这个网桥。

#### 4. 网络拓扑

这张图，是Pod在相同Node通信、不同Node通信的总结。后面的实验，都是为了验证这张图。

<img title="" src="https://github.com/yuchuanwang/docs/blob/main/Assets/Flannel.png" alt="" width="" height="">

**相同Node、不同Pod之间的通信，走的是黑色 + 绿色的线**。Pod A和Pod B都通过veth接在同一个网桥cni0上。Pod A将数据通过veth发到cni0之后，cni0将数据转发到Pod B的veth。

**不同Node、不同Pod之间的通信，走的是黑色 + 蓝色的线**。Node 1上的Pod A，给Node 2上的Pod C发数据时，途径cni0，然后经过flannel.1进行VXLAN封装，最后依靠Node 1的物理网卡将数据发送给Node 2的物理网卡。Node 2的物理网卡再将数据依次发送给Node 2上的flannel.1、cni0，最后到达Pod C。

#### 5. 相同Node通信实验

登录运行在Master Node上的一个Pod：

```shell
$ kubectl exec -it busybox-deployment-6fc48fb64-4c5l2 -- sh
/ # ip addr
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host
       valid_lft forever preferred_lft forever
2: eth0@if11: <BROADCAST,MULTICAST,UP,LOWER_UP,M-DOWN> mtu 1450 qdisc noqueue
    link/ether a6:72:78:4a:0b:72 brd ff:ff:ff:ff:ff:ff
    inet 10.244.0.19/24 brd 10.244.0.255 scope global eth0
       valid_lft forever preferred_lft forever
    inet6 fe80::a472:78ff:fe4a:b72/64 scope link
       valid_lft forever preferred_lft forever
```

该Pod的IP是：10.244.0.19。用它去ping相同Node上的另一个Pod：10.244.0.18：

```shell
/ # ping 10.244.0.18
PING 10.244.0.18 (10.244.0.18): 56 data bytes
64 bytes from 10.244.0.18: seq=0 ttl=64 time=0.373 ms
64 bytes from 10.244.0.18: seq=1 ttl=64 time=0.065 ms
......
```

然后在Node上的cni0抓包：

```shell
$ sudo tcpdump -i cni0 -s 0 -X -nnn -vvv
17:06:16.588123 IP (tos 0x0, ttl 64, id 58509, offset 0, flags [DF], proto ICMP (1), length 84)
    10.244.0.19 > 10.244.0.18: ICMP echo request, id 15, seq 17, length 64
      0x0000:  4500 0054 e48d 4000 4001 400f 0af4 0013  E..T..@.@.@.....
      0x0010:  0af4 0012 0800 f64d 000f 0011 e9d3 17be  .......M........
      0x0020:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0030:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0050:  0000 0000                                ....
17:06:16.588135 IP (tos 0x0, ttl 64, id 12216, offset 0, flags [none], proto ICMP (1), length 84)
    10.244.0.18 > 10.244.0.19: ICMP echo reply, id 15, seq 17, length 64
      0x0000:  4500 0054 2fb8 0000 4001 34e5 0af4 0012  E..T/...@.4.....
      0x0010:  0af4 0013 0000 fe4d 000f 0011 e9d3 17be  .......M........
      0x0020:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0030:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0050:  0000 0000
```

可以看到，ICMP的往返报文在cni0上传递。

如果此时在Node上的flannel.1抓包，可以发现什么报文都没有：

```shell
$ sudo tcpdump -i flannel.1 -s 0 -X -nnn -vvv

```



所以，相同Node、不同Pod之间的通信，依靠的是veth + cni0网桥。

#### 6. 不同Node通信实验

继续用Master Node的Pod 10.244.0.19，用它去ping在另一个Node上的Pod 10.244.1.13：

```shell
/ # ping 10.244.1.13
PING 10.244.1.13 (10.244.1.13): 56 data bytes
64 bytes from 10.244.1.13: seq=0 ttl=62 time=1.099 ms
64 bytes from 10.244.1.13: seq=1 ttl=62 time=0.544 ms
......
```

此时，在Master Node上的cni0抓包：

```shell
$ sudo tcpdump -i cni0 -s 0 -X -nnn -vvv
tcpdump: listening on cni0, link-type EN10MB (Ethernet), snapshot length 262144 bytes
17:10:01.770478 IP (tos 0x0, ttl 64, id 48266, offset 0, flags [DF], proto ICMP (1), length 84)
    10.244.0.19 > 10.244.1.13: ICMP echo request, id 19, seq 4, length 64
      0x0000:  4500 0054 bc8a 4000 4001 6717 0af4 0013  E..T..@.@.g.....
      0x0010:  0af4 010d 0800 0647 0013 0004 6dd6 83cb  .......G....m...
      0x0020:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0030:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0050:  0000 0000                                ....
17:10:01.771463 IP (tos 0x0, ttl 62, id 15347, offset 0, flags [none], proto ICMP (1), length 84)
    10.244.1.13 > 10.244.0.19: ICMP echo reply, id 19, seq 4, length 64
      0x0000:  4500 0054 3bf3 0000 3e01 29af 0af4 010d  E..T;...>.).....
      0x0010:  0af4 0013 0000 0e47 0013 0004 6dd6 83cb  .......G....m...
      0x0020:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0030:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0050:  0000 0000
```

可以看到，ICMP的往返报文也是通过cni0进行传递的。

继续在Master Node上的flannel.1抓包：

```shell
$ sudo tcpdump -i flannel.1 -s 0 -X -nnn -vvv
tcpdump: listening on flannel.1, link-type EN10MB (Ethernet), snapshot length 262144 bytes
17:12:02.155017 IP (tos 0x0, ttl 63, id 10040, offset 0, flags [DF], proto ICMP (1), length 84)
    10.244.0.19 > 10.244.1.13: ICMP echo request, id 20, seq 3, length 64
      0x0000:  4500 0054 2738 4000 3f01 fd69 0af4 0013  E..T'8@.?..i....
      0x0010:  0af4 010d 0800 cc53 0014 0003 7ac2 b0d2  .......S....z...
      0x0020:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0030:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0050:  0000 0000                                ....
17:12:02.156094 IP (tos 0x0, ttl 63, id 36934, offset 0, flags [none], proto ICMP (1), length 84)
    10.244.1.13 > 10.244.0.19: ICMP echo reply, id 20, seq 3, length 64
      0x0000:  4500 0054 9046 0000 3f01 d45b 0af4 010d  E..T.F..?..[....
      0x0010:  0af4 0013 0000 d453 0014 0003 7ac2 b0d2  .......S....z...
      0x0020:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0030:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0040:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0050:  0000 0000
```

这种情况下，报文经cni0，传送到了flannel.1。Flannel.1进行VXLAN封装后，会继续传送到Master Node的物理网卡。

继续在Master Node上的ens33抓包：

```shell
$ sudo tcpdump -i ens33 udp -s 0 -X -nnn -vvv
tcpdump: listening on ens33, link-type EN10MB (Ethernet), snapshot length 262144 bytes
17:12:46.181905 IP (tos 0x0, ttl 64, id 21130, offset 0, flags [none], proto UDP (17), length 134)
    192.168.111.128.60568 > 192.168.111.129.8472: [bad udp cksum 0x60d6 -> 0x76a1!] OTV, flags [I] (0x08), overlay 0, instance 1
IP (tos 0x0, ttl 63, id 15296, offset 0, flags [DF], proto ICMP (1), length 84)
    10.244.0.19 > 10.244.1.13: ICMP echo request, id 20, seq 47, length 64
      0x0000:  4500 0086 528a 0000 4011 c78a c0a8 6f80  E...R...@.....o.
      0x0010:  c0a8 6f81 ec98 2118 0072 60d6 0800 0000  ..o...!..r`.....
      0x0020:  0000 0100 3a87 cf81 6ace ce5f f673 cfb9  ....:...j.._.s..
      0x0030:  0800 4500 0054 3bc0 4000 3f01 e8e1 0af4  ..E..T;.@.?.....
      0x0040:  0013 0af4 010d 0800 2e59 0014 002f 788e  .........Y.../x.
      0x0050:  50d5 0000 0000 0000 0000 0000 0000 0000  P...............
      0x0060:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0070:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0080:  0000 0000 0000                           ......
17:12:46.182333 IP (tos 0x0, ttl 64, id 8026, offset 0, flags [none], proto UDP (17), length 134)
    192.168.111.129.55939 > 192.168.111.128.8472: [udp sum ok] OTV, flags [I] (0x08), overlay 0, instance 1
IP (tos 0x0, ttl 63, id 42036, offset 0, flags [none], proto ICMP (1), length 84)
    10.244.1.13 > 10.244.0.19: ICMP echo reply, id 20, seq 47, length 64
      0x0000:  4500 0086 1f5a 0000 4011 faba c0a8 6f81  E....Z..@.....o.
      0x0010:  c0a8 6f80 da83 2118 0072 88b6 0800 0000  ..o...!..r......
      0x0020:  0000 0100 ce5f f673 cfb9 3a87 cf81 6ace  ....._.s..:...j.
      0x0030:  0800 4500 0054 a434 0000 3f01 c06d 0af4  ..E..T.4..?..m..
      0x0040:  010d 0af4 0013 0000 3659 0014 002f 788e  ........6Y.../x.
      0x0050:  50d5 0000 0000 0000 0000 0000 0000 0000  P...............
      0x0060:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0070:  0000 0000 0000 0000 0000 0000 0000 0000  ................
      0x0080:  0000 0000 0000 
```

可以看到，这是Overlay的报文。报文内层是10.244.0.19 > 10.244.1.13的ICMP请求；报文外层是192.168.111.128 > 192.168.111.129的UDP请求。只不过由于它的目标端口是自定义的8472，而不是RFC定义的4789，tcpdump没把它解释成VXLAN报文而已。

所以，不同Node、不同Pod之间的通信，依靠的是veth + cni0 + flannel.1(VXLAN)。

