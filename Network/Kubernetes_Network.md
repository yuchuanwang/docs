## Kubernetes网络模型分析

前一篇文章，介绍了[Docker容器的网络模型](https://github.com/yuchuanwang/docs/blob/main/Network/Docker_Network.md)。

容器是要被K8S编排、管理的。而K8S又有自己的网络模型。我们继续学习、实验，来理解K8S是怎么样处理网络流量的。

实验之前，先分清楚K8S里面的三种IP地址，和三种端口。

**三种IP地址：**

1. Cluster IP：K8S在创建Service时，生成的虚拟IP地址。需要与Service Port合到一起，成为一个有效的通信端口。该IP地址用于**集群内部**的访问。

2. Node IP：集群节点的IP地址。节点可能是物理机，也可能是虚拟机。该地址真实存在于物理网络。**集群外部**，可以通过该地址访问到集群内的节点。

3. Pod IP：K8S创建Pod时，为该Pod分配的IP地址。该地址在集群外不可见。具体的IP地址网段、以及Pod之间通信的方式，取决于集群创建时选用的CNI模型。本文选用了Flannel做为CNI，但不涉及CNI的分析。

**三种端口：**

1. port：是集群Service侦听的端口，与前面的Cluster IP合到一起，即Cluster IP:port，提供了**集群内部**访问Service的入口。在K8S的yaml文件里面，port就是Service Port的缩写。这是K8S创建Service时，默认的方式。
   
   *个人感觉，这个名字其实挺容易让人混淆的，还不如直接用关键字ServicePort来的清楚。*

2. nodePort：是在节点上侦听的端口。通过Node IP:nodePort的形式，提供了**集群外部**访问集群中Service的入口。

3. targetPort：是在Pod上侦听的端口，比如运行Nginx的Pod，会在80端口上监听HTTP请求。所有对Service Port和Node Port的访问，最后都会被转发到Target Port来处理。

下面，开始我们的实验。

---

#### 一. 安装K8S集群

如果还没有集群的话，可以参考我的另一篇文章，先去把环境搭建好：[基于Ubuntu安装Kubernetes集群指南](https://github.com/yuchuanwang/docs/blob/main/Network/Kubernetes_Installation.md)。

#### 二. 创建Deployment

随便找个目录，执行如下命令，创建一个yaml文件：

```shell
$ vi nginx_deployment.yaml
```

输入这些内容后，按:wq保存、退出，得到nginx_deployment.yaml文件：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
```

这个Deployment将会生成2个副本的Pod，每个Pod里面都运行nginx，Pod开放80端口。

然后用该yaml文件，去创建K8S资源：

```shell
$ kubectl apply -f nginx_deployment.yaml 
deployment.apps/nginx-deployment configured
$ kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP           NODE                   NOMINATED NODE   READINESS GATES
nginx-deployment-55f598f8d-49l2v   1/1     Running   0          87m   10.244.0.2   ycwang-ubuntu          <none>           <none>
nginx-deployment-55f598f8d-pxp6x   1/1     Running   0          87m   10.244.1.4   ycwang-ubuntu-worker   <none>           <none>
```

可以看到，两个Pod已经在运行，并且它们有各自的IP。

此时，通过Pod IP即可访问Nginx：

```shell
$ wget 10.244.0.2
Connecting to 10.244.0.2:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 615 [text/html]
Saving to: ‘index.html’

index.html                                              100%[===============================================================================================================================>]     615  --.-KB/s    in 0s      

 (62.7 MB/s) - ‘index.html’ saved [615/615]

$ wget 10.244.1.4
Connecting to 10.244.1.4:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 615 [text/html]
Saving to: ‘index.html.1’

index.html.1                                            100%[===============================================================================================================================>]     615  --.-KB/s    in 0s      

 (84.2 MB/s) - ‘index.html.1’ saved [615/615]
```

但从集群外，是无法访问这两个IP地址的。

#### 三. 创建Service

生成一个新的yaml文件，用来创建Service资源。

```shell
$ vi nginx_svc.yaml
```

输入这些内容后，按:wq保存、退出。

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-svc
  labels:
    run: nginx-svc
spec:
  ports:
  - port: 8080
    targetPort: 80
    protocol: TCP
  selector:
    app: nginx
```

这个Service使用app=nginx标签选择器，来选择对应的Pod，做为Service的后端。Service的类型是默认的Service Port，所以不必写出来。

Service监听在8080端口，集群内部可以通过ClusterIP:8080进行访问，并把流量转发到Pod的80端口进行实际的业务处理。

执行命令，用该yaml文件去创建Service：

```shell
$ kubectl apply -f nginx_svc.yaml 
service/nginx-svc created
$ kubectl get service
NAME         TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
kubernetes   ClusterIP   10.96.0.1        <none>        443/TCP    145m
nginx-svc    ClusterIP   10.104.112.175   <none>        8080/TCP   3s
```

可以看到，该Service已经创建成功。并且出现了一个新的IP地址：10.104.112.175。这就是我们前面介绍的第一种IP地址：Cluster IP。

通过该虚拟的IP地址，加上指定的8080端口，即可实现集群内部对Service的访问：

```shell
$ wget 10.104.112.175:8080
Connecting to 10.104.112.175:8080... connected.
HTTP request sent, awaiting response... 200 OK
Length: 615 [text/html]
Saving to: ‘index.html’

index.html                                              100%[===============================================================================================================================>]     615  --.-KB/s    in 0s      

 (60.4 MB/s) - ‘index.html’ saved [615/615]
```

那么，K8S是如何访问这个不存在的虚拟IP地址的呢？

#### 四. 分析Cluster IP访问流程

先看一下这个Service的信息：

```shell
$ kubectl describe service nginx-svc 
Name:              nginx-svc
Namespace:         default
Labels:            run=nginx-svc
Annotations:       <none>
Selector:          app=nginx
Type:              ClusterIP
IP Family Policy:  SingleStack
IP Families:       IPv4
IP:                10.104.112.175
IPs:               10.104.112.175
Port:              <unset>  8080/TCP
TargetPort:        80/TCP
Endpoints:         10.244.0.2:80,10.244.1.4:80
Session Affinity:  None
Events:            <none>
```

这个Service对应了后端的两个Pod：10.244.0.2:80,10.244.1.4:80。

就是说，对于10.104.112.175:8080的访问，最后会被转发到10.244.0.2:80或者10.244.1.4:80。

这要感谢iptables在后面默默的干活。

查看目前iptables的情况：

```shell
$ sudo iptables-save | grep 10.104.112.175
......
-A KUBE-SERVICES -d 10.104.112.175/32 -p tcp -m comment --comment "default/nginx-svc cluster IP" -m tcp --dport 8080 -j KUBE-SVC-HL5LMXD5JFHQZ6LN
......
```

对于10.104.112.175的访问，会被跳转到规则：KUBE-SVC-HL5LMXD5JFHQZ6LN。

继续查看这条规则：

```shell
$ sudo iptables-save | grep KUBE-SVC-HL5LMXD5JFHQZ6LN
-A KUBE-SVC-HL5LMXD5JFHQZ6LN -m comment --comment "default/nginx-svc -> 10.244.0.2:80" -m statistic --mode random --probability 0.50000000000 -j KUBE-SEP-T5AFCZ323NYPWW2A
-A KUBE-SVC-HL5LMXD5JFHQZ6LN -m comment --comment "default/nginx-svc -> 10.244.1.4:80" -j KUBE-SEP-RQ66ZV5Y2RYOH2X3
```

iptables把对10.104.112.175的访问，采用轮询的负载均衡策略，依次转发给：10.244.0.2:80和10.244.1.4:80。

从而实现了在集群内部对Cluster IP:port的访问，并自带了负载均衡功能。

另外，这些iptables规则的增删改都是由运行在每个节点的kube-proxy来实现的。

#### 五. 创建NodePort类型的Service

现在，我们把Service的类型改成NodePort。

```shell
$ vi nginx_svc.yaml
```

输入如下内容，并按:wq保存、退出：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-svc
  labels:
    run: nginx-svc
spec:
  type: NodePort
  ports:
  - port: 8080
    targetPort: 80
    nodePort: 30000
    protocol: TCP
  selector:
    app: nginx
```

在这个yaml文件，把Service的类型指定为NodePort，并在每个Node上，侦听30000端口。对30000端口的访问，最后会被转发到Pod的80端口。

先把之前的Service和Deployment都删掉，再用新的yaml文件重新创建：

```shell
$ kubectl delete -f nginx_svc.yaml 
service "nginx-svc" deleted
$ kubectl delete -f nginx_deployment.yaml 
deployment.apps "nginx-deployment" deleted
$ kubectl apply -f nginx_deployment.yaml 
deployment.apps/nginx-deployment created
$ kubectl apply -f nginx_svc.yaml 
service/nginx-svc created
```

查看Service和Pod的具体信息：

```shell
$ kubectl get services
NAME         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
kubernetes   ClusterIP   10.96.0.1       <none>        443/TCP          3h41m
nginx-svc    NodePort    10.110.65.131   <none>        8080:30000/TCP   10s
$ kubectl get pods -o wide
NAME                               READY   STATUS    RESTARTS   AGE   IP           NODE                   NOMINATED NODE   READINESS GATES
nginx-deployment-55f598f8d-ls786   1/1     Running   0          11m   10.244.1.6   ycwang-ubuntu-worker   <none>           <none>
nginx-deployment-55f598f8d-pj2xk   1/1     Running   0          11m   10.244.0.3   ycwang-ubuntu          <none>           <none>
```

确认都在正常运行了。

此时，可以在集群内，通过Node IP:NodePort进行访问，此节点的IP是192.168.111.128：

```shell
$ wget 192.168.111.128:30000
Connecting to 192.168.111.128:30000... connected.
HTTP request sent, awaiting response... 200 OK
Length: 615 [text/html]
Saving to: ‘index.html’

index.html                                              100%[===============================================================================================================================>]     615  --.-KB/s    in 0s      

 (87.4 MB/s) - ‘index.html’ saved [615/615]
```

也可以在集群外的Windows机器，通过Node IP:NodePort进行访问：

```shell
$ curl 192.168.111.128:30000
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100   615  100   615    0     0   207k      0 --:--:-- --:--:-- --:--:--  300k
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
html { color-scheme: light dark; }
body { width: 35em; margin: 0 auto;
font-family: Tahoma, Verdana, Arial, sans-serif; }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and
working. Further configuration is required.</p>

<p>For online documentation and support please refer to
<a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at
<a href="http://nginx.com/">nginx.com</a>.</p>

<p><em>Thank you for using nginx.</em></p>
</body>
</html>
```

#### 六. 分析Node IP:NodePort访问流程

我们继续探究，访问Node IP:NodePort是怎么被转到Pod上面去的？

答案依然是iptables：

```shell
$ sudo iptables-save | grep 30000
-A KUBE-NODEPORTS -p tcp -m comment --comment "default/nginx-svc" -m tcp --dport 30000 -j KUBE-EXT-HL5LMXD5JFHQZ6LN
```

对30000端口的访问，会被跳转到规则：KUBE-EXT-HL5LMXD5JFHQZ6LN。

而KUBE-EXT-HL5LMXD5JFHQZ6LN，又被跳转到：KUBE-SVC-HL5LMXD5JFHQZ6LN

```shell
$ sudo iptables-save | grep KUBE-EXT-HL5LMXD5JFHQZ6LN
-A KUBE-EXT-HL5LMXD5JFHQZ6LN -j KUBE-SVC-HL5LMXD5JFHQZ6LN
```

KUBE-SVC-HL5LMXD5JFHQZ6LN这条规则的具体内容：

```shell
$ sudo iptables-save | grep KUBE-SVC-HL5LMXD5JFHQZ6LN
-A KUBE-SERVICES -d 10.110.65.131/32 -p tcp -m comment --comment "default/nginx-svc cluster IP" -m tcp --dport 8080 -j KUBE-SVC-HL5LMXD5JFHQZ6LN
-A KUBE-SVC-HL5LMXD5JFHQZ6LN -m comment --comment "default/nginx-svc -> 10.244.0.3:80" -m statistic --mode random --probability 0.50000000000 -j KUBE-SEP-PU7AOSZG6OVFMASF
-A KUBE-SVC-HL5LMXD5JFHQZ6LN -m comment --comment "default/nginx-svc -> 10.244.1.6:80" -j KUBE-SEP-OZ4KTOWKCOJKYUPL
```

跟Cluster IP的做法一样，iptables把对Node IP:NodePort的访问，采用轮询的负载均衡策略，依次转发给：10.244.0.3:80和10.244.1.6:80这两个Endpoints。

K8S里面的网络访问流程差不多就这样了。它采用了一个很巧妙的设计，去中心化、让每个节点都承担了负载均衡的功能。

---



补充点题外话，在Node IP:NodePort这种模式下，直接访问节点还是会有点问题的。

因为客户需要指定某个Node进行访问。这样会带来单点问题；而且，客户按理不应该知道、也不需要知道具体的Node和它的IP。

所以，在实际应用中，可以在K8S集群外部，搭建一个负载均衡器。客户访问此负载均衡器，再由该负载均衡器把流量分发到各个Node上。很多云厂商也已经带了这样的功能。

但是，既然外部有了支持各种负载均衡算法的职业选手，把流量分发到各个Node上。如果Node收到后，再次用iptables进行负载均衡，就没有什么意义了。不清楚Google为什么要这么设计？

是不是可以考虑在K8S里面内置一个负载均衡的模块，专门运行在某个Node上。在NodePort模式下，可以选择启用该模块，由它来专门提供客户访问的入口并做负载均衡，然后此刻各个Node上的iptables负载均衡可以禁用了？期待各路高人高见…… 

BTW一下，既然都说到了负载均衡，捆绑推销一下我的另一篇文章吧：[负载均衡算法的实现](https://github.com/yuchuanwang/docs/blob/main/Cpp/Cpp_Load_Balance.md)。


