## 基于Ubuntu安装Kubernetes集群指南

Kubernetes(K8S)做为容器编排的系统，被广泛应用于容器、微服务、服务网格中。

想体验、学习K8S的话，首先得把它装好。但由于国内的GFW网络、公司内部局域网的各种限制，安装过程会出现一些坑。故整理一下新鲜出炉的步骤，尽量把坑填上。

在这安装过程中，我准备了两台Ubuntu 22.04的系统，一台做为Master，另一台做为Worker，一起组成一个mini版的集群。

#### 0. 证书问题 - 可选

如果你在公司内部的局域网，而且通过公司提供的代理服务器上网，那么，多半需要此步骤。否则会遇到各种自签名证书验证失败的问题。

此步骤，两台机器都需要做。

具体的证书下载，各个公司都不一样，请咨询你们家的IT人员。下载完之后，执行：update-ca-certificates。

```shell
# 下载你家的自签名证书
$ sudo cp 自签名证书.crt /usr/local/share/ca-certificates/
$ sudo update-ca-certificates
```

这样，可以避免后续的各种证书验证失败。

#### 1. 安装Docker

此步骤，两台机器都需要做。

具体安装步骤请参考：

[Install Docker Engine on Ubuntu | Docker Documentation](https://docs.docker.com/engine/install/ubuntu/)

#### 2. 禁用Swap

此步骤，两台机器都需要做。

分两步，一步时临时禁用，并马上生效；另一步是永久性禁用。

```shell
$ sudo swapoff -a
$ sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab
```

#### 3. 加载内核模块

此步骤，两台机器都需要做。

```shell
$ cat <<EOF | sudo tee /etc/modules-load.d/k8s.conf
overlay
br_netfilter
EOF

$ sudo modprobe overlay
$ sudo modprobe br_netfilter
```

#### 4. 设置sysctl 参数

此步骤，两台机器都需要做。

```shell
$ cat <<EOF | sudo tee /etc/sysctl.d/k8s.conf
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF

sudo sysctl --system
```

#### 5. 安装K8S

此步骤，两台机器都需要做。

```shell
$ sudo apt-get update
$ sudo apt-get install -y apt-transport-https ca-certificates curl
$ sudo curl --insecure -fsSLo https://mirrors.aliyun.com/kubernetes/apt/doc/apt-key.gpg | apt-key add - 
$ sudo cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb https://mirrors.aliyun.com/kubernetes/apt/ kubernetes-xenial main
EOF
$ sudo apt-get update
$ sudo apt-get install -y kubelet kubeadm kubectl
$ sudo apt-mark hold kubelet kubeadm kubectl
```

目前最新版的K8S是1.27.1。这些步骤做完之后，会被安装到两台机器上。

#### 6. 修改containerd配置

此步骤，两台机器都需要做。

主要修改两个地方，一个是从阿里云下载镜像；另一个是启用systemd。

第一个修改是启用systemd：

```shell
$ containerd config default | sudo tee /etc/containerd/config.toml >/dev/null 2>&1
$ sudo sed -i 's/SystemdCgroup \= false/SystemdCgroup \= true/g' /etc/containerd/config.toml
```

第二个修改是把远程下载地址从Google家的改为阿里云的：

```shell
$ sudo vim /etc/containerd/config.toml
```

把这行：sandbox_image = "registry.k8s.io/pause:3.6"

改成：sandbox_image = "registry.aliyuncs.com/google_containers/pause:3.9"

其实用sed，跟第一个systemd的修改一起做了也行。

然后重启containerd：

```shell
$ sudo systemctl restart containerd
```

这一步如果不做的话，后面的kubeadm init会因为无法下载镜像而一直失败。

#### 7. 初始化Master节点

此步骤，仅需要在Master节点上操作。

```shell
sudo kubeadm init --apiserver-advertise-address 192.168.111.128 --pod-network-cidr 10.244.0.0/16 --image-repository registry.aliyuncs.com/google_containers
```

因为后面打算用Flannel来作为CNI，所以CIDR按要求配置了：10.244.0.0/16。

如果一切正常的话，可以看到这样的输出：

```shell
[init] Using Kubernetes version: v1.27.1
[preflight] Running pre-flight checks
[preflight] Pulling images required for setting up a Kubernetes cluster
[preflight] This might take a minute or two, depending on the speed of your internet connection
[preflight] You can also perform this action in beforehand using 'kubeadm config images pull'
W0421 15:26:43.924760    8817 images.go:80] could not find officially supported version of etcd for Kubernetes v1.27.1, falling back to the nearest etcd version (3.5.7-0)
[certs] Using certificateDir folder "/etc/kubernetes/pki"
[certs] Generating "ca" certificate and key
[certs] Generating "apiserver" certificate and key
[certs] apiserver serving cert is signed for DNS names [kubernetes kubernetes.default kubernetes.default.svc kubernetes.default.svc.cluster.local ycwang-ubuntu] and IPs [10.96.0.1 192.168.111.128]
[certs] Generating "apiserver-kubelet-client" certificate and key
[certs] Generating "front-proxy-ca" certificate and key
[certs] Generating "front-proxy-client" certificate and key
[certs] Generating "etcd/ca" certificate and key
[certs] Generating "etcd/server" certificate and key
[certs] etcd/server serving cert is signed for DNS names [localhost ycwang-ubuntu] and IPs [192.168.111.128 127.0.0.1 ::1]
[certs] Generating "etcd/peer" certificate and key
[certs] etcd/peer serving cert is signed for DNS names [localhost ycwang-ubuntu] and IPs [192.168.111.128 127.0.0.1 ::1]
[certs] Generating "etcd/healthcheck-client" certificate and key
[certs] Generating "apiserver-etcd-client" certificate and key
[certs] Generating "sa" key and public key
[kubeconfig] Using kubeconfig folder "/etc/kubernetes"
[kubeconfig] Writing "admin.conf" kubeconfig file
[kubeconfig] Writing "kubelet.conf" kubeconfig file
[kubeconfig] Writing "controller-manager.conf" kubeconfig file
[kubeconfig] Writing "scheduler.conf" kubeconfig file
[kubelet-start] Writing kubelet environment file with flags to file "/var/lib/kubelet/kubeadm-flags.env"
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[kubelet-start] Starting the kubelet
[control-plane] Using manifest folder "/etc/kubernetes/manifests"
[control-plane] Creating static Pod manifest for "kube-apiserver"
[control-plane] Creating static Pod manifest for "kube-controller-manager"
[control-plane] Creating static Pod manifest for "kube-scheduler"
[etcd] Creating static Pod manifest for local etcd in "/etc/kubernetes/manifests"
W0421 15:26:46.894146    8817 images.go:80] could not find officially supported version of etcd for Kubernetes v1.27.1, falling back to the nearest etcd version (3.5.7-0)
[wait-control-plane] Waiting for the kubelet to boot up the control plane as static Pods from directory "/etc/kubernetes/manifests". This can take up to 4m0s
[apiclient] All control plane components are healthy after 5.502175 seconds
[upload-config] Storing the configuration used in ConfigMap "kubeadm-config" in the "kube-system" Namespace
[kubelet] Creating a ConfigMap "kubelet-config" in namespace kube-system with the configuration for the kubelets in the cluster
[upload-certs] Skipping phase. Please see --upload-certs
[mark-control-plane] Marking the node ycwang-ubuntu as control-plane by adding the labels: [node-role.kubernetes.io/control-plane node.kubernetes.io/exclude-from-external-load-balancers]
[mark-control-plane] Marking the node ycwang-ubuntu as control-plane by adding the taints [node-role.kubernetes.io/control-plane:NoSchedule]
[bootstrap-token] Using token: sd27s0.dpnlrf96at6uwgl0
[bootstrap-token] Configuring bootstrap tokens, cluster-info ConfigMap, RBAC Roles
[bootstrap-token] Configured RBAC rules to allow Node Bootstrap tokens to get nodes
[bootstrap-token] Configured RBAC rules to allow Node Bootstrap tokens to post CSRs in order for nodes to get long term certificate credentials
[bootstrap-token] Configured RBAC rules to allow the csrapprover controller automatically approve CSRs from a Node Bootstrap Token
[bootstrap-token] Configured RBAC rules to allow certificate rotation for all node client certificates in the cluster
[bootstrap-token] Creating the "cluster-info" ConfigMap in the "kube-public" namespace
[kubelet-finalize] Updating "/etc/kubernetes/kubelet.conf" to point to a rotatable kubelet client certificate and key
[addons] Applied essential addon: CoreDNS
[addons] Applied essential addon: kube-proxy

Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:

  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config

Alternatively, if you are the root user, you can run:

  export KUBECONFIG=/etc/kubernetes/admin.conf

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.168.111.128:6443 --token sd27s0.dpnlrf96at6uwgl0 \
      --discovery-token-ca-cert-hash sha256:7a6acd******3e154c
```

这样，集群的Master节点基本上好了。

#### 8. 配置Master节点

此步骤，仅需要在Master节点上操作。

根据上面的输出，在Master节点输入命令，使得可以用非root用户操作kubectl：

```shell
$ mkdir -p $HOME/.kube
$ sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
$ sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

继续输入这些命令，实现Shell命令自动补全：

```shell
$ echo 'source <(kubectl completion bash)' >> ~/.bashrc
$ source ~/.bashrc
```

#### 9. 安装Flannel网络

此步骤，仅需要在Master节点上操作。

在Master节点上，安装Flannel：

```shell
$ kubectl apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
```

至此，Master节点的安装、配置都结束了。

下一步，把Worker节点加入集群。

#### 10. Worker加入集群

此步骤，仅需要在Worker节点上操作。

如果你有多台Worker节点，每个结点都需要做一次。

把第7步的join命令Copy出来，然后执行：

```shell
$ sudo kubeadm join 192.168.111.128:6443 --token sd27s0.dpnlrf96at6uwgl0 \
      --discovery-token-ca-cert-hash sha256:7a6acd******3e154c
```

一切正常的话，可以看到这样的输出，表明该机器已经加入集群了：

```shell
[preflight] Running pre-flight checks
[preflight] Reading configuration from the cluster...
[preflight] FYI: You can look at this config file with 'kubectl -n kube-system get cm kubeadm-config -o yaml'
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[kubelet-start] Writing kubelet environment file with flags to file "/var/lib/kubelet/kubeadm-flags.env"
[kubelet-start] Starting the kubelet
[kubelet-start] Waiting for the kubelet to perform the TLS Bootstrap...

This node has joined the cluster:
* Certificate signing request was sent to apiserver and a response was received.
* The Kubelet was informed of the new secure connection details.

Run 'kubectl get nodes' on the control-plane to see this node join the cluster.
```

#### 11. 查看集群信息

回到Master节点，可以看到集群已经建立：

```shell
$ kubectl get nodes
NAME                   STATUS     ROLES           AGE   VERSION
ycwang-ubuntu          NotReady   control-plane   30m   v1.27.1
ycwang-ubuntu-worker   NotReady   <none>          43s   v1.27.1
```

一开始的时候，状态都是NotReady。

稍等片刻，等需要的镜像都下载、运行之后，就会全部变成Ready了。

```shell
$ kubectl get nodes
NAME                   STATUS   ROLES           AGE     VERSION
ycwang-ubuntu          Ready    control-plane   33m     v1.27.1
ycwang-ubuntu-worker   Ready    <none>          3m23s   v1.27.1
```

可以看到所有的Pod都在正常运行了：

```shell
$ kubectl get pods --all-namespaces -o wide
NAMESPACE      NAME                                    READY   STATUS    RESTARTS   AGE     IP                NODE                   NOMINATED NODE   READINESS GATES
kube-flannel   kube-flannel-ds-gl795                   1/1     Running   0          6m10s   192.168.111.128   ycwang-ubuntu          <none>           <none>
kube-flannel   kube-flannel-ds-lq8p8                   1/1     Running   0          6m10s   192.168.111.129   ycwang-ubuntu-worker   <none>           <none>
kube-system    coredns-7bdc4cb885-lgsvw                1/1     Running   0          36m     10.244.1.3        ycwang-ubuntu-worker   <none>           <none>
kube-system    coredns-7bdc4cb885-ss4n5                1/1     Running   0          36m     10.244.1.2        ycwang-ubuntu-worker   <none>           <none>
kube-system    etcd-ycwang-ubuntu                      1/1     Running   0          36m     192.168.111.128   ycwang-ubuntu          <none>           <none>
kube-system    kube-apiserver-ycwang-ubuntu            1/1     Running   0          36m     192.168.111.128   ycwang-ubuntu          <none>           <none>
kube-system    kube-controller-manager-ycwang-ubuntu   1/1     Running   0          36m     192.168.111.128   ycwang-ubuntu          <none>           <none>
kube-system    kube-proxy-wtg74                        1/1     Running   0          6m48s   192.168.111.129   ycwang-ubuntu-worker   <none>           <none>
kube-system    kube-proxy-x4s95                        1/1     Running   0          36m     192.168.111.128   ycwang-ubuntu          <none>           <none>
kube-system    kube-scheduler-ycwang-ubuntu            1/1     Running   0          36m     192.168.111.128   ycwang-ubuntu          <none>           <none>
```

这样，整个集群搭建成功。可以开始使用了。

验证通过后，可以把上述命令合到一起，保存成一个Shell文件。以后每次执行该文件即可。

欢迎来到微服务、云原生的时代！
