## Kubernetes Device Plugin开发简介

在K8S的集群中，K8S默认会根据CPU和内存的资源情况进行调度。对于其它的设备资源，它就不认识、也无法调度了。

因此，各个设备厂商针对自己的设备，纷纷开发了各种各样的Device Plugin，然后注册到每个节点的kubelet上去，使得K8S的调度器能够通过各个kubelet获取设备信息。然后在创建Pod的时候，根据各个节点上报的设备信息，把Pod调度到合适的节点上去运行。

比如，针对AI训练和推理所需要的GPU/NPU，为了能让K8S识别、调度，NVIDIA、AMD和华为分别为各自的设备开发了这些Device Plugin：

[GitHub - NVIDIA/k8s-device-plugin: NVIDIA device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin)

[GitHub - ROCm/k8s-device-plugin: Kubernetes (k8s) device plugin to enable registration of AMD GPU to a container cluster](https://github.com/ROCm/k8s-device-plugin)

https://github.com/Ascend/ascend-device-plugin

本文将介绍如何开发一个基本的Device Plugin，在节点上构造一个虚拟的NPU设备，并使得该设备可以在创建Pod的时候被K8S识别和调度。

一个典型的Device Plugin的工作流程是这样的：

- 注册设备：插件调用Kubelet的gRPC接口：Register(RegisterRequest)，向其注册设备。需要发送插件的Unix Socket、版本、ResourceName 

- 分配设备：插件必须实现以下两个gRPC接口。
  
  返回节点上的设备列表：rpc ListAndWatch(Empty) returns (stream ListAndWatchResponse) {}
  
  创建容器时，分配、绑定设备：rpc Allocate(AllocateRequest) returns (AllocateResponse) {}

所以开发设备插件的主要工作也是在于：

- 如何调用Register接口去注册设备；

- 如何实现ListAndWatch函数去上报本节点的设备情况；

- 当在本节点创建Pod时，如何实现Allocate函数去分配设备。

以下是详细步骤。



#### 1. 创建Go项目

创建一个dummy_npu的项目：

```shell
$ mkdir dummy_npu
$ cd dummy_npu
$ go mod init dummy_npu
```



#### 2. 编辑Go代码

新建dummy.go文件，并输入以下代码：

```go
package main

import (
	"fmt"
	"net"
	"os"
	"os/signal"
	"path"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"

	pluginAPI "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	connectTimeout = 5 * time.Second
	devicesCount   = 64
	resourceName   = "dummy.com/npu"
	//pluginSock     = "/var/lib/kubelet/device-plugins/dummy.sock"
	pluginSock = "dummy.sock"
)

type DummyDevicePlugin struct {
	devices map[string]*pluginAPI.Device
	socket  string
	server  *grpc.Server
	mutex   sync.Mutex
	sigs    chan os.Signal
}

// Create DummyDevicePlugin
func NewDevicePlugin() *DummyDevicePlugin {
	dp := &DummyDevicePlugin{
		devices: make(map[string]*pluginAPI.Device),
		socket:  path.Join(pluginAPI.DevicePluginPath, pluginSock),
	}
	dp.init()
	return dp
}

// Init initialize the device plugin
func (dp *DummyDevicePlugin) init() error {
	fmt.Println("Initializing device plugin")
	dp.devices = make(map[string]*pluginAPI.Device, devicesCount)
	for i := 0; i < devicesCount; i++ {
		name := strconv.Itoa(i)
		dev := &pluginAPI.Device{
			ID:     name,
			Health: pluginAPI.Healthy,
		}
		dp.devices[name] = dev
	}

	// Wait for signals
	dp.sigs = make(chan os.Signal, 1)
	signal.Notify(dp.sigs, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM, syscall.SIGQUIT)

	return nil
}

// discoverDevices get device list
// Monitor and update per health check
func (dp *DummyDevicePlugin) discoverDevices() map[string]*pluginAPI.Device {
	healthyDevices := make(map[string]*pluginAPI.Device)
	for _, dev := range dp.devices {
		if dev.Health == pluginAPI.Healthy {
			healthyDevices[dev.ID] = dev
		}
	}

	fmt.Println("Healthy devices found:", len(healthyDevices))
	return healthyDevices
}

// Start starts the gRPC server of device plugin
func (dp *DummyDevicePlugin) Start() error {
	err := dp.cleanup()
	if err != nil {
		return fmt.Errorf("failed to clean existing socket file")
	}

	listen_sock, err := net.Listen("unix", dp.socket)
	if err != nil {
		return fmt.Errorf("failed to listen on plugin socket")
	}

	dp.server = grpc.NewServer([]grpc.ServerOption{}...)
	pluginAPI.RegisterDevicePluginServer(dp.server, dp)
	go dp.server.Serve(listen_sock)
	fmt.Println("Device plugin gRPC server begins to serve at:", dp.socket)

	// Wait for server to start by launching a blocking connection
	conn, err := grpc.Dial(dp.socket, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithTimeout(connectTimeout),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)
	if err != nil {
		return fmt.Errorf("failed to dial to plugin socket")
	}
	conn.Close()

	go dp.healthCheck()

	fmt.Println("Device plugin gRPC server is ready")

	return nil
}

// Stop stops the gRPC server
func (dp *DummyDevicePlugin) StopServer() error {
	if dp.server == nil {
		return nil
	}

	dp.server.Stop()
	dp.server = nil

	fmt.Println("Device plugin gRPC server is stopped")

	return dp.cleanup()
}

// healthCheck
// TODO: monitor and update devices
func (dp *DummyDevicePlugin) healthCheck() error {
	for {
		time.Sleep(60 * time.Second)
	}
}

func (dp *DummyDevicePlugin) cleanup() error {
	if err := os.Remove(dp.socket); err != nil && !os.IsNotExist(err) {
		return err
	}

	return nil
}

func (dp *DummyDevicePlugin) exit() {
	dp.sigs <- syscall.SIGTERM
}

// Register with kubelet
func Register() error {
	conn, err := grpc.Dial(pluginAPI.KubeletSocket, grpc.WithInsecure(),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}))
	if err != nil {
		return fmt.Errorf("failed to connect to kubelet: %v", err)
	}
	defer conn.Close()

	client := pluginAPI.NewRegistrationClient(conn)
	req := &pluginAPI.RegisterRequest{
		Version: pluginAPI.Version,
		// Name of socket that device plugin is listening
		Endpoint:     pluginSock,
		ResourceName: resourceName,
	}

	_, err = client.Register(context.Background(), req)
	if err != nil {
		return fmt.Errorf("failed to register to kubelet: %v", err)
	}

	fmt.Println("Device plugin registers to kubelet")
	return nil
}

// ListAndWatch returns a stream of List of Devices
// Whenever a device state change or a device disappears, ListAndWatch returns the new list
// lists devices and update that list according to the health status
func (dp *DummyDevicePlugin) ListAndWatch(emtpy *pluginAPI.Empty, stream pluginAPI.DevicePlugin_ListAndWatchServer) error {
	fmt.Println("ListAndWatch starts")
	for {
		// Build response
		resp := new(pluginAPI.ListAndWatchResponse)
		healthyDevices := dp.discoverDevices()
		for _, dev := range healthyDevices {
			resp.Devices = append(resp.Devices, dev)
		}

		// Send response
		//fmt.Println("ListAndWatch sends devices")
		if err := stream.Send(resp); err != nil {
			fmt.Println("Failed to send devices to kubelet:", err)
			// FIXME: Something is wrong when sending devices to kubelet
			// How about restart this device plugin
			fmt.Println("Since it is failed to communicate with kubelet, let's restart device plugin")
			dp.exit()
		}

		time.Sleep(10 * time.Second)
	}
}

// Allocate is called during container creation, so that the Device Plugin can run device specific operations
// and instruct Kubelet of the steps to make the device available in the container
func (dp *DummyDevicePlugin) Allocate(ctx context.Context, reqs *pluginAPI.AllocateRequest) (*pluginAPI.AllocateResponse, error) {
	// Get unallocated and healthy device
	fmt.Println("Allocate starts")
	ret := pluginAPI.AllocateResponse{}
	fmt.Println("Recv request:", reqs.ContainerRequests)
	for _, req := range reqs.ContainerRequests {
		fmt.Println("Recv request DevicesIDs:", req.DevicesIDs)
		// Discover healthy devices
		healthyDevices := dp.discoverDevices()
		if len(healthyDevices) < len(req.DevicesIDs) {
			fmt.Println("Number of available devices is less than request devices:", len(healthyDevices), len(req.DevicesIDs))
			return nil, fmt.Errorf("invalid allocate request of devices count: %d", len(req.DevicesIDs))
		}

		// Allocate healthy devices, and change allocated devices to unhealthy
		dp.mutex.Lock()
		var ids []string
		device_allocated := 0
		for _, dev := range healthyDevices {
			ids = append(ids, dev.ID)
			dp.devices[dev.ID].Health = pluginAPI.Unhealthy

			device_allocated++
			if device_allocated >= len(req.DevicesIDs) {
				break
			}
		}
		dp.mutex.Unlock()

		// For NV, it passes devices to ENV NVIDIA_VISIBLE_DEVICES
		fmt.Println("Allocate devices:", ids)
		resp := pluginAPI.ContainerAllocateResponse{
			Envs: map[string]string{"DUMMY_VISIBLE_DEVICES": strings.Join(ids, ",")},
		}
		ret.ContainerResponses = append(ret.ContainerResponses, &resp)
	}
	return &ret, nil
}

// GetPreferredAllocation returns a preferred set of devices to allocate from a list of available ones.
// The resulting preferred allocation is not guaranteed to be the allocation ultimately performed by the devicemanager.
// It is only designed to help the devicemanager make a more informed allocation decision when possible.
func (dp *DummyDevicePlugin) GetPreferredAllocation(_ context.Context, _ *pluginAPI.PreferredAllocationRequest) (*pluginAPI.PreferredAllocationResponse, error) {
	return &pluginAPI.PreferredAllocationResponse{}, nil
}

// GetDevicePluginOptions returns options to be communicated with Device Manager
func (dp *DummyDevicePlugin) GetDevicePluginOptions(context.Context, *pluginAPI.Empty) (*pluginAPI.DevicePluginOptions, error) {
	return &pluginAPI.DevicePluginOptions{}, nil
}

// PreStartContainer is called, if indicated by Device Plugin during registeration phase, before each container start.
// Device plugin can run device specific operations such as reseting the device before making devices available to the container
func (dp *DummyDevicePlugin) PreStartContainer(context.Context, *pluginAPI.PreStartContainerRequest) (*pluginAPI.PreStartContainerResponse, error) {
	return &pluginAPI.PreStartContainerResponse{}, nil
}

func main() {
	dp := NewDevicePlugin()

	// Start grpc server
	err := dp.Start()
	if err != nil {
		fmt.Println("Failed to start device plugin:", err)
	}
	fmt.Println("Start to server at:", dp.socket)

	// Registers with Kubelet.
	err = Register()
	if err != nil {
		fmt.Println("Failed to register device plugin:", err)
	}
	fmt.Println("Device plugin is registered")

	// TODO: Watch kubelet sock file
	//err = dp.watchKubelet()
	//if err != nil {
	//	fmt.Println("Failed to watch kubelet:", err)
	//}

	s := <-dp.sigs
	fmt.Println("Receive signal and will exit:", s)

	dp.StopServer()
}

```



这些代码里面，主要的过程和文章一开始描述的是一样的。

程序启动的时候，首先启动一个gRPC的Server、并通过RegisterDevicePluginServer使其成为一个Device Plugin专用的server，这样kubelet可以通过连接这个Server获取设备信息；

第二，通过NewRegistrationClient，创建一个kubelet的客户端，去连接kubelet并注册新的设备"dummy.com/npu"；

第三，实现DevicePluginServer这个gRPC服务所需要的接口，关键是ListAndWatch和Allocate两个必选的接口，其它的三个接口GetPreferredAllocation、GetDevicePluginOptions、PreStartContainer可以不实现。





#### 3. 编辑Dockerfile

代码写完之后，需要把它编译、并得到一个Docker的镜像文件，以供K8S调度、创建Pod。

新建Dockerfile文件，并输入以下内容：

```dockerfile
FROM golang:1.21 AS builder
WORKDIR /app
COPY go.mod ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o dummy-device-plugin .

FROM alpine:latest
COPY --from=builder /app/dummy-device-plugin /usr/local/bin/dummy-device-plugin
ENTRYPOINT ["/usr/local/bin/dummy-device-plugin"]
```



在这个文件里，把Go代码编译成一个可执行文件dummy-device-plugin，并把它复制到/usr/local/bin/目录。容器启动时，会自动运行/usr/local/bin/dummy-device-plugin





#### 4. 编译和导入Docker镜像

现在可以编译Docker镜像了。执行命令：

```shell
$ docker build -t docker.io/ycwang/dummy_npu:latest .
```



可以得到一个名为dummy_npu的镜像文件。

但是还不能直接使用。

因为在新的K8S里面，默认的容器运行时不再是Docker，而是containerd了。在Docker里面的镜像，并不能被containerd所识别。所以，需要把dummy_npu这个镜像文件导出、再导入到containered去：

```shell
$ docker save ycwang/dummy_npu -o dummy_npu.tar
$ sudo ctr -n k8s.io images import dummy_npu.tar
```

这样，就能在containerd里面看到dummy_npu这个镜像了。

后面如果对代码做了修改，也需要重复这三个步骤，重新编译、导出、导入。



#### 5. 编辑daemonset yaml文件

现在可以创建yaml文件，让K8S去调度管理了。

首先需要新建文件dummy_npu_daemonset.yaml，并输入以下内容：

```yaml
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dummy-device-plugin
  namespace: kube-system
  labels:
    app: dummy-device-plugin
spec:
  selector:
    matchLabels:
      app: dummy-device-plugin
  template:
    metadata:
      labels:
        app: dummy-device-plugin
    spec:
      #nodeSelector:
      #  dummy-device-plugin: 'true'
      hostNetwork: true
      containers:
      - image: docker.io/ycwang/dummy_npu:latest
        imagePullPolicy: Never
        name: dummy-device-plugin
        volumeMounts:
          - name: device-plugin
            mountPath: /var/lib/kubelet/device-plugins
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins


```



通过应用这个文件，K8S会在节点上创建daemonset，使得节点上保持一份程序始终在运行。一般来说，Device Plugin都是以Daemon的方式在后台运行的。



#### 6. 创建Pod yaml文件

现在，创建一个dummy_npu_pod.yaml文件，在这个Pod里面，我们可以去声明需要使用该Device Plugin所提供的设备

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dummy-npu-pod
spec:
  containers:
    - name: dummy-npu-container
      image: busybox
      imagePullPolicy: Never
      command: ["sh", "-c", "echo $DUMMY_DEVICES && sleep 3600"]
      resources:
        limits:
          dummy.com/npu: 2


```



这个Pod里面，启动了一个busybox的镜像，并且声明需要使用8个假的NPU。

K8S在创建这个Pod的时候，会在集群里寻找具有可用Dummy NPU的节点，并在该节点上创建这个Pod。





#### 7. 创建K8S资源

代码的工作都完成了，现在可以试着去看看结果了。

输入以下命令，分别创建daemonset和pod：

```shell
$ kubectl apply -f dummy_npu_daemonset.yaml
$ kubectl apply -f dummy_npu_pod.yaml
```





#### 8. 查看效果

一切正常的话，可以看到节点上多了dummy NPU的信息：

```shell
$ kubectl describe nodes

......
Capacity:
  cpu:                4
  dummy.com/npu:      64
  ephemeral-storage:  203800640Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             8086140Ki
  pods:               110
......
```



其中的"dummy.com/npu:      64"就是这个Device Plugin提供的设备，每个节点上提供64个。



在Go代码中，当Allocate被调用的时候，我们在环境变量DUMMY_VISIBLE_DEVICES里面写了点信息，可以验证一下。

进入busybox的pod：

```shell
$ kubectl exec dummy-npu-pod -it sh
```

看看该pod的环境变量：

```shell
# echo $DUMMY_VISIBLE_DEVICES
31,48
```



可以看到，该pod成功的得到了2个NPU的设备。这样，就可以在Pod里面使用这些设备了。




