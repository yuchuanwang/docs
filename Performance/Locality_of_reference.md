## 性能优化 - 访问局部性

#### 背景

根据[Wiki]([访问局部性 - 维基百科，自由的百科全书](https://zh.m.wikipedia.org/zh-hans/%E8%AE%BF%E9%97%AE%E5%B1%80%E9%83%A8%E6%80%A7))定义：

> 访问局部性（英语：Locality of reference）指的是在计算机科学领域中应用程序在访问内存的时候，倾向于访问内存中较为靠近的值。
> 访问局部性分为三种基本形式，一种是时间局部性，另一种是空间局部性。
> 时间局部性指的是，程序在运行时，最近刚刚被引用过的一个内存位置容易再次被引用，比如在调取一个函数的时候，前不久才调取过的本地参数容易再度被调取使用。
> 空间局部性指的是，最近引用过的内存位置以及其周边的内存位置容易再次被使用。空间局部性比较常见于循环中，比如在一个数列中，如果第3个元素在上一个循环中使用，则本次循环中极有可能会使用第4个元素。
> 第三种为循序区域性。
> 局部性是出现在计算机系统中的一种可预测行为。系统的这种强访问局部性，可以被用来在处理器内核的指令流水线中进行性能优化，如缓存，内存预读取以及分支预测。




在实际应用场景中，比如Intel的CPU，一般会有三级缓存。每个Core有自己的L1和L2缓存；所有的Cores共享L3缓存。
存取速度上，L1快于L2，L2快于L3，而L3又快于内存。当然速度快的代价就是价格的上升。
当CPU需要处理某个数据的时候，它会先从L1缓存中找，如果找不到的话，就依次到L2、L3和内存中找。如果能从L1缓存中直接得到，那么程序的运行速度会大大的快于到内存中去寻址。否则需要从内存中寻址，然后依次进入L3、L2、L1缓存，最后再交给CPU进行处理。
另外当CPU从内存中读取数据到缓存时，每次是以Cache Line的大小为单位来读取的。现在Cache Line一般都是64字节。
利用这些原理，如果我们的代码在处理一个数据A的时候，让后续要被处理的数据B、C、D等，也通过Cache Line一起进入CPU缓存，那么可以预期的是，处理速度将会有明显的提升。

#### 实验

下面通过代码来实验，检查效果。
实验用的机器是一台陈年的笔记本电脑，Intel i3的CPU，两个物理核，Ubuntu 22.04系统。(自从离开了Intel，没机会在土豪的Ice Lake、Sapphire Rapids服务器上跑各种程序了，甚是想念……)
CPU信息如下：

```shell
$ lscpu
Architecture:            x86_64
  CPU op-mode(s):        32-bit, 64-bit
  Address sizes:         39 bits physical, 48 bits virtual
  Byte Order:            Little Endian
CPU(s):                  4
  On-line CPU(s) list:   0-3
Vendor ID:               GenuineIntel
  Model name:            Intel(R) Core(TM) i3-1005G1 CPU @ 1.20GHz
    CPU family:          6
    Model:               126
    Thread(s) per core:  2
    Core(s) per socket:  2
    Socket(s):           1
    Stepping:            5
    CPU max MHz:         3400.0000
    CPU min MHz:         400.0000
    BogoMIPS:            2380.80
    Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rd
                         tscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monit
                         or ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_
                         lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase 
                         tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt intel_pt avx512cd sha_ni avx512bw avx512vl
                          xsaveopt xsavec xgetbv1 xsaves split_lock_detect dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp hwp_pkg_req avx512vbmi umip p
                         ku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid fsrm md_clear flush_l1d arch_capabilities
Virtualization features: 
  Virtualization:        VT-x
Caches (sum of all):     
  L1d:                   96 KiB (2 instances)
  L1i:                   64 KiB (2 instances)
  L2:                    1 MiB (2 instances)
  L3:                    4 MiB (1 instance)
```

在实验之前，需要先把进程允许的栈大小改大，否则会因为栈溢出，得到segment fault的错误。

```shell
$ ulimit -s
8192
```

目前默认的栈是8192 KB (8 MB)。把它放大到128 MB以满足实验要求的10000 * 10000 * char：

```shell
$ ulimit -s 131064
```

**第一个C文件 - Locality_yes.c：**

```c
// Locality_yes.c
// Using Locality

#include <stdio.h>

int main(int argc, char *argv[])
{
    const int rows = 10000;
    const int cols = 10000;
    int sum = 0;
    int val = 10;

    char matrix[rows][cols];
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            // By row
            matrix[row][col] = val;
            sum += val;
        }
    }

    printf("Sum is %d.", sum);
    return 0;
}
```

**第二个C文件 - Locality_no.c**

```c
// Locality_no.c
// Not using Locality

#include <stdio.h>

int main(int argc, char *argv[])
{
    const int rows = 10000;
    const int cols = 10000;
    int sum = 0;
    int val = 10;

    char matrix[rows][cols];
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            // By column
            matrix[col][row] = val;
            sum += val;
        }
    }

    printf("Sum is %d.", sum);
    return 0;
}
```

分别编译这两个文件：

```shell
$ gcc Locality_yes.c -o Locality_yes
$ gcc Locality_no.c -o Locality_no
```

然后分别运行两个程序，并使用time命令对比它们的耗时：

```shell
$ time ./Locality_yes
Sum is 1000000000.
real    0m0.276s
user    0m0.251s
sys     0m0.024s
```

```shell
$ time ./Locality_no
Sum is 1000000000.
real    0m0.472s
user    0m0.444s
sys     0m0.028s
```

可以看到，第二个程序花了0.472秒，而第一个程序只花了0.276秒。差不多**2**倍的差异。经实验，在配置高一些的机器上，会有更大的差异。

而它们之间的区别只在循环的方式。第一个是按行来操作；第二个是按列来操作。

#### 结论

两个操作方式，所带来的差异是由于文章开始所描述的**局部性**。

二维数组在内存中，是逐行保存的，每一行的数据连续保存在内存中，下一行的数据紧跟其后。

第一个程序处理数据的方法，是逐行处理。当第1行、第1列的数据被读取的时候，CPU会把第1行的后续若干个数据同时读取进入缓存，直到Cache Line满了为止。这样下次循环，处理第1行、第2列的数据时，CPU可以直接操作缓存，而不需要再经历内存 -> L3缓存 -> L2缓存 -> L1缓存的过程。

而第二个程序处理数据的方法，是逐列处理。当它处理完第1行、第1列的数据时，下次循环要处理第2行、第1列的数据。而第2行、第1列的数据并不在缓存中，需要重新去内存寻址。每操作一个数，都要从内存读取，直到进入L1缓存。大大增加了操作的延时。

正是因为如上差异，从而导致了前面的实验结果和性能差异。
