## 细胞分裂问题的原创解法

这几天，应公司要求，需要做一些算法题。看到一道挺有趣的题目，把整个思考过程整理记录下来，感觉应该是原创的。

如有雷同，纯属巧合(或者说，英雄所见略同，嘿嘿)



题目是这样的：

> 细胞分裂
> 有一个细胞，每一个小时分裂一次，一次分裂一个子细胞，第三个小时后会死亡。那么n个小时后，共有多少活着的细胞？

看上去，第一直觉是个递归问题。但是怎么进行递归呢？经过草稿，思路是这样的：

每个细胞有4个状态，A代表初始状态，B代表一小时后的状态，C代表二小时后的状态，D代表三小时后的状态 - 死亡。

每个细胞都会经历A -> B -> C -> D的状态转换；当它处于A、B、C状态的时候，它会分裂出新的细胞，而分裂出来的新细胞也会一样经历A -> B -> C -> D的状态转换。

细胞状态与个数的过程，可以用这个图来表示：

![Cells](https://github.com/yuchuanwang/docs/blob/main/Assets/Cells.jpg)



所以，我们可以按照每个状态来推导细胞的总数：

D的状态直接无视了，因为我们只考虑活着的那些；

C由B转化而来，所以在第n小时的时候，C的个数等于前一个小时B的个数：

```
C(n) = B(n-1)
```

B由A转化而来，所以在第n小时的时候，B的个数等于前一个小时A的个数：

```
B(n) = A(n-1)
```

所有A、B、C状态的细胞，都会分裂出新的细胞并处于A状态，所以A的个数等于前一个小时A、B、C的个数之和：

```
A(n) = A(n-1) + B(n-1) + C(n-1)
```

那么在第n个小时的时候，所有活着的细胞总数，就是A、B、C状态的总和：

```
All(n) = A(n) + B(n) + C(n)
```

至此，基本的推导过程完成，也得到了递归的公式。

代码如下：

```cpp
// CellsSlow.cpp
#include <iostream>

int Cells_A(unsigned int n);
int Cells_B(unsigned int n);

int Cells_C(unsigned int n)
{
    if ((n == 0) || (n == 1))
    {
        return 0;
    }

    return Cells_B(n-1);
}

int Cells_B(unsigned int n)
{
    if (n == 0)
    {
        return 0;
    }
    return Cells_A(n-1);
}

int Cells_A(unsigned int n)
{
    if (n == 0)
    {
        return 1;
    }

    return Cells_A(n-1) + Cells_B(n-1) + Cells_C(n-1);
}

int Cells_All(unsigned int n)
{
    return Cells_A(n) + Cells_B(n) + Cells_C(n);
}

int main()
{
    for (int i = 0; i <= 30; i++)
    {
        int ans = Cells_All(i);
        std::cout<< i << " -> " << ans << std::endl;
    }

    return 0;
}
```

但可想而知，这个递归是非常非常慢的，存在大量的重复计算，比常见的斐波那契数列f(n) = f(n-1) + f(n-2)还慢得多，运算量呈指数级增长。



那么，怎么优化呢？

思路一，是把前面A、B、C的计算结果缓存下来，减少后面的重复计算。

思路二，是把递归公式简化。

以下是第二个思路的推导过程，把A(n)、B(n)、C(n)的值，代入All(n)的表达式：

```
A(n) = A(n-1) + B(n-1) + C(n-1)
B(n) = A(n-1)
C(n) = B(n-1)
All(n) = A(n) + B(n) + C(n)
= A(n-1) + B(n-1) + C(n-1) + A(n-1) + B(n-1)
= A(n-1) + B(n-1) + C(n-1) + A(n-1) + B(n-1) + C(n-1) - C(n-1)
= 2 * All(n-1) - C(n-1)
```

上面的最后两步是重点，通过加、减一个C(n-1)，我们把All(n)跟前一个小时的总数All(n-1)成功关联起来了。现在就看如何把多余的那个尾巴 -C(n-1)给消除掉了。

```
C(n-1) = B(n-1-1) = A(n-1-1-1) = A(n-3) = A(n-4) + B(n-4) + C(n-4)
```

然后根据All(n)的等式：

```
All(n) = A(n) + B(n) + C(n)  =>
All(n-4) = A(n-4) + B(n-4) + C(n-4)
```

可以发现，C(n-1)和All(n-4)其实是一样的。所以，All(n)的等式可以推导为：

```
All(n) = 2 * All(n-1) - C(n-1) = 2 * All(n-1) - All(n-4)
```

现在，我们得到了一个很精简的递推公式，第n个小时的总数，只取决于第n-1，和第n-4小时的个数(**动态规划**的思路出来了)。

加上一些边界条件，我们就可以写出这个程序了：

```
// CellsFast.cpp
#include <iostream>
#include <vector>

std::vector<int> Cells(unsigned int n)
{
    // 0 -> 1
    // 1 -> 2
    // 2 -> 4
    // 3 -> 7
    if (n == 0)
    {
        std::vector<int> ret = {1};
        return ret;
    }
    else if(n == 1)
    {
        std::vector<int> ret = {1, 2};
        return ret;
    }
    else if(n == 2)
    {
        std::vector<int> ret = {1, 2, 4};
        return ret;
    }
    else if(n == 3)
    {
        std::vector<int> ret = {1, 2, 4, 7};
        return ret;
    }
    else
    {
        std::vector<int> ret;
        ret.push_back(1);
        ret.push_back(2);
        ret.push_back(4);
        ret.push_back(7);
        for (int i = 4; i <= n; i++)
        {
            // All(n) = 2 * All(n-1) - All(n-4)
            int ans = 2 * ret[i-1] - ret[i-4];
            ret.push_back(ans);
        }

        return ret;
    }
}

int main()
{
    std::vector<int> ret = Cells(30);
    int index = 0;
    for (auto iter = ret.begin(); iter != ret.end(); ++iter, index++)
    {
        std::cout<< index << " -> " << *iter << std::endl;
    }
    return 0;
}
```

根据上面的推导结论，用Excel简单的做了个表格+公式，可以得到下面的数据：

| 第n小时 | 细胞总数       |
|:----:|:----------:|
| 0    | 1          |
| 1    | 2          |
| 2    | 4          |
| 3    | 7          |
| 4    | 13         |
| 5    | 24         |
| 6    | 44         |
| 7    | 81         |
| 8    | 149        |
| 9    | 274        |
| 10   | 504        |
| 11   | 927        |
| 12   | 1,705      |
| 13   | 3,136      |
| 14   | 5,768      |
| 15   | 10,609     |
| 16   | 19,513     |
| 17   | 35,890     |
| 18   | 66,012     |
| 19   | 121,415    |
| 20   | 223,317    |
| 21   | 410,744    |
| 22   | 755,476    |
| 23   | 1,389,537  |
| 24   | 2,555,757  |
| 25   | 4,700,770  |
| 26   | 8,646,064  |
| 27   | 15,902,591 |
| 28   | 29,249,425 |
| 29   | 53,798,080 |
| 30   | 98,950,096 |

程序的输出如下：

```
0 -> 1
1 -> 2
2 -> 4
3 -> 7
4 -> 13
5 -> 24
6 -> 44
7 -> 81
8 -> 149
9 -> 274
10 -> 504
11 -> 927
12 -> 1705
13 -> 3136
14 -> 5768
15 -> 10609
16 -> 19513
17 -> 35890
18 -> 66012
19 -> 121415
20 -> 223317
21 -> 410744
22 -> 755476
23 -> 1389537
24 -> 2555757
25 -> 4700770
26 -> 8646064
27 -> 15902591
28 -> 29249425
29 -> 53798080
30 -> 98950096
```

两者是一致的。

搞定！
