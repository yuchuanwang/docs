## C++ 友元与运算符重载那些事

*之前用C++写代码的时候，从没仔细地思考过运算符重载的细节。直到最近复习的时候，才发现这一块知识是自己以前没注意过的。因此记录并分享出来。*



友元函数，friend，该函数与类的成员函数具有相同的访问权限。朋友嘛，所以不管类里面的变量是public、protected还是private，都是可以随便访问的。

运算符重载，operator，让用户自定义的类型支持运算符操作，比如+、-、*、/、=、[]、>>、<<、()等等。使得代码看起来更加自然。比如类里面经常使用的赋值操作，就是重载了=运算符。

那么它们俩有啥关系呢？

我们编个美好的故事，一步步的用代码来说明。

首先，王小二写了个C++的类，代表每个员工的年终奖：

```cpp
#include <iostream>

class Bonus
{
public:
    Bonus(int val = 0)
    {
        count = val;
    }
    ~Bonus() {}

    void Change(int val)
    {
        count += val; 
    }

    int GetCount() const
    {
        return count; 
    }

private:
    int count;
};

int main()
{
    Bonus a;
    a.Change(1000);
    int v = a.GetCount();
    std::cout << v << std::endl;
}


```

成员函数已经可以实现了奖金的变动。但我们往往习惯直接用四则运算，来操作数值相关的事情。

比如说，这个时候，老板需要王小二的代码可以把每个人的奖金加起来，看看到底花了多少钱。于是王小二加了几行代码：

```cpp
int main()
{
    Bonus a;
    a.Change(1000);
    int v = a.GetCount();
    std::cout << v << std::endl;

    Bonus b(500);
    Bonus total = a + b; // Error!
    std::cout << total.GetCount() << std::endl;
}
```

不出意外的，a + b报错了。因为编译器不知道怎么去处理这个Bonus类型的加法。

这个时候，我们需要重载第一个运算符：

```cpp
    Bonus operator+(const Bonus& b) const
    {
        Bonus ret(count + b.count); 
        return ret; 
    }
```

通过这个运算符的重载，**a + b将被翻译成：a.operator+(b)**，计算和，返回一个新的Bonus实例。

类似加减乘除这样的**二元运算符，运算符左侧的a是调用方，运算符右侧的b是作为参数被传递进去。**

实现了这个之后，完成了老板的任务，老板很开心：小伙子干的不错，给你的奖金翻番吧：

```cpp
    Bonus c = a * 2.0; // Error!
    std::cout << c.GetCount() << std::endl;
```

然后，王小二发现又报错了。原来乘法运算符还没重载。趁老板没注意，赶紧加上去：

```cpp
    Bonus operator*(double times) const
    {
        Bonus ret(count * times); 
        return ret; 
    }
```

阿弥陀佛，这回终于可以做乘法了。

老板欣赏的看着王小二，他决定自己亲自改代码，把王小二的奖金从2被变成3倍：

```cpp
    Bonus d = 3.0 * a; 
    std::cout << d.GetCount() << std::endl;
```

然后，又错了…… 

难道乘法分配律在这里失效了？a * 2可以，但3 * a不可以？王小二同学一脸问号。

按照前面的描述：“二元运算符，运算符左侧是调用方，运算符右侧是作为参数被传递进去。” 现在左侧是double，难道王小二需要去改double类型的代码，让它认识Bonus这个类型吗……

可以使用非Bonus类成员函数的运算符重载，王小二急中生智，非成员函数不需要依赖对象进行调用。于是，在Bonus类的外面，王小二加了这个函数：

```cpp
Bonus operator*(double times, const Bonus& b) 
{
    Bonus ret(times * b.count); // Error!
    return ret; 
}
```

但是，编译器继续报错。因为count是类的私有成员，非成员函数不能访问。咋办？

这时，我们的另一个主角：友元终于可以登场了：

通过把这个非成员函数声明为Bonus的朋友，该函数就可以访问Bonus的内部数据了。

在Bonus类的声明部分(头文件)，加上这么一行：

```cpp
friend Bonus operator*(double times, const Bonus& b);
```

终于圆满的编译、运行通过了。

完整的代码如下：

```cpp
#include <iostream>

class Bonus
{
public:
    Bonus(int val = 0)
    {
        count = val;
    }
    ~Bonus() {}

    void Change(int val)
    {
        count += val; 
    }

    int GetCount() const
    {
        return count; 
    }

    Bonus operator+(const Bonus& b) const
    {
        Bonus ret(count + b.count); 
        return ret; 
    }

    Bonus operator*(double times) const
    {
        Bonus ret(count * times); 
        return ret; 
    }

    friend Bonus operator*(double times, const Bonus& b);

private:
    int count;
};

Bonus operator*(double times, const Bonus& b) 
{
    Bonus ret(times * b.count);
    return ret; 
}

int main()
{
    Bonus a;
    a.Change(1000);
    int v = a.GetCount();
    std::cout << v << std::endl;

    Bonus b(500);
    Bonus total = a + b;
    std::cout << total.GetCount() << std::endl;

    Bonus c = a * 2.0; 
    std::cout << c.GetCount() << std::endl;

    Bonus d = 3.0 * a; 
    std::cout << d.GetCount() << std::endl;
}
```

所以，**在为类Bonus重载运算符时，如果第一项操作数是Bonus类，那么把运算符定义为普通的类成员函数即可；如果第一项操作数并非Bonus类，我们则需要使用友元函数来翻转操作数的顺序**。

至此，王小二同学终于圆满的完成了任务，并且拿到了3倍的奖金！


