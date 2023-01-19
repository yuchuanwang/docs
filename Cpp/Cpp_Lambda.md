## C++ lambda表达式

lambda表达式，也称为匿名函数、闭包函数，在别的编程语言很早就有了。

C++ 11开始，也支持了这个功能。而后续的C++ 版本又陆陆续续做了些改进。

整理了编笔记，把lambda表达式的用法试验、记录一下。

lambda表达式的语法如下：

```cpp
[ captures ] ( params ) specs -> return-type { body }
```

- captures：捕获列表，用来捕获当前作用域的变量，然后可以在lambda表达式内部使用。支持按值捕获、按引用捕获。用法和行为，跟普通的C++函数调用很像。
  
- params：参数列表，可选。在调用lambda表达式时，额外传递的参数。
  
- specs：限定符，可选。比如说mutable，后面的试验会用到它。在后续的C++ 17、20、23等版本，还添加了constexpr、consteval、static这些。
  
- return-type：返回类型，可选。
  
- body：函数体
  

整个看起来，跟普通的C++函数很像：

- lambda的捕获列表 + 参数列表 <--> 函数的参数列表
  
- lambda的限定符 <--> 函数的限定符
  
- lambda的返回类型 <--> 函数的返回类型
  
- lambda的函数体 <--> 函数的函数体
  

就是函数的函数名，在lambda里面不需要定义。所以也把lambda表达式称为匿名函数。

这是一个简单的lambda表达式例子：

```cpp
#include <iostream>

int main()
{
    int x = 1;
    // Define a simple lambda
    auto add_func = [x] (int y) {
        return x + y; 
    }; 

    // Call lambda
    int ans = add_func(10);
    std::cout << ans << std::endl; 
    return 0; 
}
```

运行结果很浅白，会输出：11

add_func表达式在定义时，把作用域的x变量捕获，然后其函数体内使用x、和额外传递的参数y进行运算，最后返回计算结果。

那么我们把代码稍作修改：

```cpp
#include <iostream>

int main()
{
    int x = 1;
    // Define a simple lambda
    auto add_func = [x] (int y) {
        return x + y; 
    }; 

    // Call lambda
    int ans = add_func(10);
    std::cout << ans << std::endl; 

    // Modify x outside
    x = 2;
    // Call lambda again
    ans = add_func(10);
    std::cout << ans << std::endl; 

    return 0; 
}
```

在第一次调用add_func之后，我们把外部的x变量修改了，然后，再次调用add_func。

运行这个代码，会发现输出：

11

11

外部x变量的修改，并没有传递到lambda内部。

这是因为：C++在编译期间，编译器自动为lambda表达式生成一个闭包ClosureType类。在lambda表达式被定义的地方，实例化该类，生成实例add_func，并对被其捕获的成员变量进行赋值：

```cpp
add_func.__x = x
```

所以，**按值捕获的变量，在lambda定义时，它在lambda内部的值已经被确定下来。后续外部对变量x的修改，不会再影响到lambda内部的__x。**

那么，如果需要修改按值捕获的变量，应该怎么做呢？修改完以后，lambda内外的变量会发生什么变化呢？

```cpp
#include <iostream>

int main()
{
    int x = 1;    
    // Define lambda to modify value captured
    auto modify_func = [x] () {
        x++; 
    };

    return 0; 
}
```

像这样，直接对按值捕获的变量进行修改，编译器会报错：

```cpp
error: increment of read-only variable 'x'
         x++;
```

需要用到一开始说的mutable限定符，改为这样就可以了：

```cpp
auto modify_func = [x] () mutable {
```

加上一些输出信息之后，代码变成了这样：

```cpp
#include <iostream>

int main()
{
    int x = 1;    
    // Define lambda to modify value captured
    auto modify_func = [x] () mutable {
        std::cout << "x inside lambda is: " << x << std::endl;
        x++; 
    };

    std::cout << "Before calling lambda, x out of lambda is: " << x << std::endl;
    modify_func();
    std::cout << "After calling lambda, x out of lambda is: " << x << std::endl;    
    modify_func();
    std::cout << "After calling lambda again, x out of lambda is: " << x << std::endl;

    return 0; 
}
```

运行这段代码，可以得到这些输出：

```cpp
Before calling lambda, x out of lambda is: 1
x inside lambda is: 1
After calling lambda, x out of lambda is: 1 
x inside lambda is: 2
After calling lambda again, x out of lambda is: 1
```

这里可以看出两个信息：

1. 按值捕获之后，lambda内外的变量已经没有关系，各自有各自的数值。
  
2. **修改lambda实例的成员变量之后，该修改会一直生效，直到lambda实例的生命周期结束。**
  

第一点信息，前面已经解释过。第二点信息，跟第一点信息的原理也密切相关。

可以这么理解，闭包ClosureType类的实例modify_func，根据捕获的变量，内部相应创建了成员变量__x。Lambda内部的x++，其实是modify_func.__x++。所以，下次再次调用modify_func时，其成员变量__x保留了上次调用的数值。

以上两点信息，只要理解了lambda表达式其实是个ClosureType类，由编译器根据捕获的变量，自动生成对应的成员变量。然后在lambda表达式定义的地方被实例化、初始化成员变量。而后的lambda表达式调用，本质上是调用了该实例的成员函数。那么这些行为就很自然而然了。

接下来，按引用捕获变量。

按引用捕获，用法、行为跟普通函数的按引用传递没什么区别。只需要在捕获的变量前，加上&符号即可。

```cpp
#include <iostream>

int main()
{
    int x = 1;
    // Define lambda to capture by reference
    auto ref_func = [&x] () {
        std::cout << "x inside lambda is: " << x << std::endl;
        x++; 
    };

    std::cout << "Before calling lambda, x out of lambda is: " << x << std::endl;
    ref_func();
    std::cout << "After calling lambda, x out of lambda is: " << x << std::endl;  

    x = 5; 
    std::cout << "Now change x out of lambda to: " << x << std::endl; 
    ref_func(); 
    std::cout << "After calling lambda again, x out of lambda is: " << x << std::endl;

    return 0; 
}
```

x变成&x，**变成了按引用捕获变量，之后，lambda内部和外部，共享同一个变量。一方的修改，将反应到另一方上面。**

所以，上面的代码将输出：

```cpp
Before calling lambda, x out of lambda is: 1
x inside lambda is: 1
After calling lambda, x out of lambda is: 2
Now change x out of lambda to: 5
x inside lambda is: 5
After calling lambda again, x out of lambda is: 6
```

另外，如果需要按值捕获外部的所有变量，通过[=]即可。

而通过[&]，可以按引用捕获外部的所有变量。

最后一点，针对外部的全局变量或者局部static变量，可以在lambda表达式内部直接使用、修改；内外共享一个变量。比如下面的代码：

```cpp
#include <iostream>

int global_val = 1;

int main()
{
    // Define lambda to use global param
    auto global_func = [] () {
        std::cout << "global_val inside lambda is: " << global_val << std::endl;
        global_val++; 
    };

    std::cout << "Before calling lambda, global_val out of lambda is: " << global_val << std::endl;
    global_func();
    std::cout << "After calling lambda, global_val out of lambda is: " << global_val << std::endl;  

    global_val = 5; 
    std::cout << "Now change global_val out of lambda to: " << global_val << std::endl; 
    global_func(); 
    std::cout << "After calling lambda again, global_val out of lambda is: " << global_val << std::endl;

    return 0; 
}
```

这段代码将输出：

```cpp
Before calling lambda, global_val out of lambda is: 1
global_val inside lambda is: 1
After calling lambda, global_val out of lambda is: 2
Now change global_val out of lambda to: 5
global_val inside lambda is: 5
After calling lambda again, global_val out of lambda is: 6
```

可以看到，**不用显式捕获全局变量，lambda表达式内部可以直接使用；lambda内部和外部，共享同一个全局变量。一方的修改，将反应到另一方上面。**

综上所述，lambda表达式有这些特点：

1. **按值捕获的变量，在lambda定义时，它在lambda内部的值已经被确定下来。之后，外部对该变量的修改，不会再影响到lambda内部的那一份。**
  
2. **在lambda内部，修改捕获的变量之后，该修改会一直生效，直到lambda实例的生命周期结束。**
  
3. **按引用捕获的变量，lambda内部和外部，共享同一份。一方的修改，将反应到另一方。**
  
4. **不用显式捕获全局变量，lambda表达式内部可以直接使用；lambda内部和外部，共享同一份全局变量。一方的修改，将反应到另一方。**
  

掌握了这些知识，就足以满足常见的lambda表达式应用了。
