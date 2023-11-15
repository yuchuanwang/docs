## C语言与C++语言混合编程的方法

在实际的开发工作中，有时会受到现有代码库的约束，必须对C和C++语言进行混合编程。

而要实现混合编程的主要办法就是声明：extern "C"。被它修饰的变量和函数，将会按照C语言方式进行编译和连接。



#### C++调用C

从C++调用C语言的函数会比较简单，只需要对函数进行如下的修改：

```cpp
// C function to be called by C and C++
#ifdef __cplusplus
extern "C"{
#endif

void foo_with_C();

#ifdef __cplusplus
}
#endif
```

通过#ifdef __cplusplus这个宏定义，使得extern "C"声明在C++时启用、C时不启用。这样此函数既可用于C语言，也可用于C++语言。

我们在调用标准库的函数时，其实标准库已经把这一步做好了，所以感觉C++可以透明的调用C的函数。



#### C调用C++

而从C语言调用C++会比较麻烦一些，需要针对C语言声明各种API函数，然后在API函数里再用C++的语法去调用原来的C++代码。基础的原理也是依赖于extern "C"这个声明。

以下用一个简单的例子来讲解这个过程。

假设现有的代码库里面，已经有了一个Employee的C++类。以下分别为该类的头文件和cpp文件。

```cpp
// Employee.h
// Pure C++ code
// Only for C++

#ifndef EMPLOYEE_H
#define EMPLOYEE_H

#include <string>

class Employee
{
public:
    Employee(int id, const std::string& first_name, const std::string& last_name, float salary);
    ~Employee();

    int GetID() const;
    void SetID(int val);

    std::string GetFirstName() const;
    void SetFirstName(const std::string& val);

    std::string GetLastName() const;
    void SetLastName(const std::string& val);

    float GetSalary() const;
    void SetSalary(float val);

    void AdjustSalary(float delta);

    std::string Summary() const;

private:
    int id;
    std::string firstName;
    std::string lastName;
    float salary;
};

#endif

```

```cpp
// Employee.cpp
// Pure C++ code
// Only for C++

#include <iostream>
#include <sstream>
#include "Employee.h"

using namespace std;

Employee::Employee(int id, const string& first_name, const string& last_name, float salary)
{
    this->id = id;
    this->firstName = first_name;
    this->lastName = last_name;
    this->salary = salary;
}

Employee::~Employee()
{
}

int Employee::GetID() const
{
    return id;
}

void Employee::SetID(int val)
{
    id = val;
}

string Employee::GetFirstName() const
{
    return firstName;
}

void Employee::SetFirstName(const string& val)
{
    firstName = val;
}

string Employee::GetLastName() const
{
    return lastName;
}

void Employee::SetLastName(const string& val)
{
    lastName = val;
}

float Employee::GetSalary() const
{
    return salary;
}

void Employee::SetSalary(float val)
{
    salary = val;
}

void Employee::AdjustSalary(float delta)
{
    salary += delta;
}

string Employee::Summary() const
{
    stringstream ss;
    ss << "ID: " << id << ", Name: " << firstName << " " << lastName << ", Salary: " << salary;
    return ss.str();
}


```



现在，我们有个C语言的项目，需要复用这个类的功能。

为了在C语言复用Employee类，我们需要创建一个Wrapper，包括头文件和cpp文件。

其中，头文件里面，根据需要声明各个API函数。该文件会被C语言使用，需要用extern "C"来声明API函数，而且不能使用C++独有的语法与功能(比如class)。

cpp文件，包括了各个API函数的实现。针对C++编译器，可以使用C++的语法与功能。

参考代码如下所示：

```cpp
// Employee_API.h
// Need to support both C and C++
// Wrap C++ classes into C interfaces
// C calls C++ interfaces by including this file

#ifndef EMPLOYEE_API_H
#define EMPLOYEE_API_H

#ifdef __cplusplus
// Link with C way
extern "C" {
#endif

void* Employee_Create(int id, const char* first_name, const char* last_name, float salary);
void Employee_Adjust_Salary(void* employee, float delta);
void Employee_Summary(void* employee, char* ret);
void Employee_Destroy(void* employee);

#ifdef __cplusplus
}
#endif

#endif


```

```cpp
// Employee_API.cpp
// Only for C++

#include "Employee_API.h"
#include "Employee.h"

#ifdef __cplusplus
extern "C" {
#endif

void* Employee_Create(int id, const char* first_name, const char* last_name, float salary)
{
    return new Employee(id, first_name, last_name, salary);
}

void Employee_Adjust_Salary(void* employee, float delta)
{
    Employee *emp = (Employee *)employee;
    if (!emp)
    {
        return;
    }

    emp->AdjustSalary(delta);
}

void Employee_Summary(void* employee, char* ret)
{
    Employee *emp = (Employee *)employee;
    if (!emp)
    {
        return;
    }

    std::string info = emp->Summary();
    info.copy(ret, info.length());
}

void Employee_Destroy(void* employee)
{
    Employee *emp = (Employee *)employee;
    if (emp)
    {
        delete emp;
    }
}

#ifdef __cplusplus
}
#endif

```



相当于我们把类的成员函数拆出来，变成一个个普通的函数。原来的类实例，则变成一个函数的指针参数。

接下来，我们就可以在C语言里面通过这个Wrapper来使用C++的功能了。

下面是个简单的C语言例子：

```c
#include <stdio.h>
#include <string.h>
#include "Employee_API.h"

int main(int argc, char const *argv[])
{
    printf("Demo to show how to call C++ class and methods from C.\n");

    void* tom = Employee_Create(100, "Tom", "Jerry", 8888.88);
    if (!tom)
    {
        return 1;
    }

    const int max_len = 256;
    char info[max_len];
    memset(info, '\0', sizeof(info)); 
    Employee_Summary(tom, info);
    printf("%s\r\n", info);

    Employee_Adjust_Salary(tom, 1111.11);

    memset(info, '\0', sizeof(info)); 
    Employee_Summary(tom, info);
    printf("%s\r\n", info);

    Employee_Destroy(tom);

    return 0;
}


```



至此，代码方面的工作已经完成。接下来有两种方法完成混编的工作。

##### 1. 通过动态链接库.so文件

第一种做法，是把原来的C++代码，还有新加的Wrapper代码，编译成一个动态链接库；然后C语言的代码link这个库，得到可执行文件。

具体做法如下：

```shell
$ g++ -Wall -fpic -shared Employee.cpp Employee_API.cpp -o libemployee_api.so
$ gcc -Wall -c main.c -o main.o
$ gcc -Wall main.o -o mixed -L . -lemployee_api
```

这样，得到一个.so动态链接库，和一个mixed程序。

通过指定动态链接库搜索目录，可以启动mixed程序：

```shell
$ LD_LIBRARY_PATH="$(pwd)"
$ export LD_LIBRARY_PATH
$ ./mixed
Demo to show how to call C++ class and methods from C.
ID: 100, Name: Tom Jerry, Salary: 8888.88
ID: 100, Name: Tom Jerry, Salary: 9999.99
```

可以看到，程序正常运行、输出期望的结果了。



##### 2. 通过连接stdc++和.o文件

第二种做法，则是把各个cpp文件都变成.o文件，最后与C语言的.o文件link到一起，得到可执行文件。具体做法如下：

```shell
$ g++ -Wall -c Employee.cpp -o Employee.o
$ g++ -Wall -c Employee_API.cpp -o Employee_API.o
$ gcc -Wall -c main.c -o main.o
$ gcc -Wall Employee.o Employee_API.o main.o -o mixed -lstdc++
```

这样，得到一个mixed程序。

直接运行该程序：

```shell
$ ./mix
Demo to show how to call C++ class and methods from C.
ID: 100, Name: Tom Jerry, Salary: 8888.88
ID: 100, Name: Tom Jerry, Salary: 9999.99
```

可以看到，这种方法也是可以正常运行、输出期望结果的。



实际项目中，具体选用哪种方法，可以根据项目的情况决定。但不管用哪种方法，繁琐的Wrapper封装成API的步骤，是少不了的。


