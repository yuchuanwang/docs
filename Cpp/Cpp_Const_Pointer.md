## C++ const与指针

写C++代码时，有时看到类似这样的代码，会愣一下、想一想它们到底是在说什么：

```cpp
	int val = 10; 
	const int* pA = &val; // Or: int const* pA
	int* const pB = &val; 
	const int* const pC = &val;
```

因此，做个笔记，把const修饰符和指针混用的情况整理一下，加深理解。

如下这个语句：

```cpp
	const int* pA = &val; // Or: int const* pA
```

const限定的是int，称为：**指向常量的指针**, 也称为：**常量指针**。即：

1. 指针本身可变，可以指向别的地址，例如：pA = &valB；
  
2. 指针指向的内容是常量，不能通过指针来修改内容，*pA = 11是非法的。
  

但如上的例子，可以通过访问val本身来修改内容，指针就无能为力了。

所以，指向常量的指针这么使用：

```cpp
	int valA = 10; 
	int valB = 20;
	const int* pA; 
	pA = &valA;
	// pA can be modified to other address
	pA = &valB;
	// Compiling error, cannot modify the value pointed by pA
	//*pA = 11;
	// While we can modify the value directly, bypass pointer
	valA = 11;
```

它的英文名称，感觉比**常量指针**这个翻译更容易理解：**Pointer to const**。


接下来一行代码：

```cpp
	int* const pB = &val; 
```

这种指针称为：**指针常量**，英文名称：**Const Pointer**。即：

1. 指针本身是常量，必须初始化。而且初始化之后，不允许再指向别的地址；
  
2. 指针指向的内容，不受const约束。
  

所以，指针常量这么使用：

```cpp
	int valA = 10; 
	int valB = 20; 
	// const pointer must be initialized
	int* const pB = &valA; 
	// Compiling, cannot modify the pointer itself
	//pB = &valB;
	// While we can modify the value pointed by the pointer
	*pB = 11; 
```

---

这两种指针长得这么像，如何区分呢？我个人的经验，就看const与*的相对位置。

1. 如果const在*的左边，那么就是Pointer to const，限定指向的内容；
  
2. 如果const在*的右边，那么就是Const Pointer，限定指针本身。
  
3. 如果const在*的左边、右边都存在，那么就是Const Pointer to const。不管是指针本身、还是指针指向的内容，都被const限定。
  

---

汇总成一张表格：

|     | 用法  | const位置 | 修改指针本身 | 修改指针指向的内容 |
| --- | --- | --- | --- | --- |
| Pointer to const | const int * pA; <br/>int const * pA; | 在星号*左边 | 允许  | 不允许 |
| Const Pointer | int* const pB = &val; | 在星号*右边 | 不允许 | 允许  |

---
