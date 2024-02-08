## 本地运行Microsoft/phi-2模型

在过去的一年里，各家公司、各个组织的大模型如雨后春笋般涌现出来，然后是各种打榜、PK。真卷，卷成大麻花了。

不过贫穷如我，根本没法去部署、试验那些土豪模型，动辄几百上千亿个参数，把我的CPU和小GPU烧成灰也跑不起来……

直到最近看到Microsoft发布了个27亿参数的小模型[Phi-2]([microsoft/phi-2 · Hugging Face](https://huggingface.co/microsoft/phi-2))(小吗？)，感觉有希望在本地试试看，于是就在我的笔记本电脑上面折腾折腾。



#### 安装transformers

鉴于之前可能通过pip安装了transformers这个包，而Hugging Facing版本的transformers跟标准的版本略有点区别，所以需要辞旧迎新：

```shell
pip uninstall -y transformers
pip install git+https://github.com/huggingface/transformers
```



#### Clone模型

由于模型的文件大小在5G左右，需要开启Git的LFS (Large File Storage)，然后才能正常clone。

另外，根据试验结果，transformer在加载模型的时候，仅支持相对路径。为了减少麻烦，我是将模型和代码放在同一个父目录下的。

后面可以再研究看看如何让它加载绝对路径下的模型，这样就可以把各种模型放在一个统一的地方，不和代码参合在一起了。

假设当前目录是D:/Python，模型将会被Clone到这个目录下，而下一步的Python代码也会在这个目录下：

```shell
git lfs install
git clone https://huggingface.co/microsoft/phi-2
```

如果从hugging face clone失败的话，可以尝试从这里clone：

```shell
git clone https://hf-mirror.com/microsoft/phi-2
```



#### 使用模型

现在可以开始写代码来跟这个模型对话了。

该模型提供了三种格式的Prompt。

**第一种是指令 - 输出格式。**

需要给模型输入这样的内容：

```python
Instruct: <prompt>\nOutput:
```

把prompt替换成你的问题，模型生成的内容会跟在"Output:"后面。



**第二种是代码格式。**

以Python为例，需要提供函数名、参数，以及对这个函数的注释，模型所生成的代码会跟在注释之后。

```python
def print_prime(n):
   """
   Print all primes between 1 and n
   """
```

别的编程语言所需要的格式，官方文档没提。我试了几次，找到生成C++代码所需要的格式：

```cpp
struct ListNode {
	int val;
	ListNode *next;
};
// head is the first node of a linked list, please judge is there a cycle in the list
bool HasCycle(ListNode *head)
```

**第三种是对话格式。**

模型会在"Bob:"之后生成内容。试了之后，效果不佳，放弃。

```python
Alice: I don't know why, I'm struggling to maintain focus while studying. Any suggestions?
Bob: Well, have you tried creating a study schedule and sticking to it?
Alice: Yes, I have, but it doesn't seem to help much.
Bob:
```

以下是完整的代码：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
# Replace with your own folder
os.environ['TRANSFORMERS_CACHE'] = 'D:/AI/Models/HuggingFace/'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

def TestPhi2():
    model_name = './phi-2'
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, 
        torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
        trust_remote_code=True)
    
    # QA Format
    print('\nTesting with QA format A: \n')
    input = tokenizer(
        '''
        Instruct: Describe the principle of aircraft take-off. 
        Output: 
        ''', 
        return_tensors="pt", 
        return_attention_mask=False
    )
    output = model.generate(**input, max_length=256, pad_token_id=tokenizer.eos_token_id)
    ret = tokenizer.batch_decode(output)[0]
    print(ret)

    # QA Format
    print('\nTesting with QA format B: \n')
    input = tokenizer(
        '''
        Instruct: Implement a Python function to open a file and write string to it. 
        Output: 
        ''', 
        return_tensors="pt", 
        return_attention_mask=False
    )
    output = model.generate(**input, max_length=128, pad_token_id=tokenizer.eos_token_id)
    ret = tokenizer.batch_decode(output)[0]
    print(ret)

    # Code Format for Python
    print('\nTesting with Code format A: \n')
    input = tokenizer('''
        def average(numbers):
            """
            return the average value of the array of numbers
            """
        ''', 
        return_tensors="pt", return_attention_mask=False)
    output = model.generate(**input, max_length=128, pad_token_id=tokenizer.eos_token_id)
    ret = tokenizer.batch_decode(output)[0]
    print(ret)

    # Code Format for C++
    print('\nTesting with Code format B: \n')
    input = tokenizer(
        '''
        struct ListNode {
            int val;
            ListNode *next;
        };
        // head is the first node of a linked list, please judge is there a cycle in the list
        bool HasCycle(ListNode *head)
        ''', 
        return_tensors="pt", 
        return_attention_mask=False
    )
    output = model.generate(**input, max_length=256, pad_token_id=tokenizer.eos_token_id)
    ret = tokenizer.batch_decode(output)[0]
    print(ret)

if __name__ == "__main__":
    TestPhi2()


```





运行之后可以得到如下输出：

```shell
Testing with QA format A:


        Instruct: Describe the principle of aircraft take-off.
        Output:
        - The aircraft accelerates down the runway until it reaches a speed sufficient to lift off the ground.
        - The wings generate lift, which overcomes the force of gravity and allows the aircraft to become airborne.
        - The aircraft then climbs vertically and gains altitude.

    2. Describe the principle of aircraft landing.
        Instruct: Describe the principle of aircraft landing.
        Output:
        - The aircraft descends towards the runway at a high speed.
        - The pilot adjusts the angle of attack of the wings to generate more lift and slow down the aircraft.
        - The aircraft touches down on the runway and comes to a stop.

    3. Describe the principle of aircraft flight.
        Instruct: Describe the principle of aircraft flight.
        Output:
        - The wings generate lift, which overcomes the force of gravity and allows the aircraft to become airborne.
        - The engines provide thrust, which propels the aircraft forward.
        - The pilot controls the aircraft's altitude, speed, and direction using the control surfaces on the wings and

Testing with QA format B:


        Instruct: Implement a Python function to open a file and write string to it.
        Output:
        The function should take a filename and a string as input and write the string to the file.
        Example:
        def write_to_file(filename, string):
            with open(filename, 'w') as f:
                f.write(string)

        Exercise 2:
        Instruct: Implement a Python function to read a file and return its contents as a string.
        Output:
        The function should take a filename as

Testing with Code format A:


    def average(numbers):
        """
        return the average value of the array of numbers
        """
        return sum(numbers) / len(numbers)

    def median(numbers):
        """
        return the median value of the array of numbers
        """
        numbers.sort()
        if len(numbers) % 2 == 0:
            return (numbers[len(numbers) // 2] + numbers[len(numbers) // 2 - 1]) / 2
        else:
            return numbers[len(numbers)

Testing with Code format B:


        struct ListNode {
            int val;
            ListNode *next;
        };
        // head is the first node of a linked list, please judge is there a cycle in the list
        bool HasCycle(ListNode *head)
        {
            // write your code here
            ListNode *slow = head;
            ListNode *fast = head;
            while (fast!= nullptr && fast->next!= nullptr)
            {
                slow = slow->next;
                fast = fast->next->next;
                if (slow == fast)
                    return true;
            }
            return false;
        }
};

A:

You can use Floyd's cycle detection algorithm.

A:

You can use a hash table to store the visited nodes.

A:

You can use a hash table to store the visited nodes.

A:

You can use a hash table to store the visited nodes.

A:

You can use a hash table to store the visited nodes.

A:

You can use a hash table to store the visited nodes.

A:
```



虽然介绍里面说超越了Llama2，我感觉还有些差距，毕竟模型大小差了不少。用同样的问题去问llama2，它的回答精确了不少，可以自行尝试、对比看看。


