## 用OpenAI的API实现基本的ChatGPT

OpenAI提供了各种模型和API，可以用于开发自己的AI应用。

具体的API请参考：[OpenAI Platform](https://platform.openai.com/docs/api-reference)

虽然这些API也在发生变化，但比LangChain稳定不少。



本文尝试用OpenAI的API来实现一个基本的对话功能，类似于ChatGPT那样的。

基本的思路是构建一个聊天的类，类里面保存了模型的参数、API Key、对话历史记录等信息。通过保存历史记录、并发给OpenAI的服务器来提供上下文信息。

同时，还提供了最长历史记录的参数，当聊天记录过长时，会把最开始的聊天记录删除。

另外，由于每个token都是需要计费的。为了节省费用，可以先把之前的聊天记录发给OpenAI，让它进行归纳总结。再用归纳得到的文字，做为上下文，这样可以大大的节省每次所需的token（和钱）。



---

广告分割线

所有的这些操作都是需要一个OpenAI的API key的。而OpenAI目前不对大陆的用户开放，所以使用国内的信用卡是无法充值、获取密钥的。我是通过国外的虚拟信用卡充值完成的。用的是这个平台：[WildCard | 一分钟注册，轻松订阅海外软件服务](https://bewildcard.com/i/YCWANG)，支持支付宝充值。卡充值完成后，就可以到OpenAI的官网：[OpenAI Platform](https://platform.openai.com/account/billing/overview)去付费、获取密钥了。

说明一下哈，链接里面带了我的邀请码。每推广成功一个人的话，我大约可以赚一个鸡腿。

---



直接上代码。



```python
from typing import List
from openai import OpenAI

class MiniChatGPT:
    def __init__(self, 
            api_key: str, 
            max_round: int = 10, 
            system_prompt: str = '', 
            temperature: float = 1.0):
        self.max_round = max_round
        self.api_key = api_key
        self.chat_model = 'gpt-3.5-turbo'
        self.completions_model = 'gpt-3.5-turbo-instruct'

        self.temperature = temperature
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0

        self.messages = []
        self.messages.append({"role": "system", "content": system_prompt})

    def Reset(self):
        self.messages = []

    def Ask(self, question):
        # Build messages, and send to OpenAI server
        # Then append the response to messages
        # Need to forget the previous message if reaching max_round
        self.messages.append({"role": "user", "content": question})

        # Send to OpenAI server
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model = self.chat_model,
            messages = self.messages,
            )
        
        # Get result
        ans = response.choices[0].message.content

        # Save response to message history
        self.messages.append({"role": "assistant", "content": ans})
        if len(self.messages) > self.max_round * 2 + 1:
            # Delete the first round
            del self.messages[1:3]

        return ans

    def Summary(self, max_tokens=100):
        # Return the summary of the previous messages
        # Format messages to history
        history = ''
        for i in range(1, len(self.messages)):
            msg = self.messages[i]['role'] + ': ' + self.messages[i]['content'] + '\n'
            history += msg

        # Send to OpenAI server
        client = OpenAI(api_key=self.api_key)
        response = client.completions.create(
            model = self.completions_model,
            # TL;DR: "too long; didn't read", OpenAI will summary the message before it
            prompt = history + '\nTL;DR\n',
            temperature = self.temperature,
            top_p = self.top_p,
            frequency_penalty = self.frequency_penalty,
            presence_penalty = self.presence_penalty,
            max_tokens = max_tokens,
            )
        
        # Get result
        ans = response.choices[0].text
        return ans


def test_mini_gpt():
    key = 'sk-Your-Own-Api-Key'
    role_context = '你是专业的厨师，请回答如何做菜。回答限制在50个字以内'
    gpt = MiniChatGPT(api_key=key, max_round=3, system_prompt=role_context)
    qiz = '你是谁？'
    ans = gpt.Ask(qiz)
    print('\n')
    print(f"User: {qiz}")
    print(f'ChatGPT: {ans}')

    qiz = "请问鱼香肉丝怎么做？"
    ans = gpt.Ask(qiz)
    print('\n')
    print(f"User: {qiz}")
    print(f'ChatGPT: {ans}')

    qiz = "那蚝油牛肉呢？"
    ans = gpt.Ask(qiz)
    print('\n')
    print(f"User: {qiz}")
    print(f'ChatGPT: {ans}')

    qiz = "蟹粉小笼呢？"
    ans = gpt.Ask(qiz)
    print('\n')
    print(f"User: {qiz}")
    print(f'ChatGPT: {ans}')

    summary = gpt.Summary()
    print(f'ChatGPT Summary: {summary}')

    gpt.Reset()
    role_context = summary + "\n请你根据已经聊了的内容，继续对话："
    gpt = MiniChatGPT(api_key=key, max_round=5, system_prompt=role_context)

    qiz = "宫保鸡丁呢？"
    ans = gpt.Ask(qiz)
    print('\n')
    print(f"User: {qiz}")
    print(f'ChatGPT: {ans}')



if __name__ == '__main__':
    test_mini_gpt()
```



修改API Key之后，运行该程序，可以得到类似下面的效果：

```shell
User: 你是谁？
ChatGPT: 我是专业的厨师，有丰富的烹饪经验。如果你有任何做菜方面的问题或需要建议，我会很乐意为你提供帮助。

User: 请问鱼香肉丝怎么做？
ChatGPT: 鱼香肉丝的做法：
1. 将猪肉切丝，加入腌料腌制。
2. 将蒜、姜、青椒切碎备用。
3. 锅中加油，烧热后加入蒜姜煸炒。
5. 加入调味料翻炒，收汁后即可出锅。

ChatGPT: 蚝油牛肉的做法：
1. 牛肉切薄片，加入腌料腌制。
2. 锅中加油，将牛肉片煎炒至变色后盛出备用。
3. 锅中再加油，加入葱姜爆香，倒入蚝油、酱油、糖等调味料炒匀。
4. 加入牛肉翻炒均匀即可出锅。

User: 蟹粉小笼呢？
ChatGPT: 抱歉，蟹粉小笼是一道复杂的菜肴，无法用50字内解释清楚。建议找到专业菜谱进行学习制作，或者品尝专业烹饪师傅做的蟹粉小笼包。祝你好
运！

ChatGPT Summary: 鱼香肉丝的做法是先切丝、腌制、炒香肉丝，再加入调味料和蔬菜炒匀。蚝油牛肉是先腌制牛肉，再爆炒，最后加入调味料和牛肉烧至
入味。

User: 宫保鸡丁呢？
ChatGPT: 宫保鸡丁的做法是先将鸡肉切丁后腌制，然后爆炒至七成熟。接着加入花生米、青椒丁、葱姜蒜等炒熟，最后加入特制的酱汁烧炒均匀，即可起
锅。这道菜又香又辣，口感鲜嫩，非常开胃。你喜欢吃宫保鸡丁吗？
```



所以，只要实现一个新的函数：循环读取客户输入的文字qiz = input()，然后通过gpt.Ask(qiz)发给OpenAI，就可以不停的与AI唠嗑了。

例子里面设定的AI角色是一位大厨。实际使用中也可以根据不同的Agent角色设定，为其设置相应的system_prompt，这样就可以得到精通不同技能的AI小助理了。




