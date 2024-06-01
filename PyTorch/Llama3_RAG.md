## Llama 3 + LangChain + HugginFace 实现本地部署RAG(检索增强生成)

本文介绍如何基于Llama 3大模型、以及使用本地的PDF文件作为知识库，实现RAG(检索增强生成)。

RAG，是三个单词的缩写：Retrieval、Augmented、Generation，代表了这个方案的三个步骤：检索、增强、生成。

基本的步骤是这样的：

1. 先用本地的各种文件，构建一个向量数据库，做为本地的知识库。

2. 然后当用户对大模型提问时，先在本地的向量数据库里面查找跟问题相关的内容。这一步叫做Retrieval检索。

3. 再把从向量数据库中查找到的内容，和用户的原始问题合到一起，作为Prompt发给大模型。这一步叫做Augmented增强。

4. 最后，大模型会根据prompt返回内容。这一步叫做Generation生成。



道理很简单，但实际用起来，里面会有很多地方需要去仔细的研究、调参。



#### 1. 准备工作

在开始写代码之前，需要先从HuggingFace下载模型文件。我选用的是Meta-Llama-3-8B-Instruct。国内用户可以从hf-mirror.com下载，网络比HuggingFace.com稳定得多。

另外，还需要下载Embeddings模型，用于将文本转为embeddings，然后才能保存到向量数据库，并进行后续的相似性查找。我选用的是微软的multilingual-e5-large-instruct模型。也可以用北大的bge-m3模型。但这两个Embeddings模型的参数和相关度数值会有比较大的差异，需要去试验、调整代码里面的参数。



模型下载之后，需要在本地安装所需的Python库：

```shell
$ pip install PyPDF2 transformers langchain langchain_community langchain_huggingface faiss-cpu
```

目前，我安装的langchain是0.2.1版本。随着版本的不同，这个库大概率会发生较大的改变，从而导致运行失败。

吐槽一下，langchain这玩意的版本兼容性真是无语的很，然后还拆成一堆的库需要分别安装，莫非他们的KPI是按照PIP所需要安装的数量考核的…… 



#### 2. 加载/创建向量数据库

现在，可以开始写代码了。

首先，我们需要看看本地的向量数据库是否已经存在。如果存在的话，直接加载、使用；否则的话，则去读取本地的PDF文件、切分文本、然后用切分好的文本和指定的embeddings模型来创建向量数据库：

```python
# Load pdf file and return the text
def load_single_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    if not pdf_reader:
        return None
    
    ret = ''
    for i, page in enumerate(pdf_reader.pages):
        txt = page.extract_text()
        if txt:
            ret += txt

    return ret

# Split the text into docs
def split_text(txt, chunk_size=256, overlap=32):
    if not txt:
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = splitter.split_text(txt)
    return docs

# Save docs to vector store with embeddings
def create_vector_store(docs, embeddings, store_path):
    vector_store = FAISS.from_texts(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store

# Load vector store from file
def load_vector_store(store_path, embeddings):
    if os.path.exists(store_path):
        vector_store = FAISS.load_local(store_path, embeddings, 
            allow_dangerous_deserialization=True)
        return vector_store
    else:
        return None
    
def load_or_create_vector_store(store_path, pdf_file_path):
    embeddings = create_embeddings()
    vector_store = load_vector_store(store_path, embeddings)
    if not vector_store:
        # Not found, build the vector store
        txt = load_single_pdf(pdf_file_path)
        docs = split_text(txt)
        vector_store = create_vector_store(docs, embeddings, store_path)

    return vector_store
```



#### 3. 检索

得到向量数据库之后，就可以根据用户的问题，在数据库内进行相关性查找(**检索**)。

```python
# Query the context from vector store
def query_vector_store(vector_store, query, k=4, relevance_threshold=0.8):
    similar_docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
    related_docs = list(filter(lambda x: x[1] > relevance_threshold, similar_docs))
    context = [doc[0].page_content for doc in related_docs]
    return context
```

这里，设置了一个relevance_threshold，当查找到的内容的相关度小于这个数值时，则认为无关，即无法从向量数据库里查找到与问题相关的信息。



#### 4. 增强与生成

根据从向量数据库查找到的信息/上下文，可以把这些信息跟用户的输入的问题合到一起(**增强**)，然后一起发给已经加载的大模型(**生成**)。

```python
def ask(model, tokenizer, prompt, max_tokens=512):
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|eot_id|>')
    ]
    input_ids = tokenizer([prompt],
        return_tensors='pt', 
        add_special_tokens=False).input_ids.to(CUDA_Device)
    generated_input = {
        'input_ids': input_ids,
        'max_new_tokens': max_tokens,
        'do_sample': True,
        'top_p': 0.95,
        'temperature': 0.9,
        'repetition_penalty': 1.1,
        'eos_token_id': terminators,
        'bos_token_id': tokenizer.bos_token_id,
        'pad_token_id': tokenizer.pad_token_id
    }

    generated_ids = model.generate(**generated_input)
    ans = tokenizer.decode(generated_ids[0], skip_special_token=True)
    return ans

def main():
    pdf_file_path = './Data/Aquila.pdf'
    store_path = './Data/Aquila.faiss'

    vector_store = load_or_create_vector_store(store_path, pdf_file_path)
    model, tokenizer = load_llm(LLM_Model)

    while True:
        qiz = input('Please input question: ')
        if qiz == 'bye' or qiz == 'exit':
            print('Bye~')
            break

        # Query context from vector store based on question, and compose prompt
        context = query_vector_store(vector_store, qiz, 6, 0.75)
        if len(context) == 0:
            # No satisfying context is found inside vector store
            print('Cannot find qualified context from the saved vector store. Talking to LLM without context.')
            prompt = f'Please answer the question: \n{qiz}\n'
        else: 
            context = '\n'.join(context)
            prompt = f'Based on the following context: \n{context}\nPlease answer the question: \n{qiz}\n'

        ans = ask(model, tokenizer, prompt)[len(prompt):]
        print(ans)
```



代码里面，事先加载了向量数据库、加载了大模型；然后在while循环里面，不停的让用户输入问题。根据输入的问题，去向量数据库查找相关的上下文。如果查找到了，则合到一起，发给大模型；否则将原始问题发给大模型。

完整的代码请查看：[DeepLearning/Llama3_RAG.py at main · yuchuanwang/DeepLearning · GitHub](https://github.com/yuchuanwang/DeepLearning/blob/main/Llama3_RAG.py)



#### 5. 问题

正如一开始所说的，RAG的道理很简单，但实际用起来，会发现里面有很多的地方需要去调参、研究。

比如说：

切分文本的时候，chunk_size和chunk_overlap取多少合适？

文本转向量时，使用哪个Embeddings模型最佳？

查找问题的相关上下文时，用欧式距离还是别的距离，比如余弦距离还是？

每个Embeddings模型，以多大的的相关度数值做为阈值合适？

如何评估RAG的整体效果？

等等等等…… 

真的是路漫漫其修远兮！


