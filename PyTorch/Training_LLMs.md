## 《从零开始训练大语言模型的最佳实践》 摘要

这是对于英文版"Current Best Practices for Training LLMs from Scratch"的阅读摘要。

虽然说，只有极少数的公司才会有财力、物力、人力去从零训练一个大语言模型，但是系统的了解一下，总归是有好处的。



#### 自己训练 vs 购买商业服务 vs 基于开源

|     | 购买商业服务                                                      | 基于开源                                                           | 自己训练                         |
| --- | ----------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------- |
| 优点  | 对训练的技能要求最低；训练成本最低，成本主要在于推理阶段；数据成本最低；充分利用市面上最好的LLM；缩短App上市时间 | 充分利用众多开源的LLM；推理时无知识产权/API费用；相比方案1，更多的自主性；相比方案3，数据、训练时间、训练成本都较低 | 最大的自主控制和灵活性；对训练数据的控制；最深的护城河  |
| 缺点  | 使用成本高；敏感数据与隐私安全问题；非护城河；依赖于提供商                               | 需要一定的训练、微调、部署技能；略慢的上市时间；整体性能略差于商业LLM                           | 高风险高成本；需要很多专业技术人员；需要海量的训练数据集 |
| 适合  | 少量技术人员、数据集有限、原型阶段                                           | 不需要修改模型架构、无海量多样性的数据集、有数据合规要求、边缘部署、有限而非通用的使用场景                  | 需要修改模型架构、LLM是战略核心、有源源不断的海量数据 |



#### Scaling Laws

2020年，OpenAI提出LLM scaling laws：增加模型大小，比增加数据大小更重要。

2022年，DeepMind发表论文Training Compute-Optimal Large Language Models (https://arxiv.org/abs/2203.15556)  指出当前LLM训练不足，未经过足够数据的训练。

论文认为，模型大小和训练token的数量应该以大致相同的速率增加。比如，算力增加100倍时，则模型大小应该增加10倍，同时训练token也应该增加10倍。

**所以，选择LLM模型大小的原则是：**

- 基于训练数据大小，参考Chinchilla的最优曲线，确定最佳的模型大小

- 基于训练的预算、推理的时延要求，确定合适的数据和模型大小组合



#### 硬件

为什么需要并行：

- 一张GPU的显存无法容纳模型参数、梯度、优化器状态

- 一张GPU的算力需要漫长的训练时间

并行方法：

- 数据并行：每个GPU上有一份模型的Copy。前向计算时，各自处理自己独一份的训练数据。反向传播时，进行梯度规约，然后用聚合后的梯度去更新每个模型Copy的参数。优点是，算力使用率较高；实现简单。缺点是，通信量较大、需在各个GPU上传递梯度数据；内存使用率较低、每个GPU上都有模型和优化器的Copy。

- 张量并行：即层内并行，张量/矩阵大到无法放到一张GPU上，将矩阵分拆到不同GPU上进行计算。优点是，内存使用率较高。缺点是，通信量较大。

- 流水线并行：即层间并行，将不同的层放到不同的GPU上，并行度受限于模型的深度/层数。优点是，内存和算力使用率较高。*文章中没提到流水线并行的气泡问题。*

*文章中没提到3D/混合并行、MoE。*

*对于DeepSpeed、Horovod、Megatron这些框架也是一笔带过。*



#### 数据集

高质量、高容量、多样化的数据集有助于提高模型性能、加快模型收敛。大多数大模型的数据集并不公开，这是它们的核心竞争力之一。 

文章介绍了免费、公开的EleutherAI Pile数据集 (https://pile.eleuther.ai/) 。

该数据集有5大类：学术论文、在线资源、书籍/散文、对话、其它(GitHub、数学、邮件等)。大小为825GB、有22种数据源。



#### 数据集预处理

数据处理步骤：

采样 (得到相对平衡的数据分布) -> 清洗 (去标签、拼写错误等) -> 非文字转为文字 (表情转文字等) -> 去重 (参考论文 https://arxiv.org/pdf/2107.06499) -> 任务数据清理 (评估数据集和训练数据集的数据不重复)。

分词方法：基于单词、基于字符、基于子词Subword。

文中推荐使用基于子词的方法：常用词不拆分，罕见词拆分成有意义的子词。

并对比了四种基于子词的方法：Byte-Pair Encoding (BPE)、WordPiece、Unigram、SentencePiece。

GPT-3采用BPE分词的方法。

*中文的分词方法不在该文章的讨论范围。*



#### 预训练

训练大模型费时费力费钱。

一般先从小一点的模型出发，整个流程跑通后，再逐渐增加模型的大小。

**通用的步骤：**

- 设计模型架构：一般参考GPT-2、GPT-3的架构，然后做一些调整、增加模型深度和宽度。比如Meta的OPT-175B，基于GPT-3，然后做了这些修改：增加Batch Size、Learning Rating调度、Token数量。

- 实验和超参数调整：权重初始化、位置编码、优化器、激活、学习率、权重衰减、损失函数、序列长度、层数、多头注意力个数、参数量、稀疏层还是稠密层、批量大小、Dropout等参数都会影响模型性能。

**经验建议**

- 学习率：开始时，线性增长，然后逐渐衰减

- 批量大小：从较小开始，然后逐渐增加


**训练的问题：**

**硬件不稳定性** (*注：这是个需要很多投入的地方，随着硬件数量的上升，硬件失效率会越来越大。如何提升可靠性、稳定性、快速恢复都是很有挑战性的课题*)。

**训练/软件不稳定性**：超参数会直接影响训练的稳定性。模型越大，损失值的毛刺现象就越难避免。目前这方面还没有系统性的分析与结论，仅有如下的经验供参考：

- 批量大小 Batch Size：用GPU允许的最大值(*注：即GPU显存的限制*)

- 批量归一化 Batch Normalization：归一化mini-batch中的激活，可加速收敛

- 学习率调度 Learning Rate Scheduling：使用台阶式衰减或者指数式衰减。我们无法事先知道最合适的LR数值，只能通过尝试不同的LR值，并观察模型的结果，最终确定最佳LR数值。

- 权重初始化 Weight Initialization：通常使用较小的高斯噪声，或者T-Fixup

- 训练起点：基于相关任务的预训练模型作为起点

- 正则化 Regularization：dropout、权重衰减可以减少过拟合、更快的收敛、提高泛化能力

- 数据增强：减少过拟合、提高泛化能力

- 热交换 Hot-swapping：热插拔优化器或激活函数(*注：我没看懂*)

- 其它：从checkpoint恢复训练、跳过产生毛刺的数据集等



#### 模型评估

大模型训练完成之后，需要评估其逻辑推理、翻译、自然语言推理、问题回答等能力。常用的评估数据集包括：

- 开放式问答：TriviaQA、Natural Questions、Web Questions

- 完形填空和文本完成：LAMBADA、HellaSwag、StoryCloze

- Winograd-style：Winograd、WinoGrande

- 常识推理: PIQA、ARC、OpenBookQA

- 语境阅读理解：DROP、CoQA、QuAC、SQuADv2、RACE、SuperGLUE

- 自然语言推理：SNLI、QNLI

- 推理：算术推理任务

- 编程：HumanEval、MBPP (text-to-code)、TransCoder (code-to-code)

- 翻译：翻译WMT语言对的评分

- BIG-bench：包括200多个任务，涵盖各种文本任务和编程任务

- LM评估工具：由EleutherAI提供的库，用于标准化评估，包括200多个任务



另外的评估步骤是n-shot学习：

- Zero-shot：在推理时，不向模型提供任何监督样本

- One-shot：在推理时，向模型提供一个监督样本

- Few-shot：在推理时，向模型提供少量监督样本



#### 偏见和毒性

人类带有偏见。人类产生、提供给大模型的数据也会带有偏见。除了需要克服偏见之外，还需确保大模型不记忆和透露隐私数据。

针对此类问题的评估工具：

- 仇恨言论检测：ETHOS数据集可检测英文的陈述

- 社会偏见检测：可以使用CrowSPairs工具

- 有毒语言响应：可以使用RealToxicityPrompts数据集

- 对话安全评估：可以使用SaferDialogues基准测试模型



#### 指令调优

现在我们已经有了预训练的基础大模型，通过指令调优这样的微调技术，可以使得预训练的大模型能够更好地响应指令，减少了对few-shot的需求，大幅提高了zero-shot性能。

通常，预训练的大模型在语言任务组A上进行调优，然后评估其执行语言任务组B的能力，证明其泛化性和zero-shot能力。

指令调优会调整全部模型参数，而不是冻结其中的一部分。

指令调优对于自然表述为指令的任务（比如自然语言推理、问答、翻译）普遍有效，但对于像推理这样的任务来说有难度。



#### 通过人类反馈进行强化学习(RLHF)

RLHF是指令优化的扩展，进一步融入了人类反馈。通过人工评估每个输出的质量，并做为额外的数据 (奖励模型)，以改善模型的整体性能。

ChatGPT采用了RLHF来提升性能。



---

题外话：大模型训练是个昂贵、庞大的系统工程。从算法到工程实现，都有很多很多的地方可以去琢磨、改进。文章里面列的步骤、名词，基本上每一个都可以展开详述、并发表n篇论文或者专利。再加上训练结束后的各种微调、优化、推理、应用，整个长长的产业链上，可以有很多的地方去发力、投入。Exciting吧？

另外，虽然我一直对这种大力出奇迹、能效比极低的做法保留一定的悲观看法，但活在当下，只能抬头看天、低头走路了。



