# Sentiment Analysis Report

- 选题为情感分析。

- 模型为[Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing]([2006.03236.pdf (arxiv.org)](https://arxiv.org/pdf/2006.03236.pdf))论文所提出的Funnel Transformer.对其简介可见repo中的PPT。其最大特点是每经过一个block(几层Transformer)之后，就会pooling序列到一半长度。

- 采用sst2数据集，用`funnel/small`配上论文的设置复现了结果之余，还做了实验检验自己关于模型结构的想法。

  

![Picture1](C:\Users\v-weishengli\Pictures\Picture1.jpg)

- 本实验的代码已经上传到GitHub [li1117heex/pyramidformer (github.com)](https://github.com/li1117heex/pyramidformer)。
- 代码用的是Transformers的框架，预训练模型也来自于Transformers。



## 核心实验

作者的参数和实验结果为下表中对应的‘sst2'列:

![image-20201231203754495](C:\Users\v-weishengli\AppData\Roaming\Typora\typora-user-images\image-20201231203754495.png)![image-20201231203832818](C:\Users\v-weishengli\AppData\Roaming\Typora\typora-user-images\image-20201231203832818.png)



我实验结果如下，基本对应。

| loss  | accuracy |  F1  |
| :---: | :------: | :--: |
| 0.204 |   93.2   | 93.4 |



### 我的猜想

多加几个block会提升模型的表现，因为到后面的block的时候序列变得更短，信息会更加充分的汇集到用于分类的位于序列首位的[cls]token中。而时间消耗则基本不变，因为



在原本参数seq_len=128,模型有3个各4层的block,即`block_sizes=[4,4,4]`的情况下，最终序列长度为32。我分别对5，6，7个各4层block的模型设置进行了实验，结果如下：

| block | layer per block | epoch |  lr  | loss  | accuracy |  F1  |
| :---: | :-------------: | :---: | :--: | :---: | :------: | :--: |
|   5   |        4        |   5   | 1e-6 | 0.553 |   74.1   | 77.0 |
|   6   |        4        |   5   | 1e-6 | 0.696 |   50.9   | 67.5 |
|   7   |        4        |   5   | 1e-6 | 0.693 |   50.9   | 67.5 |

（参数

非常不佳。甚至在6，7block时得到了recall=1的荒谬结果。

我觉得有2个可能原因：

1. 学习率太低
2. 模型层数过深

以下实验结果：

| block | layer per block | epoch |  lr  | loss  | accuracy |  F1  |
| :---: | :-------------: | :---: | :--: | :---: | :------: | :--: |
|   7   |        2        |   5   | 1e-6 | 0.694 |   53.6   | 62.1 |
|   6   |        2        |   5   | 1e-6 | 0.666 |   60.8   | 60.0 |
|   7   |        4        |   5   | 1e-5 | 0.698 |   50.9   | 67.5 |
|   7   |        4        |   5   | 1e-4 | 0.698 |   50.9   | 67.5 |

还是很荒谬。

但是在2层block搭配`lr=1e-5`时，得到了不错的结果：

| block | layer per block | epoch |  lr  | loss  | accuracy |  F1  |
| :---: | :-------------: | :---: | :--: | :---: | :------: | :--: |
|   7   |        2        |   5   | 1e-5 | 0.358 |   87.3   | 87.6 |
|   7   |        2        |   5   | 1e-4 | 0.698 |   50.9   | 67.5 |
|   7   |        2        |   5   | 5e-6 | 0.368 |   87.5   | 88.1 |

再减少block:

| block | layer per block | epoch |  lr  | loss  | accuracy |  F1  |
| :---: | :-------------: | :---: | :--: | :---: | :------: | :--: |
|   6   |        2        |   5   | 1e-5 | 0.333 |   87.7   | 88.3 |
|   4   |        2        |   5   | 1e-5 | 0.356 |   88.1   | 88.6 |

和原来的模型表现只差几个点，时间也节省了不少，从10min降到了6min.



## 结论

- 增加block只有反效果
- 其实原本的模型每个block的层数减少一些也可以接受

## 不足

- 对于多出来的层，参数直接用原本最后一层复制过去。这也会影响新增的block的表现。如果能找到更好的初始化方法，或者训练更加充分会好一些。

## 结尾

这个模型我在微软实习的这段时间里曾经研究过，还曾经从头跑过pretrain,这次算实践了一下自己的idea.计算资源都来自微软服务器。