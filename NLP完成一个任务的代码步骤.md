**NLP完成一个任务的代码步骤**

***

1. 准备预训练模型

   以bert-base-uncased为例，我们从huggingface官网https://huggingface.co上下载好所需要的文件

   ![](https://github.com/Chen-Shaobin/Learning-Notes/blob/main/figure/bert_download1.png)

<img src="D:\blog\blog image\bert_download2.png" alt="image" style="zoom:25%;" />

下载好这三个文件后放到目标文件夹即可

2.准备训练数据

将其分为训练集、开发集和测试集

3.写代码

- 导入包

- 解析命令行参数

- 判断是否存在输出文件夹、是否远程debug、单gpu训练还是多gpu训练

- 定义任务对应的preprocessor处理我们输入的文件

  def processor(DataProcessor):

  ​	定义五个函数：读取数据集、开发集、读取测试集、获取返回label、制作后面模型需要用到的那种格式的数据

- 加载预训练模型（以bert为例）

  import torch as t from transformers

  import BertTokenizer,BertModel

  str = "I used to be a bank, but I lose interest."  

  tokenzier = BertTokenizer.from_pretrained("model/bert-base-uncased") 

  bert = BertModel.from_pretrained("model/bert-base-uncased")

- 加载数据集

  load_and_cache_examples用于加载数据集

  核心代码：examples = processor.get_train_examples(args.data_dir)

  目的：利用processor读取训练集

  核心代码：features = convert_examples_to_features(examples,tokenizer,label_list = label_list,max_length=args.max_seq_length,output_mode=output_mode,pad_on_left=bool(args.model_type in ['xlnet']),pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

  目的：将文本形式数据转为向量

  核心代码：dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens,all_labels)

  目的：将转化之后的数据再转为tensor,然后使用TensorDataset构造最终的数据集并返回

- 训练模型

  - 对数据随机采样：randomsampler
  - dataloader读取数据
  - 参数设置：warm_up参数，优化器等等
  - 将batch送进模型进行训练
  - 输出参数、打印日志等等

- 测试模型

- 保存训练好的模型

​	
