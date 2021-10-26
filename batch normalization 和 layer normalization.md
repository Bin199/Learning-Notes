batch normalization 和 layer normalization:

共同点：
作用：两者均可加速模型训练
为什么能加速模型训练：一开始人们认为是神经网络的各个层参数分布不一致，导致训练的时候，信息从这一层到
另外一层需要 “适应”，而batch normalization 将每一层的参数分布变得相似，不过后来有人推翻了这个原因，
认为原因是batch normalization使得损失函数曲面变得更加平滑，导致训练时间缩短；在知乎推文里（https://www.zhihu.com/question/38102762）
作者说BN 有效的原因是防止“梯度弥散”，但不是很理解他的说法

<img src="D:\blog\blog image\why BN.png" style="zoom:33%;" />

再后来，batch normalization
在batch size很小的时候性能不好（原因见下文），于是提出了layer normalization弥补了不足

不同点：
理论上的不同：batch normalization是针对一个batch当中的所有数据求在某一个特征上的均值、方差，然后normalize
求均值和方差要求数据量比较大，而当batch size很下的时候，也即batch中的数据量小，便没有很好的效果
layer normalization就不一样了，它是对某一个数据的不同特征，求这些特征的均值和方差，然后normalize

应用场景不同：数据量较小的时候，用layer normalization;否则，用batch normalization较好，因为求不同数据的某一特征的均值与方差
作normalize效果更好



附录：为了防止BN破坏输入数据的性质，我们会加上两个参数，使得经过学习有可能恢复输入数据，至于为什么这么做，贴一个精彩的评论：

![](D:\blog\blog image\BN and LN.png)