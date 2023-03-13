# Keras-LSTM
基于Keras搭建一个简单的LSTM，用IMDB影评数据集对LSTM进行训练，完成模型的保存和加载以及测试。

环境：<br />
CUDA：11.6.134<br />
cuDNN：8.4.0<br />
keras：2.9.0<br />
tensorflow：2.9.1<br /><br />

注意：<br />
项目内目录中两个文件夹：<br />
1. /datasets：存放数据集文件<br />
2. /save_models：保存训练好的模型权重文件<br /><br />

加载模型权重时请确保使用的模型和保存的权重一致<br />
比如，当你保存的是自定义的lstm模型权重，那么同样需要构建自定义的lstm模型来读取这个权重<br />
当使用Keras官方实现的lstm时，就会报错，即使官方实现的lstm和自定义的lstm参数量是一样的<br />
反之亦然<br /><br />

如同自定义实现简单RNN时所说，实现自己的自定义LSTM：<br />
Keras实现自定义循环神经网络需要：<br />
1.实现自定义Cell，比如一个自定义的LSTMCell，注意需要定义状态参数维度：self.state_size<br />
2.将实现好的Cell作为参数cell传入Keras.layers.RNN()，让Keras自动推断每个时刻的传递过程<br /><br />

数据集：<br />
IMDB：影评数据集,训练集/测试集包含25000/25000条影评数据<br />
链接：https://pan.baidu.com/s/18nX-2mqJzYU8XKQ5cfhxvw?pwd=52dl 提取码：52dl<br /><br />

通过对训练集切分10%比例用于训练时验证模型<br />
训练好的模型对测试集进行测试评价效果<br />
经测试，简单的lstm在测试集accuracy能达到~81%<br />
