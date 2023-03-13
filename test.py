# coding=gbk

from model import My_LSTM
from keras.datasets import imdb
from keras.utils import pad_sequences


#---------------------------------设置参数-------------------------------------
voc_size = 20000        # 词表大小
vec_dim = 64            # 单词向量维度
max_len = 80            # 句子长度
lstm_units = 128        # LSTM隐层维度
output_dim = 1          # 输出向量维度
batch_size = 128        # 测试批次大小
#-----------------------------------------------------------------------------


#----------------------------------设置路径-----------------------------------
data_path = "D:/科研/python代码/炼丹手册/LSTM/datasets/IMDB/imdb.npz"
load_path = "D:/科研/python代码/炼丹手册/LSTM/save_models/lstm_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------加载数据集-----------------------------------
_, (testX, testY) = imdb.load_data(path=data_path, num_words=voc_size)
print('testX shape:', testX.shape)
print('testY shape:', testY.shape)
#-----------------------------------------------------------------------------


#---------------------------序列预处理，截断或补齐为等长---------------------------
testX = pad_sequences(testX, maxlen=max_len)
print('testX shape:', testX.shape)
#-----------------------------------------------------------------------------


#-----------------------------------搭建模型----------------------------------
lstm = My_LSTM(
    voc_size = voc_size,
    vec_dim = vec_dim,
    max_len = max_len,
    units = lstm_units,
    output_dim = output_dim, 
    )
lstm.build(input_shape = (None, max_len))
lstm.summary()
lstm.compile(
    optimizer='adam',                    # adam优化器
    loss='binary_crossentropy',          # 二元交叉熵损失
    metrics='acc'                        # 准确率指标
    )
lstm.load_weights(load_path)
#-----------------------------------------------------------------------------


#--------------------------------读取权重，测试---------------------------------
lstm.evaluate(
    testX, 
    testY, 
    batch_size=batch_size, 
    )
#-----------------------------------------------------------------------------