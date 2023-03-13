# coding=gbk

from model import My_LSTM
from keras.datasets import imdb
from keras.utils import pad_sequences


#---------------------------------���ò���-------------------------------------
voc_size = 20000        # �ʱ��С
vec_dim = 64            # ��������ά��
max_len = 80            # ���ӳ���
lstm_units = 128        # LSTM����ά��
output_dim = 1          # �������ά��
batch_size = 128        # �������δ�С
#-----------------------------------------------------------------------------


#----------------------------------����·��-----------------------------------
data_path = "D:/����/python����/�����ֲ�/LSTM/datasets/IMDB/imdb.npz"
load_path = "D:/����/python����/�����ֲ�/LSTM/save_models/lstm_imdb.h5"
#-----------------------------------------------------------------------------


#----------------------------------�������ݼ�-----------------------------------
_, (testX, testY) = imdb.load_data(path=data_path, num_words=voc_size)
print('testX shape:', testX.shape)
print('testY shape:', testY.shape)
#-----------------------------------------------------------------------------


#---------------------------����Ԥ�����ضϻ���Ϊ�ȳ�---------------------------
testX = pad_sequences(testX, maxlen=max_len)
print('testX shape:', testX.shape)
#-----------------------------------------------------------------------------


#-----------------------------------�ģ��----------------------------------
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
    optimizer='adam',                    # adam�Ż���
    loss='binary_crossentropy',          # ��Ԫ��������ʧ
    metrics='acc'                        # ׼ȷ��ָ��
    )
lstm.load_weights(load_path)
#-----------------------------------------------------------------------------


#--------------------------------��ȡȨ�أ�����---------------------------------
lstm.evaluate(
    testX, 
    testY, 
    batch_size=batch_size, 
    )
#-----------------------------------------------------------------------------