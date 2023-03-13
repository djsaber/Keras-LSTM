# coding=gbk
import keras.backend as K
from keras.layers import Layer, LSTM, Embedding, Dense, RNN, Input
from keras.models import Model
from keras import activations


class My_LSTM_Cell(Layer):
    def __init__(
        self, 
        units,
        activation='tanh',
        recurrent_activation='hard_sigmoid',
        use_bias=True,
        **kwargs
        ):
        super(My_LSTM_Cell, self).__init__(**kwargs)
        self.units = units
        self.state_size = [units, units]
        self.activation=activations.get(activation)
        self.recurrent_activation=activations.get(recurrent_activation)
        self.use_bias=use_bias
    
    def build(self, input_shape):
        super().build(input_shape)
        self.wf = self.add_weight(name='wf', shape=(input_shape[-1]+self.units, self.units,))
        self.wi = self.add_weight(name='wi', shape=(input_shape[-1]+self.units, self.units,))
        self.wc = self.add_weight(name='wc', shape=(input_shape[-1]+self.units, self.units,))
        self.wo = self.add_weight(name='wo', shape=(input_shape[-1]+self.units, self.units,))
        if self.use_bias:
            self.bf = self.add_weight(name='bf', shape=(self.units,),)
            self.bi = self.add_weight(name='bi', shape=(self.units,),)
            self.bc = self.add_weight(name='bc', shape=(self.units,),)
            self.bo = self.add_weight(name='bu', shape=(self.units,),)

    def call(self, inputs, states):
        h = states[0]
        c = states[1]
        i = K.concatenate([h, inputs])
        ft = self.recurrent_activation(K.dot(i, self.wf))
        it = self.recurrent_activation(K.dot(i, self.wc))
        ct_hat = self.activation(K.dot(i, self.wi))
        ot = self.recurrent_activation(K.dot(i, self.wo))
        if self.use_bias:
            ft = K.bias_add(ft, self.bf)
            ct_hat = K.bias_add(ct_hat, self.bc)  
            it = K.bias_add(it, self.bi)  
            ot = K.bias_add(ot, self.bo)  
        c = c*ft + ct_hat*it
        h = self.activation(c) * ot
        return h, [h, c]



class My_LSTM_layer(Layer):
    def __init__(
        self, 
        units,
        activation = 'tanh',
        recurrent_activation = 'hard_sigmoid',
        use_bias = True,
        return_sequences = False,
        **kwargs
        ):
        super(My_LSTM_layer, self).__init__(**kwargs)
        self.return_sequences = return_sequences
        self.units = units
        self.cell = My_LSTM_Cell(units, activation, recurrent_activation, use_bias)
        self.layer = RNN(self.cell, return_sequences=return_sequences)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        x = self.layer(inputs)
        return x



class My_LSTM(Model):
    def __init__(
        self, 
        voc_size = 1000,
        vec_dim=64,
        max_len = 50,
        units=128,
        output_dim=1,
        *args, 
        **kwargs
        ):
        super(My_LSTM, self).__init__(*args, **kwargs)
        self.emb = Embedding(input_dim=voc_size, output_dim=vec_dim, input_length=max_len)
        # Keras实现好的LSTM层（快） / 我们自己实现的LSTM层（慢）
        # self.lstm = LSTM(units=units, return_sequences=False)
        self.lstm = My_LSTM_layer(units=units, return_sequences=False)
        self.dense = Dense(units=output_dim, activation='sigmoid')

    def call(self, inputs):
        x = self.emb(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)
        self.call(Input(input_shape[1:]))
