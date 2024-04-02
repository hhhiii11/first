import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, target_update_frequency=100):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)

        self._target_update_frequency = target_update_frequency
        self._target_model = self._build_model(num_layers, width)
        self._target_model.set_weights(self._model.get_weights())  # 初始化目标网络权重与原始网络相同

 

    
    def _build_model(self, num_layers, width):
        """
        构建神经网络
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)

        #outputs = layers.Dense(self._output_dim, activation='linear')(x)

        value_stream = layers.Dense(1)(x)
        advantage_stream = layers.Dense(self._output_dim)(x)
        outputs = value_stream + (advantage_stream - tf.reduce_mean(advantage_stream, axis=1, keepdims=True))


        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(learning_rate=self._learning_rate))
        return model
    

    def update_target_model(self):
        """
        更新目标网络的权重
        """
        self._target_model.set_weights(self._model.get_weights())
        

    def predict_one(self, state, target=False):
        """
        预测给定状态的动作值函数
        """
        state = np.reshape(state, [1, self._input_dim])
        if target:
            return self._target_model.predict(state)
        else:
            return self._model.predict(state)
    """
    def predict_one(self, state, target=False):
        
        #从单个状态预测值
        
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)
    """
    
    def predict_batch(self, states):
        
        #从经验组预测值
        
        return self._model.predict(states)
    

    def train_batch(self, states, q_sa):
        """
        使用更新的 q 值训练 cnn
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def train_target(self, states, q_sa):
        """
        使用经验数据对目标网络进行训练
        """
        self._target_model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        将当前模型保存为 h5 文件，并将模型结构保存为 png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        加载存储在指定的文件夹中的模型（如果存在）
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        从单个状态预测操作值
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim