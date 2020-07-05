import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import initializers
from keras.utils import generic_utils
import sys


class My_const_init(initializers.Initializer):
    """
    Мой постоянный инициализатор
    """
    def __init__(self,my_parm):
        self.m_p=my_parm
    def __call__(self,shape,dtype=None):
        return np.zeros(shape,dtype=dtype)+0.5674321
    def get_config(self): # тест
        return {'my_parm':self.m_p}
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class My_subcl_model_checkpoint(ModelCheckpoint):
    """
    Прерывание обучения по условиям ( в зависимости от знания loss threshold и acc - доли верных ответов на обучающем наборе)
    """
    def __init__(self,loss_tresho, acc_shuren, filepath, loger):
        self.loss_threshold=loss_tresho
        self.acc_shureness=acc_shuren
        self.filepath=filepath
        self.loger=loger
    def on_epoch_end(self, epoch, logs=None):
        """ При окончании переборки пакета сравниваем loss и acc
        прирываем если условие подходит"""
        print("logs", logs)
        loss=logs.get('loss')
        acc=logs.get('acc')
        if loss<=self.loss_threshold and acc==self.acc_shureness:
            print("Interrupted")
            self.model.save_weights(self.filepath, overwrite=True)
            with open("model_json.json", 'w') as f:
                f.write(self.model.to_json())
            print("Weights saved")
            self.loger.info("Weights saved")
            print("Model saved")
            self.loger.info("Model saved")
            sys.exit(0)

# Для того чтобы можно было сериализовать мои настроечные обьекты
generic_utils._GLOBAL_CUSTOM_OBJECTS['My_const_init']=My_const_init(7)

