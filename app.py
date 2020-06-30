import keras
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import initializers
from  keras.models import Sequential, model_from_json
from  keras.layers import Dense
from keras.utils import plot_model
from keras import initializers
import keras.backend as K
import numpy as np
from numpy.fft import rfft, irfft, ifft,fft
from util import get_logger, make_train_img_matr, calc_out_nn, make_2d_arr,l_test_after_contr, calc_out_nn_n, matr_img
from keras.callbacks import LearningRateScheduler, History, ModelCheckpoint
import json
from keras.utils import generic_utils
import matplotlib.pyplot as plt
import logging
import sys


# from  keras.optimizers import
"""
Замечания: для Python3 с f строками(используются в логинге)
"""
class My_const_init(initializers.Initializer):
    def __init__(self,my_parm):
       self.m_p=my_parm
    def __call__(self,shape,dtype=None):
       return np.zeros(shape,dtype=dtype)+0.5674321
    def get_config(self):
       return {'my_parm':self.m_p}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class My_subcl_model_checkpoint(ModelCheckpoint):
    def __init__(self,loss_tresho, acc_shuren, filepath, loger):
        self.loss_threshold=loss_tresho
        self.acc_shureness=acc_shuren
        self.filepath=filepath
        self.loger=loger
    # def on_batch_begin(self, batch, logs=None):
        # print("logs",logs)
    # def on_epoch_begin(self, epoch, logs=None):
    #     print("logs",logs)
    def on_epoch_end(self, epoch, logs=None):
        print("logs", logs)
        loss=logs.get('loss')
        acc=logs.get('acc')
        if loss<=self.loss_threshold and acc==self.acc_shureness:
            print("Interrupted")
            self.model.save_weights(self.filepath, overwrite=True)
            with open("model_json.json", 'w') as f:
                f.write(self.model.to_json())
            print("Weights saved")
            loger.info("Weights saved")
            print("Model saved")
            loger.info("Model saved")
            sys.exit(0)




generic_utils._GLOBAL_CUSTOM_OBJECTS['My_const_init']=My_const_init(7)
generic_utils._GLOBAL_CUSTOM_OBJECTS['My_subcl_mod_checkpoint']=My_subcl_model_checkpoint

len_=10
stop=-1
push_i = 0
push_fl = 1
push_str = 2
cr_nn_ = 3
fit_ = 4
predict = 5
evalu_=6
determe_X_Y=7
cl_log=8
sav_model=9
load_model_wei=10
get_weis=11
on_contrary=12
make_X_matr_img_=13
make_img_=14
make_img_one_decomp=15
get_weis_to_json=16
load_json_wei_pr_fft=17
make_net=19
k_plot_model=20
k_summary=21
compile_net=22
fit_net=23
make_net_load_wei=24
make_net_on_contrary=25
plot_train=26
get_mult_class_matr=27
cr_sav_model_wei_best_callback=28
sav_model_wei=29
cr_callback_wi_loss_treshold_and_acc_shure=30

ops=("")  #  No need in console input in this programm


def console(prompt, progr=[],loger=None, date=None):
    if len(progr)==0:
        buffer = [0] * len_ * 2  # байт-код для шелл-кода
        input_ = '<uninitialized>'
        # splitted_cmd и splitted_cmd_src - т.к. работаем со статическим массивом
        splitted_cmd: list = [''] * 2
        splitted_cmd_src: list = None
        main_cmd = '<uninitialized>'
        par_cmd = '<uninitialized>'
        idex_of_bytecode_is_bytecode = 0
        cmd_in_ops = '<uninitialized>'
        we_exit = 'exit'
        we_run = 'r'
        pos_bytecode = -1
        shell_is_running = True
        print("Здравствуйте я составитель кода этой программы")
        print("r - выполнить")
        print("exit - выход")
        print("Доступные коды:")
        for c in ops:
            print(c, end=' ')
        print()
        while shell_is_running:
            input_ = input(prompt)
            # полностью выходим из программы
            if input_ == we_exit:
                break
            # выполняем байткод вирт-машиной
            elif input_ == we_run:
               pos_bytecode += 1
               buffer[pos_bytecode] = stop
               vm(buffer, loger, date)
               pos_bytecode = -1
        splitted_cmd_src = input_.split()
        for pos_to_write in range(len(splitted_cmd_src)):
            splitted_cmd[pos_to_write] = splitted_cmd_src[pos_to_write]
        main_cmd = splitted_cmd[0]
        par_cmd = splitted_cmd[1]
        # Ищем код в списке код-строку
        for idex_of_bytecode_is_bytecode in range(len(ops)):
            cmd_in_ops = ops[idex_of_bytecode_is_bytecode]
            is_index_inside_arr = idex_of_bytecode_is_bytecode < len(ops)
            if main_cmd == cmd_in_ops and is_index_inside_arr:
                pos_bytecode += 1
                # формируем числовой байт-код и если нужно значения параметра
                buffer[pos_bytecode] = idex_of_bytecode_is_bytecode
                if par_cmd != '':
                    pos_bytecode += 1
                    buffer[pos_bytecode] = par_cmd
                # очищаем
                splitted_cmd[0] = ''
                splitted_cmd[1] = ''
                break
    else:
       vm(progr,loger,date)
            
            
len_=256
X_matr_img=None
Y_matr_img=np.array([[0,1],[0,1],[0,1],[0,1]])
Y_matr_img_one=np.array([[0,1]])
and_X = [[1, 1], [1, 0], [0, 1], [0, 0]]
and_Y = [[1], [0], [0], [0]]
X_comp=np.array([[0,1,0,1,1]])
Y_comp=np.array([[0,1]])
X = np.array(and_X)
Y = np.array(and_Y)
X_t=None
Y_t=None
X_t=X  #  по умолчанию
Y_t=Y


def plot_history_(_file:str,history:History,name_gr:str,logger:logging.Logger):
    fig:plt.Figure=None
    ax:plt.Axes=None
    fig, ax=plt.subplots()
    plt.text(0.1, 1.1, name_gr)
    ax.plot(history.history["loss"],label="Уменьшение значения целевой функции")
    plt.plot(history.history['acc'],label="Доля верных ответов на обучающем наборе")
    if 'val_acc' in history.history:
      plt.plot(history.history['val_acc'],label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов / loss')
    ax.legend()
    plt.savefig(_file)
    print("Graphic saved")
    logger.info("Graphic saved")
    plt.show()


def vm(buffer:tuple,logger:logging.Logger, date:str)->None:
    global X_t, Y_t,d0_w,d1_w,d2_w, X_matr_img
    save_wei_best_callback:ModelCheckpoint=None
    threshold_callback=None
    model_obj: keras.models.Model = None
    history:History=None
    l_weis=[]
    logger.info(logger.debug(f'Log started {date}'))
    vm_is_running=True
    ip=0
    sp=-1
    sp_str=-1
    steck=[0]*len_
    steck_str=['']*len_
    op=buffer[ip]
    while vm_is_running:
        if op==stop:
            return
        if op == push_i:
            sp += 1
            ip += 1
            steck[sp] = int(buffer[ip])  # Из строкового параметра
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])  # Из строкового параметра
        elif op == push_str:
            sp_str += 1
            ip += 1
            steck_str[sp_str] = buffer[ip]
        elif op==determe_X_Y:
            var_Y = steck_str[sp_str]
            sp_str -= 1
            var_X = steck_str[sp_str]
            sp_str -= 1
            var_iner_Y = globals().get(var_Y, 'Not found %s'.format(var_Y))
            var_iner_X=globals().get(var_X, 'Not found %s'.format(var_Y))
            X_t=var_iner_X
            Y_t=var_iner_Y
            print("( Data found )")
            # print(f"Data X Y (found): \nX = {X}  \nY={Y}")
            # logger.debug(f'Data X Y (found): \nX = {X}  \nY={Y}')
        elif op == make_X_matr_img_:
            X_img: np.ndarray = None
            path_s = steck_str[sp_str]
            sp_str -= 1
            elems=steck[sp]
            sp-=1
            rows=steck[sp]
            sp-=1
            X_img = make_train_img_matr(path_s,rows,elems)
            X_matr_img=np.array(X_img)
            X_matr_img.astype('float32')
            X_matr_img/= 255.0
            print("X matr img",X_matr_img.tolist())
            logger.debug(f'shape X matr img: {X_matr_img.shape}')
            # X_matr_img-=np.mean(X_matr_img,axis=0,dtype='float64')
            # X_matr_img /= 255
            # X_matr_img=np.std(X_matr_img_n,axis=0,dtype='float64')
            # X_matr_img /= 255
            # loger.debug(f'd0  {d0.get_weights()}')
        elif op==predict:
            out_nn=model_obj.predict(X_t)
            print("Predict matr: ",out_nn)
            logger.info(f"Predict matr: {out_nn}")
        elif op==evalu_:
            out_ev=model_obj.evaluate(X_t, Y_t)
            print("Eval model: ",out_ev)
            logger.info(f"Eval model: {out_ev}")
        elif op==cl_log:
            with open('log.txt','w') as f:
                f.truncate()
            print("Log file cleared")
        elif op==sav_model:
            with open("model_json.json", 'w') as f:
                f.write(model_obj.to_json())
            print("Model saved")
            logger.info('Model saved')
        elif op==sav_model_wei:
            model_obj.save_weights("wei.h5", overwrite=True)
            print("Weights saved")
            logger.info('Weights saved')
        elif op==load_model_wei:
               loaded_json_model=''
               with open('model_json.json','r') as f:
                  loaded_json_model=f.read()
               model_obj=model_from_json(loaded_json_model)
               model_obj.load_weights('wei.h5')
               print("Loaded model and weights")
               logger.info("Loaded model and weights")
        elif op==get_weis:
            for i in range(len(model_obj.layers)):
               l_weis.append(model_obj.layers[i].get_weights())
        elif op==get_weis_to_json:
            wei_t=None
            for i in range(len(model_obj.layers)):
                l_weis.append(model_obj.layers[i].get_weights())
                wei_t=model_obj.layers[i].get_weights()
                ke_t,bi_t=wei_t
                tenz=[ke_t.tolist(),bi_t.tolist()]
                wei_dic={i:tenz}
                with open('weis_json.json','w') as f:
                    json.dump(wei_dic,f)
                    print("Json weights written")
                    logger.info("Json weights written")
        elif op==get_mult_class_matr:
            pix_am=steck[sp]
            sp-=1
            path_=steck_str[sp_str]
            sp_str-=1
            X_t, Y_t=matr_img(path_,pix_am)
            X_t=np.array(X_t)
            Y_t=np.array(Y_t)
        # elif op == make_img_:
        #     d0_w,d1_w,d2_w=l_weis
        #     d0_w=l_weis[0]
        #     model_new = Sequential()
        #     l0 = Dense(10000, activation=act_funcs[0], use_bias=True)
        #     l0.build((None,2))
        #     ke2,be2=d0_w
        #     bi2_n=np.zeros(10000)+be2[0]
        #     l0.set_weights([ke2.T,bi2_n])
        #     l0.set_weights([d2])
        #     print("ke2 ri",ke2)
        #     model_new.add(l0)
        #     l1 = Dense(10, activation=act_funcs[1], use_bias=True)
        #     l1.build((None,8))
        #     ke1,be1=d1_w
        #     be1_n=np.zeros(10)+be1[0]
        #     l1.set_weights([ke1.T,be1_n])
        #     model_new.add(l1)
        #     l2 = Dense(10000, activation=act_funcs[0], use_bias=True)
        #     l2.build((None,10))
        #     ke0,bi0=d0_w
        #     bi0_n=np.zeros(10000)+bi0[0]
        #     l2.set_weights([ke0.T,bi0_n])
        #     model_new.add(l2)
        #     out_nn = model_new.predict(np.array([[0,1]]))
        #     loger.debug("out_nn",str(out_nn))  # Похоже 10_000 массивы трудно логирует
        #     print("out_nn", out_nn.tolist())
        #     l_test_after_contr_=l_test_after_contr(out_nn.tolist()[0],10000)
        #     print("l test af contr",l_test_after_contr_)
        #     print(all([0.00010001]*10000==out_nn.tolist()))
        #     vec_tested = calc_out_nn(l_test_after_contr_)
        #     print("vec tested",vec_tested)
        #     print(X_matr_img[0].tolist()==vec_tested)
        #     _2d_img: np.ndarray = make_2d_arr(vec_tested)
        #     vec_tested_np=np.array(vec_tested)
        #     img_prep=vec_tested_np.reshape(100,100)
        #     img_prep=img_prep.astype('uint8')
        #     new_img = Image.fromarray(img_prep,'L')
        #     new_img.save("img_net_creative.png")
        #     print("Img written")
        #     logger.info("Img written")
        #     loger.debug("in make_img")
        # elif op == make_img_one_decomp:
        #     dw0, ke0= l_weis[0]
        #     model_new = Sequential()
        #     l0 = Dense(10000, activation=act_funcs[0], use_bias=True)
        #     l0.build((None, 2))
        #     model_new.add(l0)
        #     ke0_n=np.zeros(10000)+ke0[0]
        #     l0.set_weights([dw0.T,ke0_n])
        #     out_nn = model_new.predict(np.array([[0,1]]))
        #     loger.debug("in make_img")
        #     loger.debug("out_nn",str(out_nn))  # Похоже 10_000 массивы трудно логирует
        #     print("out_nn", out_nn.tolist()[0])
        #     l_test_after_contr_=l_test_after_contr(out_nn.tolist()[0],10000)
        #     print("l test af contr",l_test_after_contr_)
        #     print(all([0.00010001]*10000==out_nn.tolist()))
        #     vec_tested = calc_out_nn(l_test_after_contr_)
        #     print("vec tested",vec_tested)
        #     print(X_matr_img[0].tolist()==vec_tested)
        #     _2d_img: np.ndarray = make_2d_arr(vec_tested)
        #     vec_tested_np=np.array(vec_tested)
        #     img_prep=vec_tested_np.reshape(100,100)
        #     img_prep=img_prep.astype('uint8')
        #     new_img = Image.fromarray(img_prep,'L')
        #     new_img.save("img_net.png")
        #     print("Img written")
        elif op==load_json_wei_pr_fft:
            json_data:dict=None
            with open("weis_json.json","r") as f:
                # for i in json.load(f):
                    json_data=json.load(f)
                    print(type(json_data))
            matrix=json_data.get('0')[0]
            print("matrix",repr(matrix))
            four_tow_data=rfft(matrix,axis=1)
            print("four_tow_data",four_tow_data)
            four_i_data=irfft(four_tow_data,axis=1)
            print("i four tow",four_i_data)
        elif op==make_net:
            if op == make_net:  # Ex:  make_net ('S', ('D','D','D'), (3, 2, 4), ('relu','sigmoid', 'softmax'), ('use_bias_1', 'use_bias_1', 'use_bias_1')))
                l_tmp = None
                acts_di:dict=None
                acts_di={'s':'sigmoid','r':'relu','t':'tanh','S':'softmax'}
                use_bias_ = False
                ip += 1
                arg = buffer[ip]
                type_m, denses, inps, acts, use_bi, kern_init = arg
                if type_m == 'S':
                    model_obj = Sequential()
                for i in range(len(denses)):
                    if denses[i] == 'D':
                        splt_bi = use_bi[i].split('_')
                        if splt_bi[-1] == '1':
                            use_bias_ = True
                        elif splt_bi[-1] == '0':
                            use_bias_ = False
                        if i == 0:
                            l_tmp = Dense(inps[i + 1], input_dim=inps[0], activation=acts_di.get(acts[i]), use_bias=use_bias_,
                                          trainable=True, kernel_initializer=kern_init)
                        else:
                            l_tmp = Dense(inps[i + 1],input_dim=inps[i],activation=acts_di.get(acts[i]), use_bias=use_bias_,
                                           trainable=True, kernel_initializer=kern_init)
                        model_obj.add(l_tmp)

        elif op==make_net_on_contrary:
            l_tmp = None
            acts_di: dict = None
            acts_di = {'s': 'sigmoid', 'r': 'relu', 't': 'tanh', 'S': 'softmax'}
            use_bias_ = False
            ip += 1
            arg = buffer[ip]
            type_m, denses, inps, acts, use_bi, kern_init = arg
            if type_m == 'S':
                model_obj = Sequential()
            for i in range(len(denses)-1,-1,-1):
                if denses[i] == 'D':
                    splt_bi = use_bi[i].split('_')
                    if splt_bi[-1] == '1':
                        use_bias_ = True
                    elif splt_bi[-1] == '0':
                        use_bias_ = False
                    if i==len(denses)-1:
                       l_tmp = Dense(inps[i],input_dim=inps[i+1], activation=acts_di.get(acts[i]),
                                      use_bias=use_bias_,
                                      kernel_initializer=kern_init)
                       l_tmp.build((None, inps[i+1]))
                    else:
                        l_tmp = Dense(inps[i],activation=acts_di.get(acts[i]), use_bias=use_bias_,
                                     kernel_initializer=kern_init)
                        l_tmp.build((None, inps[i+1]))
                    if use_bias_:
                        # Only if we have biases
                        wei_t=l_weis[i]
                        ke,bi=wei_t
                        bi_n=np.zeros(inps[i])+bi[0]
                        l_tmp.set_weights([ke.T, bi_n])
                    else:
                        raise RuntimeError("Without biases on-contrary net not implemented")
                model_obj.add(l_tmp)
            print("On-contrary net created")
            logger.info("On-contrary net created")
        elif op==k_plot_model:
            plot_model(model_obj, to_file='model.png', show_shapes=True)
        elif op==k_summary:
            model_obj.summary()
        elif op==plot_train:
            ip+=1
            arg=buffer[ip]
            plot_history_('./graphic/train_graphic.png', history, arg, logger)
        elif op==compile_net:
            ip+=1
            arg=buffer[ip]
            opt, loss_obj, metrics=arg
            model_obj.compile(optimizer=opt, loss=loss_obj, metrics=metrics)
        elif op==fit_net:  # 1-ep 2-bach_size 3-validation_split 4-shuffle 5-callbacks
            ip+=1
            arg=buffer[ip]
            ep,ba_size,val_spl,shuffle,callbacks=arg
            if save_wei_best_callback:
                callbacks.append(save_wei_best_callback)
            if threshold_callback:
                callbacks.append(threshold_callback)
            history=model_obj.fit(X_t, Y_t, epochs=ep, batch_size=ba_size, validation_split=val_spl, shuffle=shuffle, callbacks=callbacks)
        elif op == cr_sav_model_wei_best_callback:
            wei_file = 'wei.h5'
            monitor='<uninitialize>'
            save_best_only=True
            ip+=1
            arg=buffer[ip]
            monitor=arg
            save_wei_best_callback=ModelCheckpoint(wei_file, monitor,save_best_only=True, period=1, verbose=1, save_weights_only=True)
        elif op == cr_callback_wi_loss_treshold_and_acc_shure:
            wei_file = 'wei.h5'
            ip+=1
            arg=buffer[ip]
            loss_threshold,acc_shureness=arg
            threshold_callback=My_subcl_model_checkpoint(loss_threshold, acc_shureness, wei_file, logger)
        else:
            raise RuntimeError("Unknown bytecode -> %d."%op)
        ip+=1
        try:
          op=buffer[ip]
        except IndexError:
            raise RuntimeError('It seems somewhere'
                               ' skipped argument of bytecode.')


opt = SGD(lr=0.01)
# 1-optimizer 2-loss_function 3-metrics
compile_pars = (opt, 'mse', ['accuracy'])
monitor_pars=('val_accuracy')
def adap_lr(epoch):
    return 0.01*epoch
my_lr_scheduler=LearningRateScheduler(adap_lr)
# 1-ep 2-bach_size 3-validation_split 4-shuffle 5-callbacks
fit_pars=(100, 1, 1, False, [my_lr_scheduler])
my_init=My_const_init(9)
ke_init=("glorot_uniform",my_init)


if __name__ == '__main__':
    loger, date=get_logger("debug","log.txt",__name__,'a')
    p1=(make_net,('S', ('D','D'), (2,3,1),('t','s'), ('use_bias_1','use_bias_1'),ke_init[1]),
        compile_net,("adam",compile_pars[1],compile_pars[2]),
        cr_callback_wi_loss_treshold_and_acc_shure, (0.001, 1),
        fit_net,(100,fit_pars[1],fit_pars[2],fit_pars[3], fit_pars[4]),
        sav_model,
        sav_model_wei,
        stop
        )
    p2=(load_model_wei,compile_net,(compile_pars[0],compile_pars[1],compile_pars[2]),evalu_,predict,stop)
    p17=(push_i,10000,push_str,r'B:\msys64\home\msys_u\code\python\keras_shel_co\train_ann\train',get_mult_class_matr,
         make_net,('S', ('D'), (10000,2),('S','S'), ('use_bias_1','use_bias_1','use_bias_1'),ke_init[1]),
         k_summary,
         compile_net,(compile_pars[0],"binary_crossentropy",compile_pars[2]),
         cr_callback_wi_loss_treshold_and_acc_shure, (0.001, 1),
         fit_net,(100,5,fit_pars[2],fit_pars[3], fit_pars[4]),
         stop)
    p18=(push_i,10000,push_str,r'B:\msys64\home\msys_u\code\python\keras_shel_co\train_ann\ask',get_mult_class_matr,
         load_model_wei,
         compile_net,("adam","binary_crossentropy",compile_pars[2]),
         evalu_,
         predict,
         stop)
    # если используем cr_sav_model_wei_best_callback и fit - не надо сохранять веса
    p19 = (
    push_i, 10000, push_str, r'B:\msys64\home\msys_u\code\python\keras_shel_co\train_ann\train', get_mult_class_matr,
    cr_sav_model_wei_best_callback,('loss'),
    make_net, ('S', ('D', 'D', 'D'), (10000, 800, 2), ('s', 's', 'S'), ('use_bias_1', 'use_bias_1', 'use_bias_1'),
               ke_init[1]),
    k_summary,
    compile_net, (compile_pars[0], compile_pars[1], compile_pars[2]),
    fit_net, (fit_pars[0], fit_pars[1], fit_pars[2], fit_pars[3], fit_pars[4]),
    evalu_,
    predict,
    sav_model,
    plot_train, "Tuples and Circs",
    stop)
    console('>>>', p17, loger, date)



