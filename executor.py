import logging
import keras
from consts import stop,nparray,make_net,determe_X_Y,push_obj,predict,push_i,push_str, \
    plot_train,sav_model_wei,sav_model,fit_net,get_mult_class_matr,cr_callback_wi_loss_treshold_and_acc_shure, \
    evalu_,load_model_wei,cr_sav_model_wei_best_callback,compile_net,get_weis,k_summary,push_fl,make_net_on_contrary
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from util import matr_img
import numpy as np
from keras.callbacks import History, ModelCheckpoint
from plot_history import plot_history_
from my_keras_customs import My_subcl_model_checkpoint

"""
Замечания: для Python3 с f строками (используются в логинге)
"""

def exec(buffer:tuple,logger:logging.Logger, date:str)->None:
    """
    Исполнитель байт-кода (управляющих команд)
    :param buffer: буфер команд
    :param logger: обьект логер
    :param date: сегодняшняя дата
    :return: None
    """
    X_t=None
    Y_t=None
    len_=256
    save_wei_best_callback:ModelCheckpoint=None
    threshold_callback=None
    model_obj: keras.models.Model = None
    history:History=None
    l_weis=[]
    logger.info(logger.debug(f'Log started {date}'))
    vm_is_running=True
    ip=0
    sp=-1
    steck=[0]*len_
    op=buffer[ip]
    while vm_is_running:
        #------------основные коды памяти---------------
        if op==stop:
            return
        if op == push_i:
            sp += 1
            ip += 1
            steck[sp] = int(buffer[ip])
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])
        elif op == push_str:
            sp += 1
            ip += 1
            steck[sp] = buffer[ip]
        elif op== push_obj:
            sp+=1
            ip+=1
            steck[sp]=buffer[ip]
        #------------------------------------
        elif op==determe_X_Y:
            Y_t=steck[sp]
            sp-=1
            X_t=steck[sp]
            sp-=1
        elif op==nparray:
            st_arg=steck[sp]
            sp-=1
            sp+=1
            steck[sp]=np.array(st_arg)
        elif op==predict:
            out_nn=model_obj.predict(X_t)
            print("Predict matr: ",out_nn)
            logger.info(f"Predict matr: {out_nn}")
        elif op==evalu_:
            out_ev=model_obj.evaluate(X_t, Y_t)
            print("Eval model: ",out_ev)
            logger.info(f"Eval model: {out_ev}")
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
        elif op==get_mult_class_matr:
            pix_am=steck[sp]
            sp-=1
            path_=steck[sp]
            sp-=1
            X_t, Y_t=matr_img(path_,pix_am)
            X_t=np.array(X_t)
            Y_t=np.array(Y_t)
            X_t.astype('float32')
            Y_t.astype('float32')
            X_t/=255
        elif op==make_net:
            if op == make_net:
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
                        # my_kern_reg=regularizers.l1(0.0001)
                        my_kern_reg=None
                        if i == 0:
                            l_tmp = Dense(inps[i + 1], input_dim=inps[0], activation=acts_di.get(acts[i]), use_bias=use_bias_,
                                          trainable=True, kernel_initializer=kern_init, kernel_regularizer=my_kern_reg)
                        else:
                            l_tmp = Dense(inps[i + 1],input_dim=inps[i],activation=acts_di.get(acts[i]), use_bias=use_bias_,
                                          trainable=True, kernel_initializer=kern_init, kernel_regularizer=my_kern_reg)
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

