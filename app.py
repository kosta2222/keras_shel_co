from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from  keras.models import Sequential, model_from_json
from  keras.layers import Dense
from  keras.layers import Layer

import numpy as np
from util import get_logger
import datetime as d
"""
Замечания: для Python3 с f строками(используются в логинге)
"""

or_X = [[1, 1], [1, 0], [0, 1], [0, 0]]
or_Y = [[1], [1], [1], [0]]
X = np.array(or_X)
Y = np.array(or_Y)

act_funcs=('tanh','tanh','sigmoid')
init_lays=(2, 3, 3, 1)
us_bias=(False, True)
len_nn_lays=len(init_lays) - 1
opt=SGD(lr=0.07)
compile_pars=(opt, 'mse', ['accuracy'] )
monitor_pars=('val_accuracy')
fit_pars=(80, 1)
def my_init(shape,dtype=None):
    return np.zeros(shape,dtype=dtype)+0.5674321
def create_nn():
    model = Sequential()
    d0=Dense(init_lays[1], input_dim=init_lays[0], activation=act_funcs[0],use_bias=us_bias[1],kernel_initializer=my_init)
    model.add(d0)
    d1=Dense(init_lays[2], activation=act_funcs[1], use_bias=us_bias[1],kernel_initializer=my_init)
    model.add(d1)
    d2=Dense(init_lays[3], activation=act_funcs[2], use_bias=us_bias[1],kernel_initializer=my_init)
    model.add(d2)
    return model,d0,d1,d2
def fit_nn(X,Y):
    es=EarlyStopping(monitor=monitor_pars[0])
    model_obj.compile(optimizer=compile_pars[0], loss=compile_pars[1], metrics=compile_pars[2])
    model_obj.fit(X, Y, epochs=fit_pars[0], validation_split=fit_pars[1])


def pred(X):
  return model_obj.predict(X)

def evalu(X, Y):
    los_func_metr_and_scores=model_obj.evaluate(X, Y)
    return los_func_metr_and_scores

or_X=[[1, 1], [1, 0], [0, 1], [0, 0]]
or_Y=[[1], [1], [1], [0]]
or_X_np=np.array(or_X)
or_Y_np=np.array(or_Y)

len_=10
push_i = 0
push_fl = 1
push_str = 2
cr_nn_ = 3
fit_ = 4
predict = 5
evalu_=6
determe_X_Y=7
cl_log=8
sav_model_wei=9
load_model_wei=10
get_weis=11
on_contrary=12
stop = 13  # stop добавляется в скрипте если we_run
ops=("push_i","push_fl", "push_str", "cr_nn", "fit", "predict","evalu","determe_X_Y","cl_log","sav_model_wei","load_model_wei","get_weis","on_contrary")
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
X_t=X  #  по умолчанию
Y_t=Y
model_obj=None
d0:Dense=None
d1:Dense=None
d2:Dense=None
def vm(buffer,logger, date):
    global model_obj, X_t, Y_t,d0,d1,d2
    # logger=get_logger(level)
    # today=d.datetime.today()
    # today_s=today.strftime('%x %X')
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
        elif op==cr_nn_:
           model_obj,d0, d1, d2=create_nn()
           print("Model created ",model_obj)
           d0=d0
           d1=d1
           d2=d2
           logger.debug(f'Model created {model_obj}')
        elif op==determe_X_Y:
            var_Y = steck_str[sp_str]
            sp_str -= 1
            var_X = steck_str[sp_str]
            sp_str -= 1
            var_iner_Y = globals().get(var_Y, 'Not found %s'.format(var_Y))
            var_iner_X=globals().get(var_X, 'Not found %s'.format(var_Y))
            X_t=var_iner_X
            Y_t=var_iner_Y
            print(f"Data X Y (found): \nX = {X}  \nY={Y}")
            logger.debug(f'Data X Y (found): \nX = {X}  \nY={Y}')
        elif op==fit_:
            fit_nn(X_t, Y_t)
            loger.debug(f'd0  {d0.get_weights()}')
        elif op==predict:
            out_nn=pred(X_t)
            print("Predict matr: ",out_nn)
            logger.info(f"Predict matr: {out_nn}")

        elif op==evalu_:
            out_ev=evalu(X_t, Y_t)
            print("Eval model: ",out_ev)
            logger.info(f"Eval model: {out_ev}")
        elif op==cl_log:
            with open('log.txt','w') as f:
                f.truncate()
            print("Log file cleared")
        elif op==sav_model_wei:
            with open("model_json.json", 'w') as f:
                f.write(model_obj.to_json())
            print("Model saved")
            logger.info('Model saved')
            model_obj.save_weights("wei.h5")
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
            l=[]
            # d0=d0.get_weights()
            # d1=d1.get_weights()
            # d2=d2.get_weights()
            for i in range(len(model_obj.layers)):
               l.append(model_obj.layers[i].get_weights())
            d0,d1,d2=l
            loger.debug(f'd0 {d0}\n')
            loger.debug(f'd1 {d1}\n')
            loger.debug(f'd2 {d2}\n')
        elif op==on_contrary:
            model1 = Sequential()
            l0 = Dense(3, activation='sigmoid', use_bias=True)
            l0.build((None,1))
            ke2, bi2=d2
            # d2=np.array(d2)
            # d2=d2.T
            # d2=np.hstack((ke2.T,np.array([bi2])))
            l0.set_weights(np.array([ke2.T]))
            # l0.set_weights([d2])

            print("ge wei",l0.get_weights())
            # model1.add(l0)
            # model1.layers[0].set_weights(d2)
            # d1 = Dense(3, activation='tanh')
            # d1.set_weights(d1)
            # model1.add(d1)
            # d2 = Dense(2, activation='sigmoid')
            # model1.set_weights(d0)
            # model1.add(d2)
            # loger.info(f'predict: {model1.predict([[1]])}\n')
        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+=1
        op=buffer[ip]

if __name__ == '__main__':
    loger, date=get_logger("debug","log.txt",__name__)
    p1=(cr_nn_,fit_,predict,sav_model_wei,stop)
    p2=(load_model_wei,get_weis,on_contrary,stop)
    console('>>>','debug', p1, loger, date)



