from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from  keras.models import Sequential, model_from_json
from  keras.layers import Dense
import numpy as np
from util import get_logger, make_train_img_matr, calc_out_nn, make_2d_arr,l_test_after_contr
from keras.callbacks import LearningRateScheduler
from PIL import Image

# from  keras.optimizers import
"""
Замечания: для Python3 с f строками(используются в логинге)
"""
act_funcs=('relu','relu','softmax')
# init_lays=(10000, 10, 8, 1)
init_lays=(4, 3, 3, 2)
# init_lays=(10000,1)
# init_lays=(3,2)
us_bias=(False, True)
len_nn_lays=len(init_lays) - 1
opt=SGD(lr=0.01)
compile_pars=(opt, 'categorical_crossentropy', ['accuracy'] )
monitor_pars=('val_accuracy')
fit_pars=(120, 1)
def adap_lr(epoch):
    return 0.001*epoch
my_lr_scheduler=LearningRateScheduler(adap_lr, 1)
def my_init(shape,dtype=None):
    return np.zeros(shape,dtype=dtype)+0.5674321
ke_init=("glorot_uniform",my_init)
def create_nn(is_my_init):
    k_i=None
    if is_my_init:
      k_i=ke_init[1]
    else:
        k_i=ke_init[0]
    model = Sequential()
    d0=Dense(init_lays[1], input_dim=init_lays[0], activation=act_funcs[0],use_bias=us_bias[1],kernel_initializer=k_i)
    model.add(d0)  # d0.input_shape
    # print("in create nn d0 pars")
    d1=Dense(init_lays[2], activation=act_funcs[1], use_bias=us_bias[1],kernel_initializer=k_i)
    model.add(d1)
    d2=Dense(init_lays[3], activation=act_funcs[2], use_bias=us_bias[1],kernel_initializer=k_i)
    model.add(d2)
    return model#,d1,d2
def fit_nn(X,Y):
    es=EarlyStopping(monitor=monitor_pars[0])
    model_obj.compile(optimizer=compile_pars[0], loss=compile_pars[1], metrics=compile_pars[2])
    model_obj.fit(X, Y, epochs=fit_pars[0], validation_split=fit_pars[1],callbacks=[my_lr_scheduler])


def pred(X):
  return model_obj.predict(X)

def evalu(X, Y):
    los_func_metr_and_scores=model_obj.evaluate(X, Y)
    return los_func_metr_and_scores
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
make_X_matr_img_=13
make_img_=14
make_img_one_decomp=15
stop = 16  # stop добавляется в скрипте если we_run
ops=("push_i","push_fl", "push_str", "cr_nn", "fit", "predict","evalu","determe_X_Y","cl_log","sav_model_wei","load_model_wei","get_weis","on_contrary","make_X_matr_img","make_img","make_img_one_decomp")
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
model_obj=None
d0_w=None
d1_w=None
d2_w=None
X_matr_img=None
Y_matr_img=np.array([[1],[1],[1],[1]])
Y_matr_img_one=np.array([[1]])
or_X = [[1, 1], [1, 0], [0, 1], [0, 0]]
or_Y = [[1], [1], [1], [0]]
X_comp=np.array([[0,1,0,1]])
Y_comp=np.array([[0,1]])
X = np.array(or_X)
Y = np.array(or_Y)
X_t=None
Y_t=None
X_t=X  #  по умолчанию
Y_t=Y
def vm(buffer,logger, date):
    global model_obj, X_t, Y_t,d0_w,d1_w,d2_w, X_matr_img
    l_weis=[]
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
           model_obj_=create_nn(True)
           model_obj=model_obj_
           print("Model created ",model_obj)
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
            X_matr_img /= 255
        elif op==fit_:
            fit_nn(X_t, Y_t)
            # loger.debug(f'd0  {d0.get_weights()}')
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
            try:
               # loaded_json_model=''
               # with open('model_json.json','r') as f:
               #    loaded_json_model=f.read()
               # model_obj=model_from_json(loaded_json_model)
               model_obj=create_nn(False)
               model_obj.load_weights('wei.h5')
               print("Loaded model and weights")
               logger.info("Loaded model and weights")
            except Exception as e:
                print("Exc in load_model_wei")
                print(e.args)
                ip+=1
                op=buffer[ip]
                continue
        elif op==get_weis:
            # l=[]
            # d0=d0.get_weights()
            # d1=d1.get_weights()
            # d2=d2.get_weights()
            for i in range(len(model_obj.layers)):
               l_weis.append(model_obj.layers[i].get_weights())
            # d0_w,d1_w,d2_w=l
            # loger.debug(f'd0 {d0_w}\n')
            # loger.debug(f'd1 {d1_w}\n')
            # loger.debug(f'd2 {d2_w}\n')
        elif op==on_contrary:
            # from keras.constraints import max_norm
            # m=Sequential()
            # t=Dense(3, input_dim=1,use_bias=True, activation='relu')
            # m.add(t)
            # print("t wei",t.get_weights())
            #
            d0_w,d1_w,d2_w=l_weis
            # d2_w=l_weis[0]
            model_new = Sequential()
            l0 = Dense(3, activation=act_funcs[2], use_bias=True)
            l0.build((None,2))
            ke2, bi2=d2_w
            bi2_n=np.zeros(3)+bi2[0]
            print("bi2",bi2)
            l0.set_weights([ke2.T,bi2_n])
            # l0.set_weights([d2])
            # print("ke2 ri",ke2)
            model_new.add(l0)
            l1 = Dense(3, activation=act_funcs[1], use_bias=True)
            l1.build((None,3))
            ke1,bi1=d1_w
            l1.set_weights([ke1.T,bi1])
            model_new.add(l1)
            l2 = Dense(4, activation=act_funcs[0], use_bias=True)
            l2.build((None,3))
            ke0,be0=d0_w
            be0_n=np.zeros(4)+be0[0]
            l2.set_weights([ke0.T,be0_n])
            model_new.add(l2)
            out_nn=model_new.predict(np.array([[0,1]]))
            loger.info(f'predict cont {out_nn}')
            tes_out_nn=l_test_after_contr(out_nn.tolist()[0],4)
            logger.debug(f'vec izn {tes_out_nn}')
        elif op == make_img_:
            dw0,dw1,dw2=l_weis
            model_new = Sequential()
            l0 = Dense(8, activation=act_funcs[2], use_bias=True)
            l0.build((None,1))
            ke2,be2=d2_w
            bi2_n=np.zeros(8)+be2[0]
            l0.set_weights([ke2.T,bi2_n])
            # l0.set_weights([d2])
            # print("ke2 ri",ke2)
            model_new.add(l0)
            l1 = Dense(10, activation=act_funcs[1], use_bias=True)
            l1.build((None,8))
            ke1,be1=d1_w
            l1.set_weights([ke1.T,be1])
            model_new.add(l1)
            l2 = Dense(10000, activation=act_funcs[0], use_bias=True)
            l2.build((None,10))
            ke0,bi0=d0_w
            l2.set_weights([ke0.T,bi0[:10000]])
            model_new.add(l2)
            out_nn = model_new.predict(np.array([[1]]))
            loger.debug("in make_img")
            # loger.debug("out_nn",str(out_nn))  # Похоже 10_000 массивы трудно логирует
            # print("out_nn", str(out_nn))
            p_vec_tested = calc_out_nn(out_nn.tolist()[0])
            p_2d_img:np.ndarray = make_2d_arr(p_2d_img)
            new_img = Image.fromarray(np.uint8(p_2d_img))
            new_img.save("img_net.png")
        elif op == make_img_one_decomp:
            dw0, ke0= l_weis[0]
            model_new = Sequential()
            l0 = Dense(10000, activation=act_funcs[0], use_bias=True)
            l0.build((None, 1))
            model_new.add(l0)
            ke0_n=np.zeros(10000)+ke0[0]
            l0.set_weights([dw0.T,ke0_n])
            out_nn = model_new.predict(np.array([[1]]))
            loger.debug("in make_img")
            # loger.debug("out_nn",str(out_nn))  # Похоже 10_000 массивы трудно логирует
            print("out_nn", str(out_nn))
            loger.debug(f'vse odinak {np.all(out_nn == out_nn[0])}')
            p_vec_tested = calc_out_nn(out_nn.tolist()[0])
            p_2d_img: np.ndarray = make_2d_arr(p_vec_tested)
            new_img = Image.fromarray(np.uint8(p_2d_img))
            new_img.save("img_net.png")
        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+=1
        op=buffer[ip]

if __name__ == '__main__':
    loger, date=get_logger("debug","log.txt",__name__,'a')
    p1=(cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p2=(load_model_wei,get_weis,on_contrary,stop)
    p3=(cr_nn_,push_str,'b:/src',make_X_matr_img_,push_str,'X_matr_img',push_str,'Y_matr_img',determe_X_Y,
        fit_,predict,evalu_,sav_model_wei,stop)
    p4=(load_model_wei,get_weis,make_img_,stop)
    p5=(push_str,'b:/src1',push_i,1,push_i,10000,make_X_matr_img_,push_str,'X_matr_img',push_str,'Y_matr_img_one',determe_X_Y,cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p6=(load_model_wei,get_weis,make_img_one_decomp,stop)
    p7=(push_str,'X_comp',push_str,'Y_comp',determe_X_Y,cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p8=(load_model_wei,get_weis,on_contrary,stop)
    console('>>>', p8, loger, date)



