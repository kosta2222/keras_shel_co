from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from  keras.models import Sequential, model_from_json
from  keras.layers import Dense
from keras.utils import plot_model
import numpy as np
from numpy.fft import rfft, irfft, ifft,fft
from util import get_logger, make_train_img_matr, calc_out_nn, make_2d_arr,l_test_after_contr, calc_out_nn_n
from keras.callbacks import LearningRateScheduler
from PIL import Image
import json

# from  keras.optimizers import
"""
Замечания: для Python3 с f строками(используются в логинге)
"""
act_funcs=('softmax','relu','softmax')
# init_lays=(10000, 10, 8, 2)
# init_lays=(10, 3, 3, 2)
init_lays=(5,2)
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
    # d1=Dense(init_lays[2], activation=act_funcs[1], use_bias=us_bias[1],kernel_initializer=k_i)
    # model.add(d1)
    # d2=Dense(init_lays[3], activation=act_funcs[2], use_bias=us_bias[1],kernel_initializer=k_i)
    # model.add(d2)
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
sav_model_wei=9
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
Y_matr_img=np.array([[0,1],[0,1]])
Y_matr_img_one=np.array([[0,1]])
or_X = [[1, 1], [1, 0], [0, 1], [0, 0]]
or_Y = [[1], [1], [1], [0]]
X_comp=np.array([[0,1,0,1,1]])
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
            X_matr_img.astype('float32')
            X_matr_img/= 255.0
            print("X matr img",X_matr_img.tolist())
            logger.debug(f'in make X matr img ravni li: {np.all(X_matr_img==X_matr_img[0])}')
            # X_matr_img-=np.mean(X_matr_img,axis=0,dtype='float64')
            # X_matr_img /= 255
            # X_matr_img=np.std(X_matr_img_n,axis=0,dtype='float64')
            # X_matr_img /= 255
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
            wei_t=None
            with open("model_json.json", 'w') as f:
                f.write(model_obj.to_json())
            print("Model saved")
            logger.info('Model saved')
            model_obj.save_weights("wei.h5")
            for i in range(len(model_obj.layers)):
                l_weis.append(model_obj.layers[i].get_weights())
                wei_t=model_obj.layers[i].get_weights()
                ke_t,bi_t=wei_t
                tenz=[ke_t.tolist(),bi_t.tolist()]
                wei_dic={i:tenz}
                with open('weis_json.json', 'w') as f:
                   json.dump(wei_dic, f)
                   print("Json weights written")
                   logger.info("Json weights written")
            print("Weights saved")
            logger.info('Weights saved')
        elif op==load_model_wei:
               # loaded_json_model=''
               # with open('model_json.json','r') as f:
               #    loaded_json_model=f.read()
               # model_obj=model_from_json(loaded_json_model)
               model_obj=create_nn(False)
               model_obj.load_weights('wei.h5')
               print("Loaded model and weights")
               logger.info("Loaded model and weights")
        elif op==get_weis:
            for i in range(len(model_obj.layers)):
               l_weis.append(model_obj.layers[i].get_weights())
            # d0_w,d1_w,d2_w=l
            # loger.debug(f'd0 {d0_w}\n')
            # loger.debug(f'd1 {d1_w}\n')
            # loger.debug(f'd2 {d2_w}\n')
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
        elif op==on_contrary:
            # d0_w,d1_w,d2_w=l_weis
            d2_w=l_weis[0]
            model_new = Sequential()
            l0 = Dense(5, activation=act_funcs[2], use_bias=True)
            l0.build((None,2))
            ke2, bi2=d2_w
            bi2_n=np.zeros(3)+bi2[0]
            l0.set_weights([ke2.T,bi2_n])
            model_new.add(l0)
            # l1 = Dense(3, activation=act_funcs[1], use_bias=True)
            # l1.build((None,3))
            # ke1,bi1=d1_w
            # l1.set_weights([ke1.T,bi1])
            # model_new.add(l1)
            # l2 = Dense(10, activation=act_funcs[0], use_bias=True)
            # l2.build((None,3))
            # ke0,be0=d0_w
            # be0_n=np.zeros(10)+be0[0]
            # l2.set_weights([ke0.T,be0_n])
            # model_new.add(l2)
            out_nn=model_new.predict(np.array([[0,1]]))
            loger.info(f'predict cont {out_nn}')
            tes_out_nn=l_test_after_contr(out_nn.tolist()[0],10)
            logger.debug(f'vec izn {tes_out_nn}')
            logger.debug(f'ravini li: {X_comp==tes_out_nn}')
        elif op == make_img_:
            # d0_w,d1_w,d2_w=l_weis
            d0_w=l_weis[0]
            model_new = Sequential()
            l0 = Dense(10000, activation=act_funcs[0], use_bias=True)
            l0.build((None,2))
            ke2,be2=d0_w
            bi2_n=np.zeros(10000)+be2[0]
            l0.set_weights([ke2.T,bi2_n])
            # l0.set_weights([d2])
            # print("ke2 ri",ke2)
            model_new.add(l0)
            # l1 = Dense(10, activation=act_funcs[1], use_bias=True)
            # l1.build((None,8))
            # ke1,be1=d1_w
            # be1_n=np.zeros(10)+be1[0]
            # l1.set_weights([ke1.T,be1_n])
            # model_new.add(l1)
            # l2 = Dense(10000, activation=act_funcs[0], use_bias=True)
            # l2.build((None,10))
            # ke0,bi0=d0_w
            # bi0_n=np.zeros(10000)+bi0[0]
            # l2.set_weights([ke0.T,bi0_n])
            # model_new.add(l2)
            out_nn = model_new.predict(np.array([[0,1]]))
            # loger.debug("out_nn",str(out_nn))  # Похоже 10_000 массивы трудно логирует
            # print("out_nn", out_nn.tolist())
            l_test_after_contr_=l_test_after_contr(out_nn.tolist()[0],10000)
            # print("l test af contr",l_test_after_contr_)
            # print(all([0.00010001]*10000==out_nn.tolist()))
            vec_tested = calc_out_nn(l_test_after_contr_)
            # print("vec tested",vec_tested)
            # print(X_matr_img[0].tolist()==vec_tested)
            # _2d_img: np.ndarray = make_2d_arr(vec_tested)
            vec_tested_np=np.array(vec_tested)
            img_prep=vec_tested_np.reshape(100,100)
            img_prep=img_prep.astype('uint8')
            new_img = Image.fromarray(img_prep,'L')
            new_img.save("img_net_creative.png")
            print("Img written")
            logger.info("Img written")
            loger.debug("in make_img")
        elif op == make_img_one_decomp:
            dw0, ke0= l_weis[0]
            model_new = Sequential()
            l0 = Dense(10000, activation=act_funcs[0], use_bias=True)
            l0.build((None, 2))
            model_new.add(l0)
            ke0_n=np.zeros(10000)+ke0[0]
            l0.set_weights([dw0.T,ke0_n])
            out_nn = model_new.predict(np.array([[0,1]]))
            loger.debug("in make_img")
            # loger.debug("out_nn",str(out_nn))  # Похоже 10_000 массивы трудно логирует
            print("out_nn", out_nn.tolist()[0])
            l_test_after_contr_=l_test_after_contr(out_nn.tolist()[0],10000)
            # print("l test af contr",l_test_after_contr_)
            # print(all([0.00010001]*10000==out_nn.tolist()))
            vec_tested = calc_out_nn(l_test_after_contr_)
            print("vec tested",vec_tested)
            # print(X_matr_img[0].tolist()==vec_tested)
            # _2d_img: np.ndarray = make_2d_arr(vec_tested)
            vec_tested_np=np.array(vec_tested)
            img_prep=vec_tested_np.reshape(100,100)
            img_prep=img_prep.astype('uint8')
            new_img = Image.fromarray(img_prep,'L')
            new_img.save("img_net.png")
            print("Img written")
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
                use_bias_ = False
                ip += 1
                arg = buffer[ip]
                type_m, denses, inps, acts, use_bi, kern_init = arg
                if type_m == 'S':
                    print("op")
                    model_obj = Sequential()
                for i in range(len(denses)):
                    if denses[i] == 'D':
                        splt_bi = use_bi[i].split('_')
                        print("splt_bi", splt_bi)
                        if splt_bi[-1] == '1':
                            use_bias_ = True
                        elif splt_bi[-1] == '0':
                            use_bias_ = False
                        if i == 0:
                            l_tmp = Dense(inps[i + 1], input_dim=inps[0], activation=acts[i], use_bias=use_bias_,
                                          kernel_initializer=kern_init)
                            l_tmp.build((None,inps[0]))
                        else:
                            l_tmp = Dense(inps[i + 1], activation=acts[i], use_bias=use_bias_,
                                          kernel_initializer=kern_init)
                            l_tmp.build((None,inps[i]))
                    model_obj.add(l_tmp)
        elif op==k_plot_model:
            plot_model(model_obj, to_file='model.png', show_shapes=True)
        elif op==k_summary:
            model_obj.summary()
        elif op==compile_net:
            ip+=1
            arg=buffer[ip]
            opt, loss_obj, metrics=arg
            model_obj.compile(optimizer=opt, loss=loss_obj, metrics=metrics)


        else:
            print("Unknown bytecode -> %d"%op)
            return
        ip+=1
        op=buffer[ip]

if __name__ == '__main__':
    loger, date=get_logger("debug","log.txt",__name__,'a')
    p1=(cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p2=(load_model_wei,get_weis,on_contrary,stop)
    p3=(cr_nn_,push_str,'B:\\msys64\\home\\msys_u\\img\\prod_nn',push_i,2,push_i,10000,make_X_matr_img_,push_str,'X_matr_img',push_str,'Y_matr_img',determe_X_Y,
        fit_,predict,evalu_,sav_model_wei,stop)
    p4=(load_model_wei,get_weis,make_img_,stop)
    p5=(push_str,'b:/src1',push_i,1,push_i,10000,make_X_matr_img_,push_str,'X_matr_img',push_str,'Y_matr_img_one',determe_X_Y,cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p6=(load_model_wei,get_weis,make_img_one_decomp,stop)
    p7=(push_str,'X_comp',push_str,'Y_comp',determe_X_Y,cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p8=(load_model_wei,get_weis,on_contrary,stop)
    p9=(push_str,'B:\\msys64\\home\\msys_u\\img\\prod_nn',push_i,1,push_i,10000,make_X_matr_img_,push_str,'X_matr_img',push_str,'Y_matr_img_one',determe_X_Y,cr_nn_,fit_,predict,evalu_,sav_model_wei,stop)
    p10=(push_str,'B:\\msys64\\home\\msys_u\\img\\prod_nn',push_i,1,push_i,10000,make_X_matr_img_,load_model_wei,get_weis,make_img_one_decomp,stop)
    p11=(load_json_wei_pr_fft,stop)
    p12=(make_net,('S', ('D','D','D'), (3,2,3,4), ('relu','sigmoid', 'softmax'), ('use_bias_1', 'use_bias_1', 'use_bias_1'), ke_init[1]),k_summary,stop)
    console('>>>', p12, loger, date)



