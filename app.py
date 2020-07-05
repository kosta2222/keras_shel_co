from keras.optimizers import SGD
from util import get_logger
from keras.callbacks import LearningRateScheduler
from my_keras_customs import My_const_init
from consts import stop,nparray,make_net,determe_X_Y,push_obj,predict,push_i,push_str,\
plot_train,sav_model_wei,sav_model,fit_net,get_mult_class_matr,cr_callback_wi_loss_treshold_and_acc_shure,\
evalu_,load_model_wei,cr_sav_model_wei_best_callback,compile_net,get_weis,k_summary,push_fl,make_net_on_contrary
from executor import exec

"""
Замечания: для Python3 с f строками(используются в логинге)
"""

opt = SGD(lr=0.01)
# 1-optimizer 2-loss_function 3-metrics
compile_pars = (opt, 'mse', ['accuracy'])
monitor_pars=('val_accuracy')
def adap_lr(epoch):
    return 0.01*epoch
my_lr_scheduler=LearningRateScheduler(adap_lr)
# 1-ep 2-bach_size 3-validation_split 4-shuffle 5-callbacks
fit_pars=(100, 5, 1, False, [my_lr_scheduler])
my_init=My_const_init(9)
ke_init=("glorot_uniform",my_init)


if __name__ == '__main__':
    loger, date=get_logger("debug","log.txt",__name__,'w')

    # Научится определять круги 32x32 пикселя
    p17=(push_i,784,push_str,r'B:\msys64\home\msys_u\code\python\keras_shel_co\train_ann\train',get_mult_class_matr,
         make_net,('S', ('D','D'), (784,34,1),('t','s'), ('use_bias_1','use_bias_1','use_bias_1'),ke_init[0]),
         k_summary,
         compile_net,(opt,"mse",compile_pars[2]),
         cr_callback_wi_loss_treshold_and_acc_shure, (0.01, 1),
         fit_net,(100,5,fit_pars[2],False, fit_pars[4]),
         sav_model,
         sav_model_wei,
         stop)
    # Восстановить модель чтобы спросить про круг и треугольник
    p18=(push_i,784,push_str,r'B:\msys64\home\msys_u\code\python\keras_shel_co\train_ann\ask',get_mult_class_matr,
         load_model_wei,
         compile_net,(compile_pars[0],"mse",compile_pars[2]),
         predict,
         stop)
    # Модель для классификации тюлпанов и кругов 100x100 пикселей
    p19=(push_i, 10000, push_str, r'B:\msys64\home\msys_u\code\python\keras_shel_co\train_ann\train', get_mult_class_matr,
    cr_sav_model_wei_best_callback,('loss'),
    make_net, ('S', ('D', 'D'), (10000, 800, 2), ('s', 's', 'S'), ('use_bias_1', 'use_bias_1', 'use_bias_1'),ke_init[0]),
    k_summary,
    compile_net, (opt, compile_pars[1], compile_pars[2]),
    fit_net, (fit_pars[0], fit_pars[1], fit_pars[2], fit_pars[3], fit_pars[4]),
    evalu_,
    predict,
    sav_model,
    plot_train, "Tuples and Circs",
    stop)
    # Обучаем логическому И
    X_and=[[0,1],[1,0],[1,1],[0,0]]
    Y_and=[[0],[0],[1],[0]]
    p20=(push_obj,X_and,nparray,push_obj,Y_and,nparray,determe_X_Y,
         make_net,('S',('D','D'),(2, 3, 1),('r','s'),('use_bias_1', 'use_bias_1'),ke_init[0]),
         k_summary,
         compile_net, (opt, compile_pars[1], compile_pars[2]),
         cr_callback_wi_loss_treshold_and_acc_shure, (0.01, 1),
         fit_net, (fit_pars[0], fit_pars[1], fit_pars[2], fit_pars[3], fit_pars[4]),
         evalu_,
         predict,
         sav_model,
         plot_train, "Logic And",
         stop
         )
    exec(p20,loger,date)



