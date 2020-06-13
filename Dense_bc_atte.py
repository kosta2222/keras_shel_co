import numpy as np
from util import print_obj
import sys
model_obj=None
(make_net,stop)=range(2)
def my_init(shape,dtype=None):
    return np.zeros(shape,dtype=dtype)+0.5674321
ke_init=("glorot_uniform",my_init)

class Dense:
    def __init__(self,units:int, input_dim:int=None,activation:str=None, use_bias:bool=None,kernel_initializer=None):
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias
        self.activation = activation
        self.input_dim = input_dim
        self.units = units
    def __str__(self):
        return print_obj('Dense',self.__dict__)

def vm(bc:list)->None:
    global model_obj
    ip=0
    op=0
    sp=-1
    steck=[0]*25
    op=bc[ip]
    while op!=stop:
        if op==make_net:  #  Ex:  make_net ('S', ('D','D','D'), (3, 2, 4), ('relu','sigmoid', 'softmax'), ('use_bias_1', 'use_bias_1', 'use_bias_1')))
            l_tmp=None
            use_bias_=False
            ip+=1
            arg=bc[ip]
            type_m, denses, inps, acts, use_bi, kern_init=arg
            if type_m=='S':
               print("op")
               model_obj=[]  #  Sequential
            for i in range(len(denses)):
                if denses[i]=='D':
                    splt_bi=use_bi[i].split('_')
                    print("splt_bi",splt_bi)
                    if splt_bi[-1]=='1':
                        use_bias_=True
                    elif splt_bi[-1]=='0':
                        use_bias_=False
                    if i==0:
                       l_tmp=Dense(inps[i+1],input_dim=inps[0],activation=acts[i], use_bias=use_bias_,kernel_initializer=kern_init)
                    else:
                        l_tmp=Dense(inps[i+1],activation=acts[i], use_bias=use_bias_,kernel_initializer=kern_init)
                model_obj.append(l_tmp)
            print("modl",model_obj)
            print([str(l) for l in model_obj])


        elif op==3:
            pass
        else:
            print("Unknown bytecode -> {0}"%op)
            sys.exit(1)
        ip+=1
        op=bc[ip]
if __name__ == '__main__':

  p1=(make_net,('S', ('D','D','D'), (3,2,3,4), ('relu','sigmoid', 'softmax'), ('use_bias_1', 'use_bias_1', 'use_bias_1'), ke_init[1]),stop)
  vm(p1)

