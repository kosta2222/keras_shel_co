#----------------------дебаг, хелп функции--------------------------------------
import logging
import numpy as np
import os
from PIL import Image
import datetime as d
from functools import wraps
from PIL import Image
import PIL
from os import listdir

def my_hesh_img(list_):
    s=''
    for i in range(len(list_)):
        if list_[i]==0:
            ch='ab'
        else:
            ch=chr(int(list_[i]))
        s+=ch
    return s
#--------------Static check---------------------
class TypeErr(Exception):
    def __init__(self,msg):
        self.txt=msg
def check_types(anots:dict,locals_or_globals:dict):
    import sys
    for k,v in anots.items():
        # print("k=",k,"v=",v)
        try:
            val_l=locals_or_globals.get(k)
            if k=='return':
                val_l=locals_or_globals.get('ret_v')
            assert type(val_l)==v
        except AssertionError:
            err=k + " must be type " + str(v)
            # print(k + " must be type " + str(v))
            raise TypeErr(err)

[stop,push_dict,push_obj,check_args,check_globs,check_locs]=range(6)
def vm_ty_check(bc:list)->None:
    ip=0
    sp=-1
    steck=[0]*25
    # steck=[]
    op=0
    op=bc[ip]
    while op!=stop:
        if op==push_dict:
            sp+=1
            ip+=1
            steck[sp]=bc[ip]
        elif op==push_obj:
            sp+=1
            ip+=1
            steck[sp]=bc[ip]
            # steck.append(bc[ip])
        elif op==check_args:
            locs=steck[sp]
            sp-=1
            annots=steck[sp].__annotations__
            print("annots",annots)
            sp-=1
            check_types(annots,locs)
        elif op==check_globs:
            annots=steck[sp]
            sp-=1
            check_types(annots,globals())
        elif op == check_locs:
            dict_ = steck[sp]
            sp -= 1
            locs = steck[sp]
            print("dict_",dict_)
            sp -= 1
            check_types(dict_, locs)
        ip+=1
        op=bc[ip]
        # print("steck",steck)
"""
Example:
def my_func(a:int, b:float,c_:str)->str:
    c:int=a+b
    s=c_ + str(c)
    ret_v=s
    vm_ty_check([push_obj,my_func,push_dict,locals(),check_args,push_dict,locals(),push_dict,{'c':int},check_locs,stop])
    return ret_v
# Global context
x:float=7
vm_ty_check([push_dict,{'x':int},check_globs,stop])               
"""
#--------------/Static check---------------------


def get_logger(level_,fname,module,mode='w'):
    today=d.datetime.today()
    today_s=today.strftime('%x %X')
    logger = None
    logger = logging.getLogger(module)
    if level_ == 'debug':
            logging.basicConfig(level=logging.DEBUG, filename=fname, filemode=mode)

    elif level_ == 'release':
        logging.basicConfig(level=logging.INFO, filename=fname, filemode=mode)
    return logger, today_s

def calc_list(list_:list):
    cn_elem = 0
    for i in range(len(list_)):
        elem=list_[i]
        cn_elem+=1
        if elem==0:
            break
        else:
            continue
    return cn_elem

def l_test_after_contr(l_:list,n):
    l_new=[0]*n
    max_=max(l_)
    min_=min(l_)
    for i in range(len(l_)):
        if l_[i]==max_:
            l_new[i]=1
        elif l_[i]==min_:
            l_new[i]=0
    return l_new




def calc_out_nn(l_: list):
    l_tested = [0] * 10000
    for i in range(len(l_)):
        # val = round(l_[i], 1)
        if l_[i] ==1:
            l_tested[i] = 255
        else:
            l_tested[i] = 0
    return l_tested

def calc_out_nn_n(l_: list):
    l_tested = [0] * 10000
    for i in range(len(l_)):
        # val = round(l_[i], 1)
        # if l_[i] ==1:
            l_tested[i] = l_[i] * 255
        # else:
        #     l_tested[i] = 0
    return l_tested

def make_2d_arr(_1d_arr: list):
    matr_make = np.zeros(shape=(100, 100))
    for i in range(100):
        for j in range(100):
            matr_make[i][j] = _1d_arr[i * 100 + j]
    return matr_make


def make_train_img_matr(p_: str,rows,elems) -> np.ndarray:
    matr = np.zeros(shape=(rows, elems))
    data = None
    img = None
    cn_img = 0
    for i in os.listdir(p_):
        ful_p = os.path.join(p_, i)
        img = Image.open(ful_p)
        print("img", ful_p)
        data = list(img.getdata())
        matr[cn_img] = data
        cn_img+=1
    return matr.tolist()


def matr_img(path_:str,pixel_amount:int)->tuple:
    p=listdir(path_)
    # print("p",p)
    X=[]
    Y=[]
    fold_cont:list=None
    num_clss=len(p)
    img:PIL.Image=None
    data=None
    for fold_name_i in p:
        # print("fold_name_i",fold_name_i)
        fold_name_ind=int(fold_name_i.split('_')[0])
        p_tmp_ful=os.path.join(path_,fold_name_i)
        fold_content=listdir(p_tmp_ful)
        # print("fold_cont",fold_content)
        rows=len(fold_content)
        X_t=np.zeros((rows,pixel_amount))
        Y_t=np.zeros((rows,num_clss))
        f_index=0
        for file_name_j in fold_content:
            Y_t[f_index][fold_name_ind]=1
            img=Image.open(os.path.join(p_tmp_ful,file_name_j))
            data=list(img.getdata())
            X_t[f_index]=data
            # print("X_t[f_index]",X_t[f_index])
            print("file name",file_name_j)
            # print("X_t[f_index]",X_t[f_index])
            # print("my he img", my_hesh_img(X_t[f_index]))
            f_index+=1
        X_t=X_t.tolist()
        Y_t=Y_t.tolist()
        X.extend(X_t)
        Y.extend(Y_t)
        # print("X",X)
    return (X,Y)
def _0_(str_):
    print("Success ->", end = " ")
    print("function",str_)
    return "Success ->function {}".format(str_)

def print_obj(name_obj_s,dict_obj:dict,si=50)->str:
    si=si
    res=''
    for k,v in dict_obj.items():
        if (not isinstance(v,int)) and (not isinstance(v, float)) and (not isinstance(v, bool)) and v (not None) :
            assert('v_maybe_matrix','v_maybe_matrix')
            if len(v)>si or (isinstance(v[0], list) and len(v[0])>si):
               res+=k+' = '+ '<size of {0} [or list[0] ] is greater {1}>\n'.format(type(v), si)
               continue
        res+=k+' = '+str(v)
        res+='\n'
    return name_obj_s+':\n'+res

def tmp_wrap(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        print(func.__name__)
        return func(*args, **kwargs)
    return tmp

