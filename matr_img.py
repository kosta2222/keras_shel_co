(push_i, push_fl, push_str, send_list, send_obj,matr_img_) = range(6)
from os import listdir
import os
import numpy as np
from PIL import Image
import PIL
def matr_img(path_:str,pixel_amount:int)->tuple:
    p=listdir(path_)
    print("p",p)
    X=[]
    fold_cont:list=None
    num_clss=len(p)
    Y=[]
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
            print("X_t[f_index]",X_t[f_index])
            f_index+=1
        X.extend(X_t)
        Y.extend(Y_t)
        print("X",X)

    return (X,Y)

def vm(buffer, logger=None, date=None):
    len_ = 25
    if logger:
        logger.info(logger.debug(f'Log started {date}'))
    vm_is_running = True
    ip = 0
    sp = -1
    sp_str = -1
    steck = [0] * len_
    op = buffer[ip]
    vm_is_running=True
    while vm_is_running:
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
            steck[sp] = buffer[ip]
        elif op==send_list:
            sp+=1
            ip+=1
            steck[sp]=buffer[ip]
        elif op==matr_img_:
            pix_am=steck[sp]
            sp-=1
            path=steck[sp]
            sp-=1
            out=matr_img(path,pix_am)
            assert isinstance(out,tuple)
            assert len(out[0])==9
            print("len out[0]",len(out[0]))
            assert len(out[1])==9
        ip+=1
        if ip>(len(buffer)-1):
            return
        op=buffer[ip]

if __name__ == '__main__':
    p1=(push_str,'B:\\msys64\\home\\msys_u\\img\\tmp',push_i,10000,matr_img_)
    vm(p1)