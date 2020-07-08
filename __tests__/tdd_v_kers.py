from os import listdir
import PIL
from PIL import Image
import os
import numpy as np


(push_i, push_fl, push_str, send_list, push_obj,get_polin_hesh_,matr_img_3_chanels) = range(7)


def matr_img_3_chanels(path_:str,pixel_amount:int)->tuple:
    p=listdir(path_)
    X=[]
    Y=[]
    fold_cont:list=None
    num_clss=len(p)
    img:PIL.Image=None
    data=None
    for fold_name_i in p:
        fold_name_ind=int(fold_name_i.split('_')[0])
        p_tmp_ful=os.path.join(path_,fold_name_i)
        fold_content=listdir(p_tmp_ful)
        rows=len(fold_content)
        X_t=np.zeros((3 * rows,pixel_amount))
        Y_t=np.zeros((3 * rows,num_clss))
        f_index=0
        for file_name_j in fold_content:
            Y_t[f_index][fold_name_ind]=1
            img=Image.open(os.path.join(p_tmp_ful,file_name_j))
            data=list(img.getdata())
            # X_t[f_index]=data
            for rgb_ind in range(3):
                for tupl_ind in range(pixel_amount):
                    X_t[f_index][tupl_ind]=data[tupl_ind][rgb_ind]
                    tupl_ind+=1
                f_index+=1
            print("file name",file_name_j)
            # f_index+=1
        X_t=X_t.tolist()
        Y_t=Y_t.tolist()
        X.extend(X_t)
        Y.extend(Y_t)
        # print("X",X)
    return (X,Y)


def get_polin_hesh(list_):
    s=''
    for i in range(len(list_)):
        if list_[i]==0:
            ch='ab'
        else:
          ch=chr(list_[i])
        s+=ch
    return s
def vm(buffer, logger=None, date=None):
    len_ = 25
    if logger:
        logger.info(logger.debug(f'Log started {date}'))
    vm_is_running = True
    ip = 0
    sp = -1
    steck = [0] * len_
    op = buffer[ip]
    while ip < len(buffer):
        if op == push_i:
            sp += 1
            ip += 1
            steck[sp] = int(buffer[ip])  # Из строкового параметра
        elif op == push_fl:
            sp += 1
            ip += 1
            steck[sp] = float(buffer[ip])  # Из строкового параметра
        elif op == push_str:
            sp += 1
            ip += 1
            steck[sp] = buffer[ip]
        elif op==push_obj:
            sp+=1
            ip+=1
            steck[sp]=buffer[ip]
        elif op==get_polin_hesh_:
            l_=steck[sp]
            sp-=1
            out=get_polin_hesh(l_)
            print("out",out)
        elif op==matr_img_3_chanels:
            pix_am=steck[sp]
            sp-=1
            pat=steck[sp]
            sp-=1
            X,Y=matr_img_3_chanels(pat,pix_am)
            print("X",X)
            print("len X",len(X))
            print("len X[0]",len(X[0]))
            X=list(zip(X[0],X[1],X[2]))
            print("X zip",X)
            X=np.array(X)
            X.astype('uint8')
            X=X.reshape((28,28,3))
            img_prep=Image.fromarray(X,'RGB')
            img_prep.save("my_chan.png")
            # assert isinstance(X,list)

        ip += 1
        if ip > (len(buffer) - 1):
            return
        try:
            op = buffer[ip]
        except IndexError:
            raise RuntimeError('Maybe arg of bytecode skipped')


if __name__ == '__main__':
    p1 = (push_obj,[255,0,255,255,0],get_polin_hesh_)
    p2=(push_str,r'B:\msys64\home\msys_u\img\tmp\test_3ch',push_i,784,matr_img_3_chanels)
    vm(p2)

