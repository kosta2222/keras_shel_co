(push_i, push_fl, push_str, send_list, send_obj,get_polin_hesh_) = range(6)

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
        elif op==send_obj:
            sp+=1
            ip+=1
            steck[sp]=buffer[ip]
        elif op==get_polin_hesh_:
            l_=steck[sp];sp-=1
            out=get_polin_hesh(l_)
            print("out",out)

        ip += 1
        if ip > (len(buffer) - 1):
            return
        try:
            op = buffer[ip]
        except IndexError:
            raise RuntimeError('Maybe arg of bytecode skipped')


if __name__ == '__main__':
    p1 = (send_obj,[255,0,255,255,0],get_polin_hesh_)
    vm(p1)

