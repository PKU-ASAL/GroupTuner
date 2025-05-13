import os
import json


module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)
def reverse_op(op):
    if op.startswith('-fno-'):
        return f'-f{op[5:]}'
    return f'-fno-{op[2:]}'

class Space:
    def __init__(self):
        self.setting={}
        self.space={}
    def convert_to_str(self):
        seq=''
        for op in self.setting:
            if self.setting[op]:
                seq+=f' {op} '
            else:
                seq+=f' -fno-{op[2:]} '
        return seq

class Default_Space(Space):
    def __init__(self):
        with open(f'{MODULE_DIR}/all_opts.txt','r') as f:
            opts=f.read().split('\n')
        self.opts=opts
        with open(f'{MODULE_DIR}/option_O3_state.json','r') as f:
            O3_state=json.load(f)
        self.O3_state={}
        self.space={}
        self.setting={}
        # save the optional range
        for op in self.opts:
            self.space[op]=list(range(0, 2))
            if op in O3_state:
                self.O3_state[op]=O3_state[op]
            else:
                self.O3_state[op]=False

       
class Group_Space(Space):
    def __init__(self,file):
        with open(f'{MODULE_DIR}/{file}','r') as f:
            opts=f.read().split('---------------------------')      

        with open(f'{MODULE_DIR}/option_O3_state.json','r') as f:
            O3_state=json.load(f)
        
        self.space={}
        self.opts=[]
        self.O3_state={}
        self.space_reverse={}
        self.group_num=len(opts)
        
        for i in range(len(opts)):

            self.space_reverse[i]=[]
            opt=opts[i].split('\n')
            for o in opt:
                if o=='' :
                    continue
                self.space[o]=i
                self.opts.append(o)
                self.O3_state[o]=O3_state[o]
                self.space_reverse[i].append(o)
