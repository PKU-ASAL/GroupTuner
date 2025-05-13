from Engine.common import Tuner
import random
import math
import os
from Dataset.process.database import TunerRecord,Md5TestTime
FLOAT_MAX = float('inf')


module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)


def reverse_op(op):
    if op.startswith('-fno-'):
        return f'-f{op[5:]}'
    return f'-fno-{op[2:]}'


class SATuner(Tuner):
    def __init__(self, search_space, evaluator,name):
        super().__init__(search_space, evaluator,name)
        self.O3_01_result=self.evaluator.evaluate(self.get_seq(self.search_space.O3_state))
        print(f'[0] default time: {self.O3_01_result[0]:.3f}s')
        self.current_state=[[self.search_space.O3_state,self.O3_01_result[0],self.O3_01_result[1]]]
        self.worst_state,self.worst_perf,self.worst_md5=self.current_state[0][0],self.current_state[0][1],self.current_state[0][2]
 
    def get_seq(self,state):
        seq=''
        for key in state:
            if state[key]:
                seq+=f' {key} '
            else:
                seq+=f' {reverse_op(key)} '
        return seq[:-1]    
    
    def generate_candidate(self, state,md5_orig):
        while True:
            new_state=dict(state)       
            for op in new_state:
                c=random.randint(0,1)
                if c==1:
                    new_state[op]=not new_state[op]

            seq2=self.get_seq(new_state)
            new_md5=self.evaluator.compile(seq2)

            if md5_orig!=new_md5 and not Md5TestTime.exist(self.evaluator.session,new_md5):
                return new_state

    def evaluate_candidate(self, candidate):
        return self.evaluator.evaluate(self.get_seq(candidate))
    
    def update_worst_state_perf(self,candidate,exec_time,bin_md5):
        self.current_state.remove([self.worst_state,self.worst_perf,self.worst_md5])
        self.current_state.append([candidate,exec_time,bin_md5])
        self.worst_state, self.worst_perf,self.worst_md5=self.current_state[0][0],self.current_state[0][1],self.current_state[0][2]
        for i in range(1,len(self.current_state)):
            if self.current_state[i][1]>self.worst_perf:
                self.worst_perf=self.current_state[i][1]
                self.worst_state=self.current_state[i][0]
                self.worst_md5=self.current_state[i][2]

    def init_state(self,budget):
        i=1
        best_perf=FLOAT_MAX
        while len(self.current_state)<10 and len(self.current_state)<=budget:
            new_state=self.generate_candidate(self.search_space.O3_state,self.O3_01_result[1])
            new_perf,new_md5=self.evaluate_candidate(new_state)
            if new_perf==None or new_perf==-1:
                continue
            if new_perf>self.worst_perf:
                self.worst_state=new_state
                self.worst_perf=new_perf
                self.worst_md5=new_md5
            if new_perf<best_perf:
                best_perf=new_perf
            self.current_state.append([new_state,new_perf,new_md5])
            TunerRecord.add(self.session,i,new_perf,best_perf)
            
            print(f'[{i}] initial state: {new_perf:.3f}s, best performance so far: {best_perf:.3f}s')
            i+=1
        return best_perf
        
    def tune(self, budget):
        initial_temp=1000
        min_temp=1
        current_temp=initial_temp
        alpha=5e-5

        best_perf=self.init_state(budget)

        if budget>10:
            cooling_rate=(min_temp/initial_temp)**(1/(budget-10))
        else:
            return best_perf
        i=10
        
        while i<=budget:
            select_state=self.current_state[random.randint(0,9)]
            candidate = self.generate_candidate(select_state[0],select_state[2])
            exec_time,bin_md5 = self.evaluate_candidate(candidate)
            if exec_time==-1 or exec_time==None:
                continue
            
            if exec_time<self.worst_perf:
                self.update_worst_state_perf(candidate,exec_time,bin_md5)    
            else:
                derta=(exec_time-self.worst_perf)/self.worst_perf
                prob=random.random() -math.exp(-(derta/ (current_temp*alpha)))
                
                if prob<0:
                    self.update_worst_state_perf(candidate,exec_time,bin_md5)

            if exec_time<best_perf:
                best_perf = exec_time
                best_opt_setting = self.get_seq(candidate)
           
            seq=f"[{i}] current trial: {exec_time:.3f}s, best performance so far: {best_perf:.3f}s"
            
            TunerRecord.add(self.session,i,exec_time,best_perf)
            i+=1
            print(seq)
            current_temp*=cooling_rate

        return best_perf


