from Dataset.process.database import TunerRecord,Md5TestTime
import random
from Engine.satuner import SATuner
import math


class GroupTuner(SATuner):
    def __init__(self,search_space,evaluator,name):
        super().__init__(search_space, evaluator,name)
    def generate_candidate(self, state,md5_orig):
        new_state=dict(state)
        while True:         
            group=random.randint(0,self.search_space.group_num-1)
            select_group=self.search_space.space_reverse[group]
            for op in select_group:
                c=random.randint(0,1)
                if c==1:
                    new_state[op]=not new_state[op]
                    
            seq2=self.get_seq(new_state)
            new_md5=self.evaluator.compile(seq2)

            if md5_orig!=new_md5 and not Md5TestTime.exist(self.evaluator.session,new_md5):
                break

        return new_state

    def evaluate_candidate(self, candidate):
        return self.evaluator.evaluate(self.get_seq(candidate))


    
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
            if exec_time==-1 or exec_time==None or bin_md5==select_state[2]:
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
