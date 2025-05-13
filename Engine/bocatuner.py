from .common import Tuner
import random, time
import os
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import math
import numpy as np
from Dataset.process.database import TunerRecord
md = int(os.environ.get('MODEL', 1))
fnum = int(os.environ.get('FNUM', 8))
decay = float(os.environ.get('DECAY', 0.5))
scale = float(os.environ.get('SCALE', 10))
offset = float(os.environ.get('OFFSET', 20))

class get_exchange(object):
    def __init__(self, incumbent):
            self.incumbent = incumbent

    def to_next(self, feature_id,len_options):
        ans = [0] * len_options
        for f in feature_id:
            ans[f] = 1
        for f in self.incumbent:
            ans[f[0]] = f[1] 
        return ans

def get_ei(pred, eta):
    pred = np.array(pred).transpose(1, 0)
    m = np.mean(pred, axis=1)
    s = np.std(pred, axis=1)

    def calculate_f():
        z = (eta - m) / s
        return (eta - m) * norm.cdf(z) + s * norm.pdf(z)
        
    if np.any(s == 0.0):
        s_copy = np.copy(s)
        s[s_copy == 0.0] = 1.0
        f = calculate_f()
        f[s_copy == 0.0] = 0.0
    else:
        f = calculate_f()
    return f


class BOCATuner(Tuner):
    def __init__(self, search_space, evaluator,name):
        super().__init__(search_space, evaluator,name)
        self.space_size=len(search_space.space)

    def generate_opts(self,independent):
        opt_setting={}
        i=0
        for op in self.search_space.space:
            opt_setting[op] = self.search_space.space[op][independent[i]]
            i+=1
        self.search_space.setting=opt_setting
        return self.search_space.convert_to_str()

    def get_objective_score(self,independent,tuner_num):
        independent = self.generate_opts(independent)
        exec_time=self.evaluator.evaluate(independent)
        if exec_time[0]==None or exec_time[0]==-1 or exec_time[1]==None:
            return None,None
        
        return exec_time[0],exec_time[0] / self.default_perf

    def generate_conf(self,x):
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.space_size - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def do_search(self, model, eta, rnum):
        features = model.feature_importances_
        feature_sort = [[i, x] for i, x in enumerate(features)]
        feature_selected = sorted(feature_sort, key=lambda x: x[1], reverse=True)[:fnum]
        feature_ids = [x[0] for x in feature_sort]
        neighborhood_iterators = []    
        for i in range(2 ** fnum):
            comb = bin(i).replace('0b', '')
            comb = '0' * (fnum - len(comb)) + comb
            inc = []
            for k, s in enumerate(comb):
                if s == '1':
                    inc.append((feature_selected[k][0], 1))
                else:
                    inc.append((feature_selected[k][0], 0))
            neighborhood_iterators.append(get_exchange(inc))

        neighbors = []
        for i, inc in enumerate(neighborhood_iterators):
            for j in range(1 + int(rnum)):
                selected_feature_ids = random.sample(feature_ids, random.randint(0, len(feature_ids)))
                n = neighborhood_iterators[i].to_next(selected_feature_ids,self.space_size)
                neighbors.append(n)

        pred = []
        estimators = model.estimators_
        for e in estimators:
            pred.append(e.predict(np.array(neighbors)))
        acq_val_incumbent = get_ei(pred, eta)

    
        return [[i, a] for a, i in zip(acq_val_incumbent, neighbors)],feature_sort



    def get_nd_solutions(self,train_indep, training_dep, eta, rnum):
        model = RandomForestRegressor()
        model.fit(np.array(train_indep), np.array(training_dep))
        estimators = model.estimators_
        pred = []
        for e in estimators:
            pred.append(e.predict(train_indep))

        # do search
        merged_predicted_objectives,feature_selected = self.do_search(model, eta, rnum)
        merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
        for x in merged_predicted_objectives:
            if x[0] not in train_indep:

                return x[0], x[1],feature_selected

    def get_training_sequence(self,training_indep, training_dep, testing_indep, rnum):
        return_nd_independent, predicted_objectives,feature_selected = self.get_nd_solutions(training_indep, training_dep, testing_indep, rnum)
        return return_nd_independent, predicted_objectives,feature_selected
    
    def tune(self, budget, batch_size=1):
        training_indep = []
        round=1
        initial_sample_size = 2
        rnum0 = int(os.environ.get('RNUM', 2 ** 8))
        sigma = -scale ** 2 / (2 * math.log(decay))
        training_dep=[]
        best_result = 1e8
        while len(training_indep) < initial_sample_size:
            x = random.randint(0, 2 ** len(self.search_space.space))
            x = self.generate_conf(x)
            if x not in training_indep:
                exec_time,result=self.get_objective_score(x,round)
                if result==None or result==-1:
                    continue
                if best_result>exec_time:
                    best_result=exec_time
                TunerRecord.add(self.session,round,exec_time,best_result)
                
                seq=f"[{round}] current trial: {exec_time:.3f}s, best performance so far: {best_result:.3f}s"
                print(seq)
                round+=1
                training_indep.append(x)
                training_dep.append(result)

        while round < budget+1:
            
            rnum = rnum0 * math.exp(-max(0, len(training_indep) - offset) ** 2 / (2 * sigma ** 2))
            best_solution, return_nd_independent,feature_selected = self.get_training_sequence(training_indep, training_dep, best_result, rnum)
            
            exec_time,result= self.get_objective_score(best_solution,round)
            if result==None:
                continue
            
            training_indep.append(best_solution)
            training_dep.append(result)

            if best_result > exec_time:
                best_result = exec_time
            TunerRecord.add(self.session,round,exec_time,best_result)
            
            seq=f"[{round}] current trial: {exec_time:.3f}s, best performance so far: {best_result:.3f}s"
            print(seq)
            round += 1
        return training_dep
