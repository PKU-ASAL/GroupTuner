from .common import Tuner
import random, time
import os
import  copy,itertools
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
import numpy as np
from sqlalchemy import asc
from Dataset.process.database import TunerRecord
from .lib.getRelated import get_related_flags
module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)
md = int(os.environ.get('MODEL', 1))
fnum = int(os.environ.get('FNUM', 8))
decay = float(os.environ.get('DECAY', 0.5))
scale = float(os.environ.get('SCALE', 10))
offset = float(os.environ.get('OFFSET', 20))
FLOAT_MAX = float('inf')

def reverse_op(op):
    if op.startswith('-fno-'):
        return f'-f{op[5:]}'
    return f'-fno-{op[2:]}'

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


class CFSCATuner(Tuner):
    def __init__(self, search_space, evaluator,name):
        super().__init__(search_space, evaluator,name)
        self.dim=len(search_space.space)
        self.related=self.get_related_group()
        self.critical = []
        self.global_best_per = -1
        self.global_best_seq = []
        self.seed=int(time.time())
    def get_related_group(self):
        return get_related_flags(self.evaluator.cfsca_path)

    def get_objective_score(self,independent):
        independent = self.generate_opts(independent)
        
        exec_time=self.evaluator.evaluate(independent)
        exec_time=exec_time[0]
        if exec_time==None or exec_time==-1:
            return None,None
        
        return exec_time, self.default_perf/exec_time
    def generate_random_conf(self, x):
        """
        :param x: random generate number
        :return: the binary sequence for x
        """
        comb = bin(x).replace('0b', '')
        comb = '0' * (self.dim - len(comb)) + comb
        conf = []
        for k, s in enumerate(comb):
            if s == '1':
                conf.append(1)
            else:
                conf.append(0)
        return conf

    def get_ei(self, preds, eta):
        """
        :param preds: sequences' speedup for EI
        :param eta: global best speedup
        :return: the EI of a sequence
        """
        preds = np.array(preds).transpose(1, 0)
        m = np.mean(preds, axis=1)
        s = np.std(preds, axis=1)

        def calculate_f(eta, m, s):
            z = (eta - m) / s
            return (eta - m) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f(eta, m, s)
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f(eta, m, s)
        return f

    def get_ei_predict(self, model, now_best, wait_for_train):
        """
        :param model: RandomForest Model
        :param now_best: global best speedup
        :param wait_for_train: sequences Set
        :return: the sequences' EI
        """
        preds = []
        estimators = model.estimators_
        for e in estimators:
            preds.append(e.predict(np.array(wait_for_train)))
        acq_val_incumbent = self.get_ei(preds, now_best)
        return [[i, a] for a, i in zip(acq_val_incumbent, wait_for_train)]
    
    def runtime_predict(self, model, wait_for_train):
        """
        :param model: RandomForest Model
        :param wait_for_train: sequences set
        :return: the speedup of sequences set
        """
        estimators = model.estimators_
        sum_of_predictions = np.zeros(len(wait_for_train))
        for tree in estimators:
            predictions = tree.predict(wait_for_train)
            sum_of_predictions += predictions
        a = []
        average_prediction = sum_of_predictions / len(estimators)
        for i in range(len(wait_for_train)):
            x = [wait_for_train[i], average_prediction[i]]
            a.append(x)
        return a
    
    def getPrecision(self, model, seq):
        """
        :param model: RandomForest Model
        :param seq: sequence for prediction
        :return: the precision of a sequence and true speedup
        """
        rounds=self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1
        result,true_running = self.get_objective_score(seq)
        if true_running==None or true_running==-1:
            return None,None
        t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        s=f"[{rounds}] {t} current trial: {result:.3f}, best performance so far: {self.global_best_per:.3f}"
        print(s)
        TunerRecord.add(self.session,self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1,result,round(self.global_best_per,4))
        estimators = model.estimators_
        res = []
        for e in estimators:
            tmp = e.predict(np.array(seq).reshape(1, -1))
            res.append(tmp)
        acc_predict = np.mean(res)
        return abs(true_running - acc_predict) / true_running, true_running
    
    def selectByDistribution(self, merged_predicted_objectives):
        """
        :param merged_predicted_objectives: the sequences' EI and the sequences
        :return: the selected sequence
        """
        # sequences = [seq for seq, per in merged_predicted_objectives]
        diffs = [abs(perf - merged_predicted_objectives[0][1]) for seq, perf in merged_predicted_objectives]
        diffs_sum = sum(diffs)
        probabilities = [diff / diffs_sum for diff in diffs]
        index = list(range(len(diffs)))
        idx = np.random.choice(index, p=probabilities)
        return idx
    
    def build_RF_by_CompTuner(self):
        rounds=self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1
        """
        :return: model, inital_indep, inital_dep
        """
        inital_indep = []
        inital_dep=[]
        # randomly sample initial training instances
        while len(inital_indep) < 2:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance in inital_indep:
                continue
            exec_time,result=self.get_objective_score(initial_training_instance)
            if exec_time==None or exec_time==-1:
                continue
            if self.global_best_per<result:
                self.global_best_per=result
            t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
            seq=f"[{rounds}] {t} current trial: {result:.3f}s, best performance so far: {self.global_best_per:.3f}"
            print(seq)
            TunerRecord.add(self.session,self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1,exec_time,round(self.global_best_per,4))
            inital_dep.append(result)
            inital_indep.append(initial_training_instance)
            rounds+=1

        
        all_acc = []
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(np.array(inital_indep), np.array(inital_dep))
        rec_size = 2
        
        while rec_size < 11:
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(inital_indep), np.array(inital_dep))
            global_best = max(inital_dep)
            estimators = model.estimators_
            if all_acc:
                all_acc = sorted(all_acc)
            neighbors = []
            for i in range(30000):
                x = random.randint(0, 2 ** self.dim)
                x = self.generate_random_conf(x)
                if x not in neighbors:
                    neighbors.append(x)
            pred = []
            for e in estimators:
                pred.append(e.predict(np.array(neighbors)))
            acq_val_incumbent = self.get_ei(pred, global_best)
            ei_for_current = [[i, a] for a, i in zip(acq_val_incumbent, neighbors)]
            merged_predicted_objectives = sorted(ei_for_current, key=lambda x: x[1], reverse=True)
            acc = 0
            flag = False
            for x in merged_predicted_objectives:
                if flag:
                    break
                if x[0] not in inital_indep:
                    
                    acc, lable = self.getPrecision(model, x[0])
                    if acc==None:
                        continue
                    inital_dep.append(lable)
                    all_acc.append(acc)
                    inital_indep.append(x[0])
                    flag = True
            rec_size += 1
            if acc > 0.05:
                indx = self.selectByDistribution(merged_predicted_objectives)
                while merged_predicted_objectives[int(indx)][0] in inital_indep:
                    indx = self.selectByDistribution(merged_predicted_objectives)
                
                acc, label = self.getPrecision(model, merged_predicted_objectives[int(indx)][0])
                if acc==None:
                    continue
                inital_dep.append(label)
                all_acc.append(acc)
                inital_indep.append(merged_predicted_objectives[int(indx)][0])
                rec_size += 1
        self.global_best_per = max(inital_dep)
        self.global_best_seq = inital_indep[inital_dep.index(max(inital_dep))]
        return model, inital_indep, inital_dep
    
    def get_critical_flags(self, model, inital_indep, inital_dep):
        """
        :param: model: RandomForest Model
        :param: inital_indep: selected sequences
        :param: inital_dep: selected sequences' performance
        :return: critical_flags_idx, new_model
        """
        rounds=self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1
        candidate_seq = []
        candidate_per = []
        inital_indep_temp = copy.deepcopy(inital_indep)
        inital_dep_temp = copy.deepcopy(inital_dep)
        while len(candidate_seq) < 30000:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in candidate_seq:
                candidate_seq.append(initial_training_instance)
        all_per = self.runtime_predict(model,candidate_seq)
        candidate_per = [all[1] for all in all_per]
        pos_seq = [0] * len(self.related)  
        while True:  
            now_best = max(candidate_per)
            now_best_seq = candidate_seq[candidate_per.index(now_best)]
            result,now_best = self.get_objective_score(now_best_seq)
            if result==None and now_best in candidate_per and now_best_seq in candidate_seq:
                candidate_per.remove(now_best)
                candidate_seq.remove(now_best_seq)
            else:
                break
        t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
        seqs=f"[{rounds}] {t} current trial: {now_best:.3f}s, best performance so far: {self.global_best_per:.3f}"
        print(seqs)
        TunerRecord.add(self.session,self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1,result,self.global_best_per)
        rounds+=1
        inital_indep_temp.append(now_best_seq)
        inital_dep_temp.append(now_best)
        model_new = RandomForestRegressor(random_state=self.seed)
        model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))
        if self.global_best_per < now_best:
            self.global_best_per = now_best
            self.global_best_seq = now_best_seq
        for idx in range(len(self.related)):
            new_candidate = []
            for j in range(len(candidate_seq)):
                seq = copy.deepcopy(candidate_seq[j])
                seq[self.related[idx]] = 1 - seq[self.related[idx]]
                new_candidate.append(seq)
            new_per = [all[1] for all in self.runtime_predict(model_new,new_candidate)]
            new_seq = [all[0] for all in self.runtime_predict(model_new,new_candidate)]
            new_best_seq = new_seq[new_per.index(max(new_per))]
            result,new_best = self.get_objective_score(new_best_seq)
            if new_best==None:
                continue
            if new_best > self.global_best_per:
                self.global_best_per = new_best
                self.global_best_seq = new_best_seq
            t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
            seq=f"[{rounds}] {t} current trial: {new_best:.3f}s, best performance so far: {self.global_best_per:.3f}"
            print(seq)
            TunerRecord.add(self.session,self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1,result,self.global_best_per)
            rounds+=1            

            for l in range(len(new_candidate)):
                if (candidate_per[l] > new_per[l] and new_candidate[l][self.related[idx]] == 1) or (candidate_per[l] < new_per[l] and new_candidate[l][self.related[idx]] == 0):
                    pos_seq[idx] -= 1
                else:
                    pos_seq[idx] += 1
            inital_indep_temp.append(new_best_seq)
            inital_dep_temp.append(new_best)
            model_new = RandomForestRegressor(random_state=self.seed)
            model_new.fit(np.array(inital_indep_temp), np.array(inital_dep_temp))

        sort_pos = sorted(enumerate(pos_seq), key=lambda x: x[1], reverse=True)
        critical_flag_idx = []
        for i in range(10):
            critical_flag_idx.append(self.related[sort_pos[i][0]])
        flags=[]
        for idx in critical_flag_idx:
            flags.append(self.search_space.opts[idx])
        return critical_flag_idx, model_new
    
    def searchBycritical(self, critical_flag):
        """
        :param: critical_flag: idx of critical flag
        :return: the bias generation sequences
        """
        permutations = list(itertools.product([0, 1], repeat=10))
        seqs = []
        while len(seqs) < 1024 * 40:
            x = random.randint(0, 2 ** self.dim)
            initial_training_instance = self.generate_random_conf(x)
            if initial_training_instance not in seqs:
                seqs.append(initial_training_instance)
        for i in range(len(permutations)):
            for idx in range(len(critical_flag)):
                for offset in range(0, 1024 * 40, 1024):
                    seqs[i + offset][critical_flag[idx]] = permutations[i][idx]
        return seqs

    def generate_opts(self,independent):
        opt_setting={}
        i=0
        for op in self.search_space.space.keys():
            opt_setting[op] = independent[i]
            i+=1
        self.search_space.setting=opt_setting
        return self.search_space.convert_to_str()
    
    def tune(self, budget, batch_size=1):
        """
        build model and get data set
        """
        i=1
        rounds=self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1
        model, inital_indep, inital_dep = self.build_RF_by_CompTuner()
        critical_flag, model_new = self.get_critical_flags(model, inital_indep, inital_dep)
        while self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round < budget:
            seq = self.searchBycritical(critical_flag)
            result = self.runtime_predict(model_new, seq)
            sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
            exec_time,true_reslut = self.get_objective_score(sorted_result[0][0])
            if not true_reslut:
                continue
            if true_reslut > self.global_best_per:
                self.global_best_per = true_reslut
                self.global_best_seq = sorted_result[0][0]
            t=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
            seqs=f"[{rounds}] {t} current trial: {true_reslut:.3f}s, best performance so far: {self.global_best_per:.3f}"
            TunerRecord.add(self.session,self.evaluator.session.query(TunerRecord).order_by(asc(TunerRecord.round)).all()[-1].round+1,exec_time,self.global_best_per)
            rounds+=1
            i+=1
            print(seqs)
