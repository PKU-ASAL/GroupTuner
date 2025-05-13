import os
from Dataset.process.database import OptionsMd5, Md5TestTime,TunerRecord
FLOAT_MAX = float('inf')
module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)

class Logger():
    def __init__(self,output_file):
        self.output=output_file

    def info(self,log):
        print(log)
        with open(self.output,'a+') as f:
            f.write(log+'\n')
        

class Tuner:
    def __init__(self, search_space, evaluator,name = "Base Tuner"):
        self.search_space = search_space
        self.evaluator = evaluator
        self.name = name
        self.session = self.evaluator.session
        self.default_perf=self.evaluator.default_perf
        print(f'[-1] O3 time:{self.default_perf:.3f}s')
        TunerRecord.add(self.session,-1,self.default_perf,self.default_perf)
        self.best_perf=self.default_perf

    def finish(self):
        self.session.close()
    
    def generate_candidates(self, batch_size=1):
        assert 0, "Undefined"
    
    def evaluate_candidates(self, candidates):
        assert 0, "Undefined"

    def reflect_feedback(perfs):
        assert 0, "Undefined"



    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf = None, FLOAT_MAX
        i = 0
        while i<budget+1:
            opt_setting = self.generate_candidates(batch_size=batch_size)
            exec_time,bin_md5= self.evaluate_candidates(opt_setting)
            if exec_time==-1 or exec_time==None or bin_md5==None:
                continue
            
            if exec_time < best_perf:
                best_perf = exec_time
                best_opt_setting = opt_setting
            
            if exec_time!=FLOAT_MAX and best_perf!=FLOAT_MAX:
                seq=f"[{i}] current trial: {exec_time:.3f}s, best performance so far: {best_perf:.3f}s"
                TunerRecord.add(self.session,i,exec_time,best_perf)
                print(seq)
                i += 1
            
            else:
                print('Compiler error!')
            self.reflect_feedback([exec_time])

        return best_opt_setting, best_perf
    
    def get_best_result(self):
        min_case_record = (
                self.evaluator.session.query(Md5TestTime)
                .order_by(Md5TestTime.case.asc())
                .first())
        min_time=min_case_record.case
        min_md5=min_case_record.bin_md5
        options=self.evaluator.session.query(OptionsMd5).filter(OptionsMd5.md5==min_md5).first()
        best_seq=options.options
        return min_time,best_seq

