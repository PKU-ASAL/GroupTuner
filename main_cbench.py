from Dataset.process.heads import cBench
from Engine.baseline_tuners import RandomTuner
from Engine.satuner import SATuner
from Engine.grouptuner import GroupTuner
from Engine.bocatuner import BOCATuner
from Engine.srtuner import SRTuner
from Engine.cfsca import CFSCATuner
from Space.search_space import Default_Space,Group_Space
import argparse
import os
import csv


module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)

parser = argparse.ArgumentParser()
parser.add_argument("--gcc-path", type=str, default='gcc')
parser.add_argument("--round", type=str, default=500)
args = parser.parse_args()
bin_path=args.gcc_path
round=int(args.round)

if __name__ == '__main__':
    prog=[]
    with open(f'{MODULE_DIR}/Dataset/dataset/cbench_prog_info.csv','r') as f:
        f_csv=csv.reader(f)
        for row in f_csv:
            prog.append(row[0])

    test_cases=['automotive_qsort1','bzip2d','network_dijkstra']
    for p in test_cases:
        print(f'{p} GroupTuner:')
        space=Group_Space('opts_group.txt')
        evaluator=cBench(p,f'test','GroupTuner',bin_path)
        tuner=GroupTuner(search_space=space,evaluator=evaluator,name='GroupTuner')
        tuner.tune(round)
        best_time,best_seq=tuner.get_best_result()
        print(f'Tunning finished, best time: {best_time:.3f}s')
        print(f'Best sequence: {best_seq}')
        print(f'The detaild tuning process is saved in {evaluator.db_path}')
    
        print(f'{p} SA:')
        space=Default_Space()
        evaluator=cBench(p,f'test','SA',bin_path)
        tuner=SATuner(search_space=space,evaluator=evaluator,name='SA')
        tuner.tune(round)
        best_time,best_seq=tuner.get_best_result()
        print(f'Tunning finished, best time: {best_time:.3f}s')
        print(f'Best sequence: {best_seq}')
        print(f'The detaild tuning process is saved in {evaluator.db_path}')

        print(f'{p} SRTuner:')
        space=Default_Space()
        evaluator=cBench(p,'test','SRTuner',bin_path)
        tuner=SRTuner(search_space=space,evaluator=evaluator,name='SRTuner')
        tuner.tune(round)
        best_time,best_seq=tuner.get_best_result()  
        print(f'Tunning finished, best time: {best_time:.3f}s')
        print(f'Best sequence: {best_seq}')
        print(f'The detaild tuning process is saved in {evaluator.db_path}')
        
        print(f'{p} BOCA:')
        space=Default_Space()
        evaluator=cBench(p,'test','BOCA',bin_path)
        tuner=BOCATuner(search_space=space,evaluator=evaluator,name='BOCA')
        tuner.tune(round)
        best_time,best_seq=tuner.get_best_result()
        print(f'Tunning finished, best time: {best_time:.3f}s')
        print(f'Best sequence: {best_seq}')
        print(f'The detaild tuning process is saved in {evaluator.db_path}')
    
        print(f'{p} RIO:')
        space=Default_Space()
        evaluator=cBench(p,'test','RIO',bin_path)
        tuner=RandomTuner(search_space=space,evaluator=evaluator,name='RIO')
        tuner.tune(round)
        best_time,best_seq=tuner.get_best_result()
        print(f'Tunning finished, best time: {best_time:.3f}s')
        print(f'Best sequence: {best_seq}')
        print(f'The detaild tuning process is saved in {evaluator.db_path}')
            
        print(f'{p} CFSCA:')
        space=Default_Space()
        evaluator=cBench(p,'test','CFSCA',bin_path)
        tuner=CFSCATuner(search_space=space,evaluator=evaluator,name='CFSCA')
        tuner.tune(round)
        best_time,best_seq=tuner.get_best_result()
        print(f'Tunning finished, best time: {best_time:.3f}s')
        print(f'Best sequence: {best_seq}')
        print(f'The detaild tuning process is saved in {evaluator.db_path}')