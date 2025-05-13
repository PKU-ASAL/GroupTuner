import os
from GroupTuner.Dataset.process.database import TunerRecord
from sqlalchemy import create_engine, MetaData,asc
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import numpy as np

def calc_perf(a,b):
    return (a-b)/b*100

module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)
result={}
algorithm=['GroupTuner','RIO','SA','CFSCA','BOCA','SRTuner']

# show performance average improvements over -O3 every 50 rounds 
interval=50

for root, dirs, filenames in os.walk(f'{MODULE_DIR}/../Dataset/output'):
    for filename in filenames:
        path=os.path.join(root, filename)
        algo=root.split('/')[-1]
        file=filename.split('.')[0]
        if file not in result:
            result[file]={}
        engine=create_engine(f'sqlite:///{path}')
        metadata=MetaData()
        metadata.bind=engine
        metadata.reflect(engine)
        Session = sessionmaker(bind=engine)
        session = Session()
        O3_perf=session.query(TunerRecord).filter(TunerRecord.round==-1).first().perf
        result[file][algo]={'O3':O3_perf,'other':[]}
        max_round = session.query(TunerRecord.round).filter(TunerRecord.round != -1).order_by(TunerRecord.round.desc()).first()[0]
        for start in range(1, max_round + 1, interval):
            end = start + interval - 1
            best_record = (
                session.query(TunerRecord)
                .filter(TunerRecord.round >= start, TunerRecord.round <= end)
                .filter(TunerRecord.round != -1)
                .order_by(asc(TunerRecord.perf))
                .first()
            ).perf
            if len(result[file][algo]['other'])==0 or best_record < result[file][algo]['other'][-1]:
                result[file][algo]['other'].append(best_record)
            else:
                result[file][algo]['other'].append(result[file][algo]['other'][-1])

final_result={}
for file in result:
    final_result[file]={key:[] for key in algorithm}

    O3_perf=sum(v['O3'] for v in result[file].values())/len(result[file])
    for algo in result[file]:

        final_result[file][algo]=[round(calc_perf(O3_perf,x),2) for x in result[file][algo]['other']]

show_result={key:[[] for i in range(10)] for key in algorithm}
for bench_results in final_result.values():
    for algo in bench_results:
        for i in range(len(bench_results[algo])):
            show_result[algo][i].append(bench_results[algo][i])
for algo in show_result:
    for i in range(len(show_result[algo])):
        tmp=np.mean(show_result[algo][i])
        if np.isnan(tmp):
            tmp=0
        show_result[algo][i]=round(float(tmp),2)

categories = ['GroupTuner', 'RIO', 'SA', 'CFSCA','BOCA','SRTuner',]
colors = ["royalblue",  "brown", "red",'green','darkorange', "purple"]
plt.figure(figsize=(10, 6))
x_values = [50 * (i + 1) for i in range(len(next(iter(show_result.values()))))]
for i, (key, values) in enumerate(show_result.items()):

    plt.plot(x_values, values, marker='o', linestyle='-', color=colors[i], label=key)
    if key == "GroupTuner":
        for x, y in zip(x_values, values):
            plt.text(x, y + 0.2, f"{y:.2f}", ha='center', fontsize=10, color='black')


legend = plt.legend(fontsize=12, edgecolor='black')


plt.xlabel("Iteration Round", fontsize=14)
plt.ylabel("Performance Improvement Over -O3 (%)", fontsize=14)


plt.xticks(x_values, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')


plt.savefig(f'{MODULE_DIR}/Fig3.png', format='png')
