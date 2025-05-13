import os
from GroupTuner.Dataset.process.database import TunerRecord
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calc_perf(a,b):
    return (a-b)/b

module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)

def plot_average_performance_from_dict(data):

    categories = ["GroupTuner", "BOCA", "SRTuner", "CFSCA"]

    plt.figure(figsize=(10, 6))
    colors = ["royalblue", "darkorange", "purple", "green"]
    fill_colors = ["lightblue", "navajowhite", "thistle", "lightgreen"]

    window_size = 10
    all_values = []
    for cat in categories:
        values = np.array(data.get(cat, []))
        if values.size == 0:
            continue
        all_values.extend(values)

    y_min, y_max = -0.15, max(all_values)
    y_range = y_max - y_min
    y_min -= y_range * 0.05
    y_max += y_range * 0.05

    for i, cat in enumerate(categories):
        values = np.array(data.get(cat, []))
        if values.size == 0:
            continue
        x_values = np.arange(len(values))
        trend_x = np.arange(0, len(values), window_size)
        trend_y = np.array([np.mean(values[i:i+window_size]) for i in trend_x])
        trend_min = np.array([np.min(values[i:i+window_size]) for i in trend_x])
        trend_max = np.array([np.max(values[i:i+window_size]) for i in trend_x])

        plt.fill_between(trend_x, trend_min, trend_max, color=fill_colors[i], alpha=0.3)

        plt.plot(trend_x, trend_y, color=colors[i], linestyle='-', linewidth=2, marker='o', label=cat)

    plt.xlabel("Iteration Round", fontsize=14)
    plt.ylabel("Average Performance Improvement over O3", fontsize=14)
    plt.ylim(y_min, y_max)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{MODULE_DIR}/Fig6.png', format='png')
    plt.show()
    plt.close()


result={}
algorithm=['GroupTuner','CFSCA','BOCA','SRTuner']


for root, dirs, filenames in os.walk(f'{MODULE_DIR}/../Dataset/output'):
    for filename in filenames:
        path=os.path.join(root, filename)
        algo=root.split('/')[-1]
        if algo not in algorithm:
            continue
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
        records = (
        session.query(TunerRecord.round, TunerRecord.perf)
        .filter(TunerRecord.round > 0)
        .order_by(TunerRecord.round.asc())
        .all())
        perf_list = [perf for round, perf in records]
        result[file][algo]=[O3_perf,perf_list]

final_result={}
for file in result:
    final_result[file]={key:[] for key in algorithm}
    O3_perf=sum(v[0] for v in result[file].values())/len(result[file])
    for algo in algorithm:
        if algo not in result[file]:
            continue
        final_result[file][algo]=[calc_perf(O3_perf,x) for x in result[file][algo][1]]

show_result={key:[[] for i in range(500)] for key in algorithm}

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

plot_average_performance_from_dict(show_result)


