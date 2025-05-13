import os
from GroupTuner.Dataset.process.database import TunerRecord
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calc_perf(a,b):
    return round((a-b)/b,2)


module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)

def plot_single_filtered_values(file_name,data):

    colors = ["lightblue", "darkorange"]
    line_colors = ["royalblue", "orange"]
    labels = ["GroupTuner", "BOCA"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for j, cat in enumerate(labels):
        values = data.get(cat, [])
        if not values:
            continue  
        values = np.array(values)
        
        window_size = 10
        trend_x = np.arange(0, len(values), window_size)
        trend_y = np.array([values[i:i+window_size].mean() for i in trend_x])
        trend_min = np.array([values[i:i+window_size].min() for i in trend_x])
        trend_max = np.array([values[i:i+window_size].max() for i in trend_x])

        ax.fill_between(trend_x, trend_min, trend_max, color=colors[j], alpha=0.2)

        ax.plot(trend_x, trend_y, color=line_colors[j], linestyle='-', linewidth=2, marker='o', label=cat)

    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

    ax.set_title(file_name, fontsize=18)
    ax.set_xlabel("Iteration Round", fontsize=14)
    ax.set_ylabel("Performance Improvement Over O3", fontsize=14)

    ax.legend(fontsize=14, loc="best", frameon=True)

    ax.grid(True, linestyle='--', alpha=0.6)


    plt.tight_layout()
    fig.savefig(f'{MODULE_DIR}/Fig7_result/{file_name}.png', format='png')
    plt.show()
    plt.close()


result={}
algorithm=['GroupTuner','BOCA']

for root, dirs, filenames in os.walk(f'{MODULE_DIR}/../Dataset/output'):
    for filename in filenames:
        path=os.path.join(root, filename)
        algo=root.split('/')[-1]
        if algo not in ['GroupTuner','BOCA']:
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
    final_result[file]={'GroupTuner':[],'BOCA':[]}
    O3_perf=sum(v[0] for v in result[file].values())/len(result[file])
    for algo in ['GroupTuner','BOCA']:
        if algo not in result[file]:
            continue

        final_result[file][algo]=[calc_perf(O3_perf,x) for x in result[file][algo][1]]

for file in final_result:
    plot_single_filtered_values(file, final_result[file])

