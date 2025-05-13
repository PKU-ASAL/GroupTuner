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

def plot_single_file_from_dict(file_name,data):
    categories = ["GroupTuner", "BOCA", "SRTuner", "CFSCA", "SA", "RIO"]
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    line_colors = ["royalblue", "orange", "purple", "green", "red", "brown"]


    fig, ax = plt.subplots(figsize=(8, 6))

    max_performance_values = {}

    for i, cat in enumerate(categories):
        values = data.get(cat, [])
        if not values:
            continue 

        values = np.array(values)

        max_values = np.maximum.accumulate(values)

        x_values = np.arange(len(max_values))

        max_performance_values[cat] = max_values

        ax.plot(x_values, max_values, color=line_colors[i], linestyle=line_styles[i], linewidth=2, label=cat)

    ax.set_title(file_name, fontsize=16)
    ax.set_xlabel("Iteration Round", fontsize=14)
    ax.set_ylabel("Max Performance Improvement Over -O3", fontsize=14)


    if max_performance_values:
        y_max = max([np.max(values) for values in max_performance_values.values() if len(values) > 0])
        ax.set_ylim(0, y_max * 1.05)

    ax.legend(fontsize=12, loc="best", frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(f'{MODULE_DIR}/Fig8_result/{file_name}.png', format='png')
    plt.show()
    plt.close()


result={}
algorithm=['GroupTuner','RIO','SA','CFSCA','BOCA','SRTuner']

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

for file in final_result:
    plot_single_file_from_dict(file, final_result[file])

