import os
from GroupTuner.Dataset.process.database import TunerRecord
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

def calc_perf(a,b):
    return (a-b)/b

module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)

def convert_time2int(time_str):
    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp())

def compute_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(n)

    t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)

    CI_range = t_value * std_error

    return mean, CI_range

def plot_effiency_CI(data):


    categories = ['GroupTuner', 'SRTuner', 'SA', 'BOCA', 'CFSCA']
    means = []
    CI_ranges = []

    for cat in categories:
        mean, CI_range = compute_confidence_interval(data[cat])
        means.append(mean)
        CI_ranges.append(CI_range)

    colors = ["royalblue", "purple", "red", "orange", "green"]

    plt.figure(figsize=(10, 6))

    bars = []
    for i, category in enumerate(categories):
        bar = plt.bar(category, means[i], edgecolor=colors[i], color="none", linewidth=2)
        bars.append(bar)

        plt.errorbar(i, means[i], yerr=CI_ranges[i], fmt='none', 
                    ecolor=colors[i], elinewidth=2, capsize=5)

        plt.text(i, means[i] + CI_ranges[i] + 5, f"{means[i]:.2f}", ha='center', fontsize=12)

    plt.xlabel("Algorithms", fontsize=14)
    plt.ylabel("Time consuming relative to RIO (%)", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f'{MODULE_DIR}/Fig7.png', format='png')


result={}
algorithm=['GroupTuner','CFSCA','BOCA','SRTuner','RIO','SA']

max_round=500

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
        first_round=session.query(TunerRecord).filter(TunerRecord.round==1).first().local_time
        last_round=session.query(TunerRecord).filter(TunerRecord.round==max_round).first().local_time
        duration=convert_time2int(last_round)-convert_time2int(first_round)

        result[file][algo]=duration

final_result={key:[] for key in algorithm}
for file in result:
    if 'RIO' not in result[file]:
        continue
    for algo in algorithm:
        final_result[algo].append(round(result[file][algo]/result[file]['RIO']*100,2))
del final_result['RIO']

plot_effiency_CI(final_result)