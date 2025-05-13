import os
import csv
from GroupTuner.Dataset.process.database import TunerRecord
from sqlalchemy import create_engine, MetaData,asc
from sqlalchemy.orm import sessionmaker

def calc_perf(a,b):
    return (a-b)/b*100

module_path = os.path.abspath(__file__)
MODULE_DIR = os.path.dirname(module_path)
result={}
algorithm=['GroupTuner','RIO','SA','CFSCA','BOCA','SRTuner']
# Directly select the best result from all iteration rounds, so make sure the iteration rounds among different
# algorithms are the same.
# If not, please modify the code in line 31
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
        best_perf=session.query(TunerRecord).filter(TunerRecord.round!=-1).order_by(asc(TunerRecord.perf)).first().perf
        result[file][algo]=[O3_perf,best_perf]
final_result={}
for file in result:
    final_result[file]={key:'-' for key in algorithm}
    O3_perf=sum(v[0] for v in result[file].values())/len(result[file])
    for algo in result[file]:
        final_result[file][algo]=calc_perf(O3_perf,result[file][algo][1])

sums={}
counts={}
for outer in final_result.values():
    for k, v in outer.items():
        if v=='-':
            continue
        sums[k] = sums.get(k, 0) + v
        counts[k] = counts.get(k, 0) + 1
final_result['Avg']= {k: sums[k] / counts[k] for k in sums}

# Store the result in Table3.csv
# The result is the average performance improvement of all benchmarks
# over the default O3 performance
with open(f'{MODULE_DIR}/Table3.csv','w') as f:
    f_csv=csv.writer(f)
    f_csv.writerow(['%']+algorithm)
    for key in final_result:
        add_row=[key]
        for v in algorithm:
            if type(final_result[key][v])==type(0.1):
                final_result[key][v]=round(final_result[key][v],2)
            add_row.append(final_result[key][v])
        f_csv.writerow(add_row)