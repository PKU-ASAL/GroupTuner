import csv
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from GroupTuner.Dataset.process.database import OptionsMd5, Md5TestTime, CompileError, RunError, WrongOutput, Base
import subprocess
import psutil
import signal
import statistics
import time
def preexec():
    os.setpgrp()

module_path = os.path.abspath(__file__)

MODULE_DIR = os.path.dirname(module_path)


cbench_prog={}
with open(f'{MODULE_DIR}/../dataset/cbench_prog_info.csv','r') as f:
    f_csv=csv.reader(f)
    for row in f_csv:
        cbench_prog[row[0]]=row[1]

def get_md5(stdout):
    return stdout.decode('ascii').split(' ')[0]


class Dataset():
    def __init__(self,name):
        self.name=name

    def pre_process(self):
        pass

    def compile(self):
        pass

    def execute(self):
        pass


class cBench(Dataset):
    def __init__(self,name,algorithm_type,space_type,bin_path):

        self.name=name
        local_time=time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime())
        self.db_path=f'{MODULE_DIR}/../output/cbench/{algorithm_type}/{space_type}/{self.name}-{local_time}.db'
        self.path=f'{MODULE_DIR}/../dataset/cBench_V1.1/{self.name}/src'
        directory=os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        self.output=cbench_prog[self.name]
        self.bin_path=bin_path
        self.default_perf,self.std_out=self.evaluate_default()
        self.cfsca_path=self.path


    def evaluate_default(self):
        command = f"""cd {self.path};
                make clean;
                make CCC_OPTS="-w -O3" LD_OPTS="-o a.out" ZCC="{self.bin_path}" LDCC="{self.bin_path}";
                """
        cp = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode!=0:
            CompileError.add(self.session,'-O3',cp.stderr.decode('utf-8'))
            return -1
        
        md5_command = f"md5sum {self.path}/a.out"
        p=subprocess.run(md5_command,shell=True,stdout=subprocess.PIPE)
        md5=get_md5(p.stdout)
        OptionsMd5.add(self.session,'-O3',md5)
        eclapse_time=[]
        output_md5=''
        for i in range(5):
            run_commands = f"""cd {self.path};
                taskset -c 0 perf stat -- ./__run 1;
                """
            try:
                p = subprocess.Popen(run_commands, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,preexec_fn=preexec)
                _, err = p.communicate(timeout=60)
                stderr=err.decode('utf-8',errors='ignore')
            except subprocess.TimeoutExpired:
                if psutil.pid_exists(p.pid):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                RunError.add(self.session,md5,i,'timeout')
                print('error: time out')
                return -1

            if i==0:
                md5_command = f"md5sum {self.path}/{self.output}"
                p2 = subprocess.run(md5_command, shell=True, stdout=subprocess.PIPE)
                output_md5 = get_md5(p2.stdout)
                if p2.returncode != 0 or p.returncode!=0:
                            
                    RunError.add(self.session,md5,i,stderr)
                    print('error: run_command failed')
                    return -1,output_md5

            index=stderr.find('seconds time elapsed')
            eclapse_time.append(float(stderr[:index].split(' ')[-2]))
        derta=statistics.stdev(eclapse_time)
        eclapse_time=sum(eclapse_time)/len(eclapse_time)

        Md5TestTime.add(self.session,md5,eclapse_time,derta)
        self.clean()
        return eclapse_time,output_md5

    def compile(self,options):
        
        if '-O' not in options:
            options='-O3 '+options
        command = f"""cd {self.path};
                make clean;
                make CCC_OPTS="-w {options}" LD_OPTS="-o a.out" ZCC="{self.bin_path}" LDCC="{self.bin_path}";
                """
        
        cp = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode==0:
            md5_command = f"md5sum {self.path}/a.out"
            p=subprocess.run(md5_command,shell=True,stdout=subprocess.PIPE)
            md5=get_md5(p.stdout)
            OptionsMd5.add(self.session,options,md5)

            return md5
        else:
            CompileError.add(self.session,options,cp.stderr.decode('utf-8'))
            return -1

    def clean(self):
        cmd=f"""cd {self.path};
            make clean ;
            rm -rf a.out;
            rm -rf {self.output};
        """
        subprocess.run(cmd,shell=True,stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)

    def run(self,md5_bin):
        eclapse_time=[]
        for i in range(5):
            run_commands = f"""cd {self.path};
                taskset -c 0 perf stat -- ./__run 1;
                """
            try:
                p = subprocess.Popen(run_commands, shell=True, stderr=subprocess.PIPE,stdout=subprocess.DEVNULL, preexec_fn=preexec)
                out, err = p.communicate(timeout=60)
                stderr=err.decode('utf-8',errors='ignore')
            except subprocess.TimeoutExpired:
                if psutil.pid_exists(p.pid):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                RunError.add(self.session,md5_bin,i,'timeout')
                print('error: time out')
                return -1

            if i==0:
                md5_command = f"md5sum {self.path}/{self.output}"
                p2 = subprocess.run(md5_command, shell=True, stdout=subprocess.PIPE)
                md5 = get_md5(p2.stdout)
                if p2.returncode != 0 or p.returncode!=0:
                            
                    RunError.add(self.session,md5_bin,i,stderr)
                    print('error: run_command failed')
                    return -1
                
                if md5!=self.std_out:
                    print('error:output different')
                    WrongOutput.add(self.session,md5_bin,i,md5)
                    return -1
            index=stderr.find('seconds time elapsed')
            eclapse_time.append(float(stderr[:index].split(' ')[-2]))
        derta=statistics.stdev(eclapse_time)
        eclapse_time=sum(eclapse_time)/len(eclapse_time)

        Md5TestTime.add(self.session,md5_bin,eclapse_time,derta)
        self.clean()
        return eclapse_time
    

    def evaluate(self,options):
        bin_md5=self.compile(options)
        if bin_md5==-1:
            return None,None
        exec_time=Md5TestTime.get_time(self.session,bin_md5)
        if not exec_time:
            status=self.run(bin_md5)
            return status,bin_md5
        return exec_time,bin_md5


class PolyBench(Dataset):
    def __init__(self,name,algorithm_type,space_type,bin_path):
        self.name=name.split('/')[-1]
        self.db_path=f'{MODULE_DIR}/../output/polybench/{algorithm_type}/{space_type}/{self.name}.db'
        self.path=name
        directory=os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        self.cfsca_path=f'{MODULE_DIR}/../dataset/polybench-c-4.2/{self.path}'
        self.bin_path=bin_path
        self.default_perf,self.std_out=self.evaluate_default()
    def evaluate_default(self):
        command = f"""cd {MODULE_DIR}/../dataset/polybench-c-4.2;
                rm -rf a.out;
                rm -rf output.txt;
                {self.bin_path} -O3 -I utilities -I ./{self.path} utilities/polybench.c ./{self.path}/{self.name}.c -lm -DPOLYBENCH_DUMP_ARRAYS -o a.out;
                """
        cp = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode!=0:
            CompileError.add(self.session,'-O3',cp.stderr.decode('utf-8'))
            return -1
        md5_command = f"md5sum {MODULE_DIR}/../dataset/polybench-c-4.2/a.out"
        p=subprocess.run(md5_command,shell=True,stdout=subprocess.PIPE)
        md5=get_md5(p.stdout)
        OptionsMd5.add(self.session,'-O3',md5)
        
        eclapse_time=[]
        output_md5=''
        for i in range(5):
            run_commands = f"""cd {MODULE_DIR}/../dataset/polybench-c-4.2;
                taskset -c 0 perf stat -- bash -c './a.out 2>output.txt';
                """
            try:
                p = subprocess.Popen(run_commands, shell=True, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL,preexec_fn=preexec)
                _, err = p.communicate(timeout=60)
                stderr=err.decode('utf-8',errors='ignore')
            except subprocess.TimeoutExpired:
                if psutil.pid_exists(p.pid):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                RunError.add(self.session,md5,i,'timeout')
                print('error: time out')
                return -1

            if i==0:
                md5_command = f"md5sum {MODULE_DIR}/../dataset/polybench-c-4.2/output.txt"
                p2 = subprocess.run(md5_command, shell=True, stdout=subprocess.PIPE)
                output_md5 = get_md5(p2.stdout)
                if p2.returncode != 0 or p.returncode!=0:
                            
                    RunError.add(self.session,md5,i,stderr)
                    print('error: run_command failed')
                    return -1,output_md5

            index=stderr.find('seconds time elapsed')
            eclapse_time.append(float(stderr[:index].split(' ')[-2]))
        derta=statistics.stdev(eclapse_time)
        eclapse_time=sum(eclapse_time)/len(eclapse_time)

        Md5TestTime.add(self.session,md5,eclapse_time,derta)
        self.clean()
        return eclapse_time,output_md5
        
    def compile(self,options):
        if '-O' not in options:
            options='-O3 '+options
        command = f"""cd {MODULE_DIR}/../dataset/polybench-c-4.2;
                rm -rf a.out;
                rm -rf output.txt;
                {self.bin_path} {options} -I utilities -I ./{self.path} utilities/polybench.c ./{self.path}/{self.name}.c -lm -DPOLYBENCH_DUMP_ARRAYS -o a.out;
                """
        cp = subprocess.run(command, shell=True, stderr=subprocess.PIPE,stdout=subprocess.PIPE)
        if cp.returncode==0:
            md5_command = f"md5sum {MODULE_DIR}/../dataset/polybench-c-4.2/a.out"
            p=subprocess.run(md5_command,shell=True,stdout=subprocess.PIPE)
            md5=get_md5(p.stdout)
            OptionsMd5.add(self.session,options,md5)

            return md5
        else:
            CompileError.add(self.session,options,cp.stderr.decode('utf-8'))
            return -1

    def clean(self):
        cmd=f"""cd {MODULE_DIR}/../dataset/polybench-c-4.2;
            rm -rf output.txt;
            rm -rf a.out;
        """
        subprocess.run(cmd,shell=True,stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)

    def run(self,md5_bin):
        eclapse_time=[]
        for i in range(5):
            run_commands = f"""cd {MODULE_DIR}/../dataset/polybench-c-4.2;
                taskset -c 0 perf stat -- bash -c './a.out 2>output.txt';
                """
            try:
                p = subprocess.Popen(run_commands, shell=True, stderr=subprocess.PIPE, preexec_fn=preexec)
                out, err = p.communicate(timeout=60)
                stderr=err.decode('utf-8',errors='ignore')
            except subprocess.TimeoutExpired:
                if psutil.pid_exists(p.pid):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                RunError.add(self.session,md5_bin,i,'timeout')
                print('error: time out')
                return -1

            if i==0:
                md5_command = f"md5sum {MODULE_DIR}/../dataset/polybench-c-4.2/output.txt"
                p2 = subprocess.run(md5_command, shell=True, stdout=subprocess.PIPE)
                md5 = get_md5(p2.stdout)
                if p2.returncode != 0 or p.returncode!=0:
                            
                    RunError.add(self.session,md5_bin,i,stderr)
                    print('error: run_command failed')
                    return -1
                
                if md5!=self.std_out:
                    print('error:output different')
                    WrongOutput.add(self.session,md5_bin,i,md5)
                    return -1
            index=stderr.find('seconds time elapsed')
            eclapse_time.append(float(stderr[:index].split(' ')[-2]))
        derta=statistics.stdev(eclapse_time)
        eclapse_time=sum(eclapse_time)/len(eclapse_time)

        Md5TestTime.add(self.session,md5_bin,eclapse_time,derta)
        self.clean()
        return eclapse_time

    def evaluate(self,options):
        bin_md5=self.compile(options)
        if bin_md5==-1:
            return None,None
        exec_time=Md5TestTime.get_time(self.session,bin_md5)
        if not exec_time:
            status=self.run(bin_md5)
            return status,bin_md5
        return exec_time,bin_md5