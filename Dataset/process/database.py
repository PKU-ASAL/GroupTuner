from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
import time
# database (per_project)
# 1. [options_md5] option, md5 of a.out
# 2. [md5_test_time] md5 of a.out, test_name, perf time (num_per_test times)
# ---special---
# 3. [compile_error] option , error_message (compile failed)
# 4. [run_error] md5 of a.out, test_name, error_message (run failed)
# 5. [wrong_output] md5 of a.out, test_name, md5 of output file (wrong output)


# engine = create_engine('sqlite:///database1.db', echo=True)

Base = declarative_base()

# options, md5
class OptionsMd5(Base):
    __tablename__ = 'options_md5'

    options = Column(String,primary_key=True)
    md5 = Column(String)

    @classmethod
    def add(cls,session,options,md5):
        add_state=session.query(cls).filter(cls.options==options).first()
        if not add_state:
            session.add(OptionsMd5(options=options,md5=md5))
            session.commit()
            return True
        return False
    

# md5, test_name, time
class Md5TestTime(Base):
    __tablename__ = 'md5_test_time'


    bin_md5 = Column(String,primary_key=True)
    case=Column(Float)
    std=Column(Float)
    local_time=Column(String)

    @classmethod
    def add(cls,session,md5,times,std):
        add_status=session.query(cls).filter(cls.bin_md5==md5).first()

        if not add_status:
            session.add(Md5TestTime(bin_md5=md5,
                                    case=times,
                                    std=std,
                                    local_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())
                                    ))
            session.commit()
            return True
        return False
    
    @classmethod
    def get_time(cls,session,md5):
        status=session.query(cls).filter(cls.bin_md5==md5).first()
        if status:
            return status.case
        return False
    
    @classmethod
    def exist(cls,session,md5):
        status=session.query(cls).filter(cls.bin_md5==md5).first()
        if status:
            return True
        return False



class TunerRecord(Base):
    __tablename__ = 'tuner_record'

    round=Column(Integer,primary_key=True)
    perf=Column(Float)
    best_perf=Column(Float)
    local_time=Column(String)

    @classmethod
    def add(cls,session,round,perf,best_perf):
        if round==-1 and session.query(cls).filter(cls.round==round).first():
            return True
        session.add(TunerRecord(round=round, perf=perf,best_perf=best_perf,
                                 local_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        session.commit()



# compile error: options, error_message
class CompileError(Base):
    __tablename__ = 'compile_error'

    id = Column(Integer, primary_key=True)
    options = Column(String)
    error_message = Column(String)
    local_time=Column(String)

    @classmethod
    def add(cls,session,options,cp_error):
        session.add(CompileError(options=options, error_message=cp_error,
                                 local_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        session.commit()

# run error: exec_md5, test_name, error_message
class RunError(Base):
    __tablename__ = 'run_error'

    id = Column(Integer, primary_key=True)
    exec_md5 = Column(String)
    test_name = Column(String)
    error_message = Column(String)
    local_time=Column(String)

    @classmethod
    def add(cls,session,exec_md5,name,run_error):
        session.add(RunError(exec_md5=exec_md5, test_name=name, error_message=run_error,
                             local_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        session.commit()

# wrong output: exec_md5, test_name, output_md5
class WrongOutput(Base):
    __tablename__ = 'wrong_output'

    id = Column(Integer, primary_key=True)
    exec_md5 = Column(String)
    test_name = Column(String)
    output_md5 = Column(String)
    local_time=Column(String)

    @classmethod
    def add(cls,session,exec_md5,name,md5):
        session.add(WrongOutput(exec_md5=exec_md5, test_name=name, output_md5=md5,
                                local_time=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime())))
        session.commit()

