# [LCTES 2025] GroupTuner: Efficient Group-Aware Copmiler Auto-Tuning

## Overview

This is an implementation of GroupTuner in LCTES 2025 paper: GroupTuner:Efficient Group-Aware Compiler Auto-Tuning. The artifacts in this paper are in the form of standalone Python scripts for the experiments. We provide artifacts, datasets and scripts to automatically generate the experimental results. Our experiments are primarily tested on GCC 9.2.0, and different compiler version and hardware may cause the results to vary. Due to the potential for performance variablity across different environments, particularly in VM images or Docker, where virtualization layers may introduce additional scheduling overhead, I/O latency, or performance jitter, we recommend running the experiments on a bare-metal machine. To ensure measurement accuracy and consistency, we provide the source code and execution scripts for GroupTuner, allowing users to install the necessary dependencies and reproduce the results in a native environment. For more detailed information, please refert to Artifact_Evaluation.pdf


## Prerequisites
To ensure the stability of the experimental results, we disable Turbo Boost and bind each process to a set of isolated CPU cores. And we use Perf to measure the execution time of program accurately.
* Disable the Turbo Boost feature of Intel. Reboot the server and enter the BOIS setup. Locate the CPU-related settings, typically found under sections such as Performance or Advanced. Then look for the Intel Turbo Boost option and set it to Disabled or Off. After making the changes, save the settings and restart the server.
* Isolate a specific CPU core (core 0 in our experiment).
```
sudo vim /etc/default/grub
# Add isolcpus=0 to the GRUB_CMDLINE_LINUX configuration line.
sudo update-grub
sudo reboot
```
* Perf Installation.
```
sudo apt update
sudo apt install linux-tools-common linux-tools-$(uname -r) linux-tools-generic
sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```

* GCC-9.2.0 Installation.
```
cd GroupTuner
chmod +x gcc_install.sh
find GroupTuner/Dataset/dataset/cBench_V1.1 -type f -name "__run" -exec chmod +x {} \;
./gcc_install.sh
```

* Create a virual enviroment and install necessary packages.
```
conda create -n grouptuner python=3.9
conda activate grouptuner
cd GroupTuner
pip install -r requirements.txt
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
```
## Running GroupTuner and Other Methods
The full experiment execution logic is encapsulated in main_cbench.py and main_poly.py, which respectively perform iterative tuning using GroupTuner and other methods (SA, SRTuner, CFSCA, BOCA, RIO) on cBench and PolyBench.

```
python3 main_cbench.py --gcc-path $PWD/gcc_install/gcc-build/build/bin/gcc --round 500
python3 main_poly.py --gcc-path $PWD/gcc_install/gcc-build/build/bin/gcc --round 500
```

We recommend setting round to 500 for sufficient convergence (as we used in experiment), but you may adjust it according to your available resources. As dynamic iteration is time-consuming, we suggest to execute experiments on multiple servers in parallel. However, we strongly suggest that the tuning results of different methods for the same program be collected on the same server to avoid performance variations caused by hardware differences.

Note that the tuning process is continuous and cannot be interrupted.  
If an experiment is accidentally interrupted, please rerun the experiment for that method from the start.

The experimental results are stored under GroupTuner/Dataset/output/.

For more detailed instructions, pelase refer to Artifact_Evaluation.pdf.

## Results Visualization

* Table 3 (Main Results)
```
cd show_result
python3 Table3.py
```

The result will be saved to show_result/Table3.csv

* Figure 4 Results

```
cd show_result
python3 Fig4.py
```

The result will be saved to show_result/Fig4.png

* Figure 7 Results

```
cd show_result
python3 Fig7.py
```

Since Figure 7 analyzes the tuning results for each individual program, the outputs are saved separately under show_result/Fig7_result/, with one figure per program.

* Figure 8 Results

```
cd show_result
python3 Fig8.py
```

Similarly, the outputs are saved under show_result/Fig8_result/, with one figure per program.

* Figure 5 Results

```
cd show_result
python3 Fig5.py
```

The result will be saved to show_result/Fig5.png

* Figure 6 Results

```
cd show_result
python3 Fig6.py
```

The result will be saved to show_result/Fig6.png. The time-consuming analysis in Figure 6 is based on the runtime of the RIO algorithm.  Please ensure that the RIO results exist under both Dataset/output/cbench/test/RIO/ and Dataset/output/polybench/test/RIO/ before running Fig6.py.

## Structure highlight
```
|- Dataset/                          #Dataset and output
   |- dataset/
      |- cBench_V1.1                 # cBench Dataset
      |- polybench-c-4.2             # polybench Dataset
      |- cbench_prog_info.csv        # cbench info
      |- poly_prog_info.csv          # poly info
   |- process/
      |- database.py                 # Database structrue of GroupTuner
      |- heads.py                    # Evalautor of cBench and polyBench
   |- output/                        # Auto-tuning result
      |- cbench                      # Auto-tuning result of cBench
      |- polybench                   # Auto-tuning result of cBench
|- Engine/                   # Auto-tuning Algorithms Structure
   |- lib/                   # library of auto-tuning framework
   |- common.py              # basic structure of the tuning framework
   |- grouptuner.py          # GroupTuner algorithm
   |- srtuner.py             # SRTuner algorithm
   |- baseline_tuners.py     # Random Iterative Optimization algorithm(RIO)
   |- cfsca.py               # CFSCA algorithm
   |- bocatuner.py           # BOCA algorithm
   |- sa_tuner.py            # Simulated Algorithm(SA)
|- Space/                       # Search space information
   |- all_opts.txt              # all 206 options as default search space
   |- opts_group.txt            # all 206 options divided into 15 groups
   |- option_O3_state.json      # the state of every option of O3
   |- search_space.py           # structure of search space, including Default_Space and Group_Space
|- show_result/                      # scripts of result visualization
|- gcc_install.sh                    # script of install GCC-9.2.0
|- main_cbench.py                    # execute experiments of auto-tuning cbench
|- main_poly.py                      # execute experiments of auto-tuning polybench
 
```


# Feedback

Should you have any questions, please post to the issue page or email Bingyu Gao via bingyugao@stu.pku.edu.cn