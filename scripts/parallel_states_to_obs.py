import argparse
import json
import random
import subprocess
import copy
import psutil
import sys
import time
import os
import h5py

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_workers', type=int, required=True)
argparser.add_argument('--dataset', type=str, required=True)
num_workers = argparser.parse_args().num_workers
dataset_path = argparser.parse_args().dataset
file_name = dataset_path[dataset_path.rfind('/') + 1:dataset_path.rfind('.hdf5')]
print('file name:', file_name)

# python parallel_states_to_obs.py --dataset ../datasets/generated/NutAssemblySquare/2024-11-11-merged-cpgen-naive-env-reset/18-24-49-175405_99demos_199demos_with_obs.hdf5 --num_workers 20

f = h5py.File(dataset_path, "r")
num_demos = len(f["data"].keys())
f.close()

each = num_demos // num_workers
extra = num_demos % num_workers
end = 0

print('num demos:', num_demos)

processes = []
for i in range(num_workers):
    start = end
    end += each
    if i < extra:
        end += 1

    print('start:', start, 'end:', end)

    gpu_id = i % 8
    run_line = f"python dataset_states_to_obs.py --dataset {dataset_path} --start {start} --end {end} --output_name {file_name}{i}.hdf5 --gpu_id {gpu_id} --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 90 --camera_width 160 --depth --segmentation instance --compress --exclude-next-obs"
    processes.append(subprocess.Popen(run_line, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    time.sleep(2)
    actual_pid = None
    for child in psutil.Process(processes[-1].pid).children(recursive=True):
        name = child.name().lower()
        if "python" in child.name().lower():
            actual_pid = child.pid
            break

    # pin to core i
    hex_string = hex(int(2 ** (i)))
    subprocess.run(['taskset', '-p', hex_string, str(actual_pid)])
    print("Started process", i)
print("Started all processes")
print("\n================================\n")

try:
    for i, p in enumerate(processes):
        p.wait()
        print("Finished process", i)
    print("All processes finished")
except KeyboardInterrupt:
    for i, p in enumerate(processes):
        p.terminate()  # or p.kill() if necessary
        time.sleep(2)  # Give it some time
        if p.poll() is None:  # Check if it's still running
            p.kill()  # Force stop
        print("Terminated process", i)
    print("All processes terminated")
    sys.exit(0)