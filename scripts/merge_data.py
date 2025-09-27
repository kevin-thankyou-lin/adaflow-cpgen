import argparse
import h5py

# python merge_data.py --path ../datasets/generated/NutAssemblySquare/2024-11-11-merged-cpgen-naive-env-reset/18-24-49-175405_99demos_199demos_with_obs.hdf5 --num_workers 20

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, required=True)
argparser.add_argument('--num_workers', type=int, required=True) 

path = argparser.parse_args().path
no_ext = path[:path.rfind('.hdf5')]
num_workers = argparser.parse_args().num_workers

print('opening:', f"{no_ext}0.hdf5")
big_data = h5py.File(f"{no_ext}0.hdf5", 'a') # a for append mode
count = len(big_data['data'].keys())

for i in range(1, num_workers):
    cur_path = f"{path}_{i}/demo.hdf5"
    cur_data = h5py.File(f"{no_ext}{i}.hdf5", 'r')
    for key in cur_data['data'].keys():
        new_key = f"demo_{count}"
        big_data.copy(cur_data['data'][key], big_data['data'], name=new_key)
        count += 1
    cur_data.close()
big_data.close()

print("Num successes:", count)