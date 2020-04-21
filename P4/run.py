import os
import subprocess
import sys
import csv

if not len(sys.argv) == 2:
    print('Error. Expected number of array copies')
    exit(1)

# Get data directories
data_dir = 'data'
dirs = os.listdir(data_dir)
dirs.sort()
dirs.remove('3')
dirs.remove('9')

print(dirs)

# Define paths for each binary and group them according to their functionality
specs_binary = 'bin/deviceProperties'
sum_binaries = ['bin/vectorAddSec', 'bin/vectorAddCUDA']

# Create output directory if it doesn't exist
out_dir = 'out'

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

# Run program that gets specs and save output
specs_out = subprocess.run(specs_binary, stdout=subprocess.PIPE).stdout.decode('utf-8')
with open(f'{out_dir}/specs.txt', 'w') as f:
    f.write(specs_out)

# Run binaries that sum arrays
out_sum = {b: [] for b in sum_binaries}
n_copies = sys.argv[1]

for num in dirs:
    input0 = f'{data_dir}/{num}/input0.raw'
    input1 = f'{data_dir}/{num}/input1.raw'

    for b in sum_binaries:
        output = subprocess.run([b, input0, input1, n_copies], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')[-2]
        output = output.split(',')
        out_tuple = (int(output[0]), float(output[1]))
        out_sum[b].append(out_tuple)

# Sort results by array size
for k in out_sum.keys():
    out_sum[k].sort(key=lambda x: x[0])

# Write results
for out_file, k in zip([f'{out_dir}/sequential.csv', f'{out_dir}/cuda.csv'], out_sum.keys()):
    with open(out_file, 'w') as f:
        csv_out = csv.writer(f)

        for row in out_sum[k]:
            csv_out.writerow(row)


