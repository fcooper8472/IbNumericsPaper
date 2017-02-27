import itertools
import os
import re
import subprocess
import time

import multiprocessing as mp
import numpy as np

try:
    import svg_to_webm
except ImportError as e:
    svg_to_webm = None
    quit("Exception: " + str(e))

try:
    import dominate
    from dominate.tags import *
except ImportError as e:
    dominate = None
    quit("Exception: " + str(e))

# Globally accessible directory paths, names, and variables
chaste_build_dir = os.environ.get('CHASTE_BUILD_DIR')
chaste_test_dir = os.environ.get('CHASTE_TEST_OUTPUT')

executable = os.path.join(chaste_build_dir, 'projects/IbNumericsPaper/apps', 'Exe_GrowingPopulationOfCells')
path_to_output = os.path.join(chaste_test_dir, 'ib_numerics_paper', 'Exe_GrowingPopulationOfCells')
path_to_sims = os.path.join(path_to_output, 'sim')
path_to_movies = os.path.join(path_to_output, 'movies')

if not(os.path.isfile(executable)):
    quit('Py: Could not find executable: ' + executable)

# List of command line arguments for the executable, and corresponding list of parameter names
command_line_args = [' --ID ', ' --CRL ', ' --CSC ', ' --TRL ', ' --TSC ', ' --DI ', ' --TS ']
params_list = ['simulation_id', 'cor_rest_length', 'cor_spring_const', 'tra_rest_length', 'tra_spring_const',
               'interaction_dist', 'num_time_steps']

# Time string when script is run, used for creating a unique archive name
today = time.strftime('%Y-%m-%dT%H%M')

# Param ranges (in lists, for itertools product)
crl = [0.25]
csc = np.linspace(1e6, 1e7, num=2)
trl = [0.01]
tsc = np.linspace(1e4, 1.01e4, num=2)
di = [0.02]
ts = [10000]

# An enumerated iterable containing every combination of the parameter ranges defined above
combined_iterable = list(itertools.product(crl, csc, trl, tsc, di, ts))


def main():
    run_simulations()
    combine_output()
    compress_output()


# Create a list of commands and pass them to separate processes
def run_simulations():

    print("Py: Starting simulations with " + str(mp.cpu_count()) + " processes")

    # Make a list of calls to a Chaste executable
    command_list = []

    if not os.path.exists(path_to_output):
        os.makedirs(path_to_output)

    params_file = open(path_to_output + '/params_file.csv', 'w')
    params_file.write(','.join(params_list) + '\n')

    base_command = 'nice -n 19 ' + executable

    for idx, param_set in enumerate(combined_iterable):

        params_file.write(str(idx) + ',' + ",".join(map(str, param_set)) + '\n')

        this_command = base_command
        this_command += ' --ID ' + str(idx)

        for arg in range(len(param_set)):
            this_command += command_line_args[arg+1] + str(param_set[arg])

        command_list.append(this_command)

    params_file.close()

    # Generate a pool of workers
    pool = mp.Pool(processes=mp.cpu_count())

    # Pass the list of bash commands to the pool and wait at most one day
    pool.map_async(execute_command, command_list).get(86400)


# Helper function for run_simulation that runs bash commands in separate processes
def execute_command(cmd):
    return subprocess.call(cmd, shell=True)


# Gather the output from all simulations and put it in the same file
def combine_output():

    print("Py: Combining output")

    if not (os.path.isdir(path_to_output)):
        quit('Py: Could not find output directory: ' + path_to_output)

    combined_results = []
    results_header = []

    for idx, param_set in enumerate(combined_iterable):
        results_file = os.path.join(path_to_output, 'sim', str(idx), 'results.csv')

        if os.path.isfile(results_file):
            with open(results_file, 'r') as local_results:
                # If the results_header is empty, append the first line of the local results file
                if not results_header:
                    results_header.append(local_results.readline().strip('\n'))
                else:
                    local_results.readline()

                # Store the results (second line of the results file) in the combined_results list
                combined_results.append(local_results.readline().strip('\n'))
        else:  # results file does not exist
            combined_results.append(str(idx) + ',' + 'simulation_incomplete')

    with open(os.path.join(path_to_output, 'combined_results.csv'), 'w') as combined_results_file:
        combined_results_file.write('\n'.join(results_header + combined_results))

    if os.path.getsize(os.path.join(path_to_output, 'combined_results.csv')) < 4:
        quit("Py: Combined results not generated as expected")


# Compress output and suffix with date run
def compress_output():

    print("Py: Compressed output to " + path_to_output + '_' + today + '.tar.gz')

    # Check that output directory exists
    if not (os.path.isdir(path_to_output)):
        raise Exception('Py: Could not find output directory: ' + path_to_output)

    # Change cwd to one above output path
    os.chdir(os.path.join(path_to_output, '..'))

    simulation_name = os.path.basename(os.path.normpath(path_to_output))

    # Compress entire output folder and append with the the simulations were started
    os.system('tar -zcf ' + simulation_name + '_' + today + '.tar.gz ' + simulation_name)


if __name__ == "__main__":
    main()
