import multiprocessing
import os
import subprocess
import time

import numpy as np
import matplotlib.pyplot as plt


# Globally accessible directory paths, names, and variables
chaste_build_dir = os.environ.get('CHASTE_BUILD_DIR')
executable = os.path.join(chaste_build_dir, 'projects/IbNumericsPaper/apps', 'Exe_ConvergenceWithTimestep')

if not(os.path.isfile(executable)):
    raise Exception('Py: Could not find executable: ' + executable)

chaste_test_dir = os.environ.get('CHASTE_TEST_OUTPUT')
path_to_output = os.path.join(chaste_test_dir, 'convergence', 'time_step')

pdf_name = os.path.join(path_to_output, 'ConvergenceWithTimeStep')
results_header_string = 'simulation_id,time_step,esf_at_end\n'

# Range for simulation parameters
num_sims = 19


def main():
    run_simulations()
    combine_output()
    plot_results()
    compress_output()


# Create a list of commands and pass them to separate processes
def run_simulations():

    # 2^-2 down to 2^-11 at half-integer powers, rounded to integer value
    param_values = 2**(np.linspace(-2, -11, num=num_sims))

    # Make a list of calls to a Chaste executable
    command_list = []

    for idx, param_val in enumerate(param_values):

        command = 'nice -n 19 ' + executable \
                  + ' --ID ' + str(idx) \
                  + ' --DT ' + str(param_val)
        command_list.append(command)

    # Use processes equal to the number of cpus available
    count = multiprocessing.cpu_count()

    print("Py: Starting simulations with " + str(count) + " processes")

    # Generate a pool of workers
    pool = multiprocessing.Pool(processes=count)

    # Pass the list of bash commands to the pool

    pool.map(execute_command, reversed(command_list))


# This is a helper function for run_simulation that runs bash commands in separate processes
def execute_command(cmd):
    return subprocess.call(cmd, shell=True)


# Gather the output from all simulations and put it in the same file
def combine_output():

    print("Py: Combining output")

    if not(os.path.isdir(path_to_output)):
        raise Exception('Py: Could not find output directory: ' + path_to_output)

    combined_results = open(os.path.join(path_to_output, 'combined_results.dat'), 'w')
    combined_results.write(results_header_string)

    for sim_id in range(num_sims):
        file_name = os.path.join(path_to_output, 'sim', str(sim_id), 'results.dat')

        if os.path.isfile(file_name):
            local_results = open(file_name, 'r')
            combined_results.write(local_results.readline())
            local_results.close()
            combined_results.write("\n")

    combined_results.close()


def plot_results():

    print("Py: Plotting results")

    ################################################################
    # Global options to ensure consistency between different plots #
    ################################################################

    bg_gray = '0.75'
    bg_line_width = 0.5
    font_size = 8

    mycol_gr = '#006532' #Green
    mycol_or = '#F39200' #Orange
    mycol_bl = '#2C2E83' #Blue
    mycol_lb = '#0072bd' #Light blue

    # SIAM max fig width = 31pi = 13.12cm = 5.17in

    # latex line width in inches
    latex_line_width = 5.126

    # Potentially useful pica conversions
    pica_to_cm = 127.0/300.0
    pica_to_in = 1.0/6.0

    golden_rat = 1.618

    fig_width = 0.45 * latex_line_width

    # Set LaTeX font rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', serif='computer modern roman')
    plt.rc('figure', figsize=[fig_width, fig_width / golden_rat])
    plt.rc('axes', linewidth=bg_line_width, edgecolor=bg_gray, axisbelow=True)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('xtick.major', size=0, pad=4)
    plt.rc('ytick.major', size=0, pad=4)

    # Data for esf vs time plot
    esf_data = np.genfromtxt(os.path.join(path_to_output, 'combined_results.dat'),
                             delimiter=',',
                             skip_header=1)

    time_step = -1.0 * np.log2(esf_data[:, 1])
    esf_at_end = esf_data[:, 2]

    # Plot the main graph
    # Calculate limits to give a comfy margin
    x_min, x_max = min(time_step), max(time_step)
    y_min, y_max = min(esf_at_end), max(esf_at_end)

    x_range, y_range = x_max - x_min, y_max - y_min
    margin = 0.05

    x_lims = [x_min - margin * x_range, x_max + margin * x_range]
    y_lims = [y_min - margin * y_range, y_max + margin * y_range]

    # Plot the two continuous lines
    plt.plot(time_step, esf_at_end, marker='o',
             linestyle='-',
             linewidth=0.75,
             color=mycol_or,
             markerfacecolor=mycol_lb,
             markeredgecolor=mycol_or,
             markersize=4.0,
             markeredgewidth=0.75)

    # Set the axis limits
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    # Label the axes
    plt.xlabel(r'$-\log_2(\Delta t)$', fontsize=font_size, labelpad=1.0*font_size)
    plt.ylabel(r'ESF', fontsize=font_size, labelpad=1.0*font_size)

    # Customise axes
    ax = plt.gca()
    ax.grid(b=True, which='major', color=bg_gray, linestyle='dotted', dash_capstyle='round')

    # Export figure
    plt.savefig(pdf_name + '.pdf', bbox_inches='tight', pad_inches=0.0)

    best_esf = esf_at_end[-1]
    time_step = time_step[0:-2]
    esf_at_end = esf_at_end[0:-2]
    log_error = -np.log2(np.fabs(esf_at_end - best_esf))

    # Calculate the linear fit and export to file
    linear_fit = np.polyfit(time_step, log_error, 1)
    np.savetxt(os.path.join(path_to_output, 'linear_fit.dat'), linear_fit, delimiter=',')

    # Plot the log-log graph
    # Calculate limits to give a comfy margin
    x_min, x_max = min(time_step), max(time_step)
    y_min, y_max = min(log_error), max(log_error)

    x_range, y_range = x_max - x_min, y_max - y_min
    margin = 0.05

    x_lims = [x_min - margin * x_range, x_max + margin * x_range]
    y_lims = [y_min - margin * y_range, y_max + margin * y_range]

    linear_x = np.linspace(x_min, x_max, 100)
    linear_y = linear_fit[1] + linear_fit[0] * linear_x

    # Plot the linear fit, then the data
    plt.plot(linear_x, linear_y,
             linewidth=0.75,
             color=mycol_or)

    plt.plot(time_step, log_error, marker='o',
             linewidth=0,
             markerfacecolor=mycol_lb,
             markeredgecolor=mycol_or,
             markersize=4.0,
             markeredgewidth=0.75)

    # Set the axis limits
    plt.xlim(x_lims)
    plt.ylim(y_lims)

    # Label the axes
    plt.xlabel(r'$-\log_2(\Delta t)$', fontsize=font_size, labelpad=1.0*font_size)
    plt.ylabel(r'$-\log_2(|$error$|)$', fontsize=font_size, labelpad=1.0*font_size)

    # Customise axes
    ax = plt.gca()
    ax.grid(b=True, which='major', color=bg_gray, linestyle='dotted', dash_capstyle='round')

    # Export figure
    plt.savefig(pdf_name + '_loglog.pdf', bbox_inches='tight', pad_inches=0.0)


# Compress output and suffix with date run
def compress_output():

    # Check that output directory exists
    if not (os.path.isdir(path_to_output)):
        raise Exception('Py: Could not find output directory: ' + path_to_output)

    # Change cwd to one above output path
    os.chdir(os.path.join(path_to_output, '..'))

    simulation_name = os.path.basename(os.path.normpath(path_to_output))

    today = time.strftime('%Y-%m-%d')

    # Compress entire output folder and append with the the simulations were started
    os.system('tar -zcf ' + simulation_name + '_' + today + '.tar.gz ' + simulation_name)

if __name__ == "__main__":
    main()
