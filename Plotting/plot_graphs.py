import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

seq_times=[0.125729, 0.491132, 1.265142, 2.236244, 3.534881, 5.035440, 6.884574, 8.944648, 11.427148, 14.044076]
seq_times = [x * 1000 for x in seq_times]

def plot_graphs_initial_state():

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('values_comparison.csv')

    # Extract the 'n' and 'kernel_time' columns
    n1 = df[df['beta'] == 0.99]['n']
    kernel_time1 = df[df['beta'] == 0.99]['sec_per_iter'].reset_index(drop=True)


    n2 = df[df['beta'] == 0.9285]['n']
    kernel_time2 = df[df['beta'] == 0.9285]['sec_per_iter'].reset_index(drop=True)
    # Perform the subtraction
    diff2 = np.abs((kernel_time2 - kernel_time1) / kernel_time1)



    n3 = df[df['beta'] == 0.9]['n']
    kernel_time3 = df[df['beta'] == 0.9]['sec_per_iter'].reset_index(drop=True)
    diff3 = np.abs(kernel_time3 - kernel_time1)/kernel_time1

    # Create a new figure and plot the data
    plt.figure()
    plt.title('Relativno odstopanje od $\\alpha$=1.0 $\\beta$=0.99 $\\gamma$=0.001')

    # Plotting the graphs
    # plt.plot(n1, kernel_time1, 'r', label='alpha=1.0 beta=0.99 gamma=0.001', alpha=0.5)
    plt.plot(n2, diff2, 'b', label='$\\alpha$=1.0 $\\beta$=0.9285 $\\gamma$=0.001', alpha=0.5, ls='dashed',marker='.')
    plt.plot(n3, diff3, 'g', label='$\\alpha$=1.0 $\\beta$=0.9 $\\gamma$=0.001', alpha=0.5, ls='dashed',marker='.')

    plt.yscale('log')

    # alpha=1.
    # beta=0.9285
    # gamma=0.001

    # Adding labels and legend
    plt.xlabel('Velikost problema')
    plt.ylabel('Relativno odstopanje')
    plt.legend()

    # Displaying the plot
    plt.show()


def plot_graphs_shared_memory_different_block_sizes():

    df = pd.read_csv('shared_memory_different_block_sizes.csv')
    n1 = df[(df['beta'] == 0.99) & (df['block_size_x'] == 8)]['n']
    time1 = df[(df['beta'] == 0.99) & (df['block_size_x'] == 8)]['sec_per_iter'].reset_index(drop=True)
    time1 = [x / y for x, y in zip(seq_times, time1)]

    n2 = df[(df['beta'] == 0.99) & (df['block_size_x'] == 16)]['n']
    time2 = df[(df['beta'] == 0.99) & (df['block_size_x'] == 16)]['sec_per_iter'].reset_index(drop=True)
    time2 = [x / y for x, y in zip(seq_times, time2)]

    n3 = df[(df['beta'] == 0.99) & (df['block_size_x'] == 32)]['n']
    time3 = df[(df['beta'] == 0.99) & (df['block_size_x'] == 32)]['sec_per_iter'].reset_index(drop=True)
    time3 = [x / y for x, y in zip(seq_times, time3)]


    plt.figure()
    plt.plot(n1, time1, 'r', label='$\\alpha$=1.0 $\\beta$=0.99 $\\gamma$=0.001 block_size=8', alpha=0.5, ls='dashed',marker='.')
    plt.plot(n2, time2, 'b', label='$\\alpha$=1.0 $\\beta$=0.99 $\\gamma$=0.001 block_size=16', alpha=0.5, ls='dashed',marker='.')
    plt.plot(n3, time3, 'g', label='$\\alpha$=1.0 $\\beta$=0.99 $\\gamma$=0.001 block_size=32', alpha=0.5, ls='dashed',marker='.')
    

    plt.xlabel('Velikost problema')
    plt.ylabel('Pohitritev')
    plt.legend()
    plt.show()


def plot_graphs_wo_vs_w_shared_mem():
    
    df = pd.read_csv('shared_memory_different_block_sizes.csv')
    n1 = df[(df['block_size_x'] == 32)]['n']
    time1 = df[(df['block_size_x'] == 32)]['sec_per_iter'].reset_index(drop=True)
    time1 = [x / y for x, y in zip(seq_times, time1)]

    df1 = pd.read_csv('analysis_wo_shared_shape.csv')
    filtered = df1[(df1['block_size_x'] == 64) & (df1['block_size_y'] == 64)]
    n2 = filtered['n']
    time2 = filtered['sec_per_iter'].reset_index(drop=True)
    time2 = [x / y for x, y in zip(seq_times, time2)]

    plt.figure()
    plt.plot(n1, time1, 'r', label='$\\alpha$=1.0 $\\beta$=0.90 $\\gamma$=0.001 block_size=32 shared_memory', alpha=0.5, ls='dashed',marker='.')
    plt.plot(n2, time2, 'b', label='$\\alpha$=1.0 $\\beta$=0.90 $\\gamma$=0.001 block_size=64 without_shared_memory', alpha=0.5, ls='dashed',marker='.')
    
    plt.xlabel('Velikost problema')
    plt.ylabel('Pohitritev')
    plt.legend()
    plt.show()

def plot_graphs_block_row_col():
    df = pd.read_csv("analysis_wo_shared_shape.csv")
    blocks = df[(df['is_block'] == 1) & (df["n"] == 10000)]

    n1 = blocks['block_size_x'].reset_index(drop=True)
    time1 = blocks['sec_per_iter'].reset_index(drop=True)

    rows = df[(df['is_row'] == 1) & (df["n"] == 10000)]
    n2 = rows['block_size_y'].reset_index(drop=True)
    time2 = rows['sec_per_iter'].reset_index(drop=True)
    diff2 = np.abs((time2- time1) / time1)

    cols = df[(df['is_column'] == 1) & (df["n"] == 10000)]
    n3 = cols['block_size_x'].reset_index(drop=True)
    time3 = cols['sec_per_iter'].reset_index(drop=True)
    diff3 = np.abs((time3- time1) / time1)

    plt.figure()
    # plt.plot(n1, time1, 'r', label='blocks', ls='dashed',marker='.')
    # plt.plot(n2, time2, 'b', label='rows', ls='dashed',marker='.')
    plt.plot(n3, time3, 'g', label='cols', ls='dashed',marker='.')
    # plt.plot(n2, diff2, 'b', label='rows', alpha=0.5, ls='dashed',marker='.')
    # plt.plot(n3, diff3, 'g', label='cols', alpha=0.5, ls='dashed',marker='.')
    # plt.yscale('log')

    plt.xlabel('Širina')
    plt.ylabel('Čas na iteracijo (ms)')
    plt.legend()
    plt.show()

def plot_graphs_copy_time():
    df = pd.read_csv("analysis_wo_shared_shape.csv")
    filtered = df[(df['block_size_x'] == 64) & (df['is_block'] == 1)]
    n1 = filtered['n'].reset_index(drop=True)
    time1 = filtered['copy_time'].reset_index(drop=True)

    plt.figure()
    plt.plot(n1, time1, 'r', label='block_size=64, without_shared_memory', alpha=0.5, ls='dashed',marker='.')

    plt.xlabel('Velikost problema')
    plt.ylabel('Čas kopiranja (ms)')
    plt.legend()
    plt.show()

def plot_graphs_write_time():
    df = pd.read_csv("k-th_write.csv")
    k1 = df['k'].reset_index(drop=True)
    time1 = df['write_time'].reset_index(drop=True)

    plt.figure()
    plt.plot(k1, time1, 'r', label='Čas pisanja', ls='dashed',marker='.')
    
    plt.xlabel('k-to pisanje')
    plt.ylabel('Čas pisanja (ms)')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # plot_graphs_initial_state()
    # plot_graphs_shared_memory_different_block_sizes()
    # plot_graphs_wo_vs_w_shared_mem()
    # plot_graphs_block_row_col()
    # plot_graphs_copy_time()
    plot_graphs_write_time()