import os

import numpy as np
import matplotlib as mlt
import matplotlib.pyplot as plt

def checkFolder(folder):
    """
    Check whether folder is present and creates them if necessary
    """

    if not os.path.exists(folder):
        print(f'Making directory {folder}')
        os.makedirs(folder)

def axisEqual3D(ax):
    '''
    Set equal limits to the axis of a 3D matplotlib plot
    https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
    '''

    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def newFigure(height=4, half=False, target='thesis', dpi=300):
    '''
    Create a new figure, used to keep figure and font sizes consistent
    Figures need to be saved with dpi=300
    Calling plt.tight_layout() before saving is advised
    '''

    # matplotlib settings
    # mlt.rcParams['font.family'] = ['serif']
    # mlt.rcParams['font.serif'] = ['Times New Roman']
    plt.rc('font', family='Times New Roman')

    if target == 'thesis':
        SMALL_SIZE = 7
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 8
    elif target == 'paper':
        SMALL_SIZE = 7
        MEDIUM_SIZE = 8
        BIGGER_SIZE = 8

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    if target == 'thesis':
        width = 6.2
    elif target == 'paper':
        width = 6.5
    else:
        print('ERRRRRRRoRRrr')

    if half == True:
        width = width/2

    fig = plt.figure(figsize=(width, height), dpi=dpi)

    return fig
