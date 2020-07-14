"""Summary
"""
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats
import itertools
import glob
from modules.misc import seaice_area_mean
import numpy as np


################################################
#                Time series                   #
################################################


################### seaice #####################

def plot_all_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/SIC/'):
    """Plots all of the timeseries for Antarctic SIC.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
    """
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        plot_seaice_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, spatial_resolution = n, detrend = dt == 'detrended')

def plot_seaice_timeseries(anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, imagefolder = 'images/timeseries/SIC/'):
    """Plots a timeseries for Antarctic SIC.
    
    Args:
        anomlous (bool, optional): If the data is anomalous.
        temporal_resolution (str, optional): What temporal resolution to load.
        spatial_resolution (int, optional): What spatial resolution to load.
        detrend (bool, optional): Wheather to load detrended data or not.
        imagefolder (str, optional): Where to save output image.
    """
    output_folder = 'processed_data/SIC/'

    if anomlous:
        temp_decomp = 'anomalous'
    else:
        temp_decomp = 'raw'


    title = temp_decomp.capitalize() + ' '

    if detrend:
        dt = 'detrended'
        title += dt + ' '
    else:
        dt = 'raw'

    title += temporal_resolution
    title += ' mean SIC in Antarctica'


    seaicename = f'{temp_decomp}_{temporal_resolution}_{spatial_resolution}_{dt}'
    seaice = xr.open_dataset(output_folder + seaicename +'.nc')

    seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice[seaicename].time.values.astype(float), seaice[seaicename].mean(dim = ('x', 'y')))
    ax = plt.gca()
    if anomlous or detrend: ax.axhline(0, alpha = 0.5)
    plt.plot(seaice.time, seaice[seaicename].mean(dim = ('x', 'y')))
    plt.plot(seaice.time, seaice_m * seaice.time.values.astype(float) + seaice_b, color = '#177E89')
    plt.title(title)
    plt.savefig(imagefolder + seaicename+'.pdf')
    plt.show()

################## indicies #####################

def plot_all_indicies(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI']):
    """Plots all the index timeseries.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
        indicies (list, optional): what indicies to load.
    """
    for temp_res, temp_decomp, dt, indexname in itertools.product(temporal_resolution, temporal_decomposition, detrend, indicies):
        plot_index_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, detrend = dt == 'detrended', indexname = indexname)

def plot_index_timeseries(anomlous = False, temporal_resolution = 'monthly', detrend = False, imagefolder = 'images/timeseries/INDICIES/', indexname = 'SAM'):
    """Plots an index time series.
    
    Args:
        anomlous (bool, optional): If the data is anomalous.
        temporal_resolution (str, optional): What temporal resolution to load.
        detrend (bool, optional): Wheather to load detrended data or not.
        imagefolder (str, optional): Where to save output image.
        indexname (str, optional): Which index to load and plot.
    """
    output_folder = 'processed_data/INDICIES/'


    if anomlous:
        temp_decomp = 'anomalous'
    else:
        temp_decomp = 'raw'

    if detrend:
        dt = 'detrended'
    else:
        dt = 'raw'

    filename = f'{indexname}_{temp_decomp}_{temporal_resolution}_{dt}'
    indicies = xr.open_dataset(output_folder + filename +'.nc')[indexname]
    data = indicies.copy()
    data = data.loc[data.time.dt.year >= 1979]

    data_m, data_b, data_r_value, data_p_value, data_std_err = scipy.stats.linregress(data.time.values.astype(float), data)

    title = temp_decomp.capitalize() + ' '

    if detrend:
        title += dt + ' '

    title += temporal_resolution
    title += f' mean {indexname}'
    ax = plt.gca()
    if anomlous or detrend: ax.axhline(0, alpha = 0.5)
    plt.plot(data.time, data)
    plt.plot(data.time, data_m * data.time.values.astype(float) + data_b, color = '#177E89')
    if anomlous or detrend:
        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    plt.title(title)
    plt.savefig(imagefolder + f'{indexname}_{filename}' + '.pdf')
    plt.show()

################## indicies and sic #####################

def plot_all_indicies_sic(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI']):
    """Plots all the index timeseries.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
        indicies (list, optional): what indicies to load.
    """
    for n, temp_res, temp_decomp, dt, indexname in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend, indicies):
        plot_index_sic_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, detrend = dt == 'detrended', indexname = indexname, n = n)

def plot_index_sic_timeseries(anomlous = False, temporal_resolution = 'monthly', detrend = False, imagefolder = 'images/timeseries/SIC_INDICIES', indexname = 'SAM', n = 5):
    """Plots an index time series.
    
    Args:
        anomlous (bool, optional): If the data is anomalous.
        temporal_resolution (str, optional): What temporal resolution to load.
        detrend (bool, optional): Wheather to load detrended data or not.
        imagefolder (str, optional): Where to save output image.
        indexname (str, optional): Which index to load and plot.
    """
    output_folder = 'processed_data/'


    if anomlous:
        temp_decomp = 'anomalous'
    else:
        temp_decomp = 'raw'

    if detrend:
        dt = 'detrended'
    else:
        dt = 'raw'

    filename = f'{indexname}_{temp_decomp}_{temporal_resolution}_{dt}'
    indicies = xr.open_dataset(output_folder + 'INDICIES/' + filename +'.nc')[indexname]
    data = indicies.copy()
    data = data.loc[data.time.dt.year >= 1979]
    seaicename = f'{temp_decomp}_{temporal_resolution}_{n}_{dt}'
    seaice = xr.open_dataset(output_folder + 'SIC/' + seaicename +'.nc')

    seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice[seaicename].time.values.astype(float), seaice[seaicename].mean(dim = ('x', 'y')))
    data_m, data_b, data_r_value, data_p_value, data_std_err = scipy.stats.linregress(data.time.values.astype(float), data)

    title = temp_decomp.capitalize() + ' '

    if detrend:
        title += dt + ' '

    title += temporal_resolution
    title += f' mean {indexname} and SIC'
    fig, ax = plt.subplots()
    ax2 = plt.twinx(ax)
    ax2.plot([],[])

    if anomlous or detrend: ax.axhline(0, alpha = 0.5)

    ln1 = ax.plot(data.time, data, label = f'{indexname}', color = '#EA1B10')
    ax.plot(data.time, data_m * data.time.values.astype(float) + data_b, color = '#EA1B10')
    ln2 = ax2.plot(seaice.time, seaice[seaicename].mean(dim = ('x', 'y')), label = 'SIC', color = '#177E89')
    ax2.plot(seaice.time, seaice_m * seaice.time.values.astype(float) + seaice_b, color = '#177E89')

    yabs_max = abs(max(ax.get_ylim(), key=abs))
    ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
    if anomlous or detrend:
        yabs_max = abs(max(ax2.get_ylim(), key=abs))
        ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    # ylabels
    ax.set_ylabel(f'{indexname}')
    ax2.set_ylabel(f'Mean SIC')

    # legend
    lines = ln1 + ln2
    labels = [line.get_label() for line in lines]
    plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 2, loc = 'upper right')

    plt.title(title)
    plt.savefig(imagefolder + f'/SIC_{indexname}_{filename}' + '.pdf')
    plt.show()
