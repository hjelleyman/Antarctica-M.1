"""Summary
"""
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats
import itertools
import glob
from modules.misc import seaice_area_mean
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import cm
import cartopy.crs as ccrs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid




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

def plot_all_indicies(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
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

def plot_all_indicies_sic(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
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
        n (int, optional): Description
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


################################################
#               Correlations                   #
################################################

############ Single Correlations ###############

def plot_single_correlations(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
    """Plots all the index timeseries.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
        indicies (list, optional): what indicies to load.
    """
    for  temp_res, temp_decomp, dt in itertools.product(temporal_resolution, temporal_decomposition, detrend):
        plot_single_correlation(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/correlations/single/', n = 1)

def plot_single_correlation(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/correlations/single/', n = 5):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/correlations/single/corr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    p_dataset = xr.open_dataset(f'processed_data/correlations/single/pval_{temp_decomp}_{temporal_resolution}_{detrend}_{n}.nc')
    p_values   = np.array([p_dataset[i].values for i in p_dataset])

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' indicies correlated with SIC'

    fig, ax = plt.subplots()
    significant = np.array(p_values) <= 0.05
    colors = np.array(['#177E89']*len(indicies))
    colors[~significant] = '#EA1B10'
    bar = plt.bar(indicies, values, color = colors)

    i = 0
    for rect in bar:
        height = rect.get_height()
        y = max(0,height)
        y_low = min(0,height)
        ax.text(rect.get_x() + rect.get_width()/2.0, y+0.01, f'corr = {height:.2f}', ha='center', va='bottom')
        ax.text(rect.get_x() + rect.get_width()/2.0, y_low-.21, f'pval = {p_values[i]:.2f}', ha='center', va='bottom')
        i+=1
    plt.title(title)
    # plt.legend(bbox_to_anchor=(0.99, -0.15), ncol = 2, loc = 'upper right')
    plt.ylim([-1,1])
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}' + '.pdf')
    plt.show()

############ Spatial Correlations ###############

def plot_spatial_correlations(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI']):
    """Plots all the index timeseries.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
        indicies (list, optional): what indicies to load.
    """
    for  temp_res, temp_decomp, dt in itertools.product(temporal_resolution, temporal_decomposition, detrend):
        plot_spatial_correlation(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/correlations/spatial/', n = 1)

def plot_spatial_correlation(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/correlations/spatial/', n = 5):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/correlations/spatial/corr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    p_dataset = xr.open_dataset(f'processed_data/correlations/spatial/pval_{temp_decomp}_{temporal_resolution}_{detrend}_{n}.nc')
    p_values   = np.array([p_dataset[i].values for i in p_dataset])

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' indicies correlated with SIC'
    cdict = {'red':   [[0.0,  234/255, 234/255],
                       [0.5,  253/255, 253/255],
                       [1.0,  23/255,  23/255]],
             'green': [[0.0,  27/255,  27/255],
                       [0.5,  231/255, 231/255],
                       [1.0,  126/255, 126/255]],
             'blue':  [[0.0,  16/255,  16/255],
                       [0.5,  76/255,  76/255],
                       [1.0,  1737255, 137/255]]}

    custom_colormap = LinearSegmentedColormap('BlueRed1', cdict)

    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, ax = plt.subplots(1,len(indicies),subplot_kw={'projection': ccrs.SouthPolarStereo()})
    for i in range(len(indicies)):
        contor = ax[i].contourf(dataset.x, dataset.y, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.SouthPolarStereo())
        ax[i].set_axis_off()
        ax[i].set_title(indicies[i])
        ax[i].coastlines()
    fig.suptitle(title)
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), shrink=0.95)
    # plt.colorbar(contor)
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}' + '.pdf')
    plt.show()
