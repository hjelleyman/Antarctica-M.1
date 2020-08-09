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
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid




################################################
#                Time series                   #
################################################


################### seaice #####################

def plot_all_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    """Plots all of the timeseries for Antarctic SIC.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
    """
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        plot_seaice_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, spatial_resolution = n, detrend = dt == 'detrended',seaice_source=seaice_source)

def plot_seaice_timeseries(anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    """Plots a timeseries for Antarctic SIC.
    
    Args:
        anomlous (bool, optional): If the data is anomalous.
        temporal_resolution (str, optional): What temporal resolution to load.
        spatial_resolution (int, optional): What spatial resolution to load.
        detrend (bool, optional): Wheather to load detrended data or not.
        imagefolder (str, optional): Where to save output image.
    """
    output_folder = 'processed_data/SIC/'
    if seaice_source == 'ecmwf':
        output_folder = 'processed_data/ERA5/SIC/'

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

    if seaice_source == 'nsidc':
        seaice = seaice/1000
        mean_seaice = seaice_area_mean(seaice[seaicename],1)
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(mean_seaice.time.values.astype(float), mean_seaice)
    if seaice_source =='ecmwf':
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice[seaicename].time.values.astype(float), seaice[seaicename].sum(dim = ('longitude', 'latitude')))
    ax = plt.gca()
    if anomlous or detrend: ax.axhline(0, alpha = 0.5)
    if seaice_source == 'nsidc':
        mean_seaice = seaice_area_mean(seaice[seaicename],1)
        plt.plot(seaice.time, mean_seaice)

    if seaice_source == 'ecmwf':
        plt.plot(seaice.time, seaice[seaicename].mean(dim = ('longitude', 'latitude')))
    plt.plot(seaice.time, (seaice_m * seaice.time.values.astype(float) + seaice_b), color = '#177E89')
    plt.title(title)
    plt.savefig(imagefolder + seaicename+f'_{seaice_source}.pdf')
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

def plot_all_indicies_sic(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO'], seaice_source = 'nsidc'):
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
        plot_index_sic_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, detrend = dt == 'detrended', indexname = indexname, n = n, seaice_source = seaice_source)

def plot_index_sic_timeseries(anomlous = False, temporal_resolution = 'monthly', detrend = False, imagefolder = 'images/timeseries/SIC_INDICIES', indexname = 'SAM', n = 5, seaice_source = 'nsidc'):
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
    if seaice_source == 'ecmwf':
        seaice = xr.open_dataset(output_folder + 'ERA5/SIC/' + seaicename +'.nc')
    if seaice_source == 'ecmwf':
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice[seaicename].time.values.astype(float), seaice[seaicename].mean(dim = ('longitude', 'latitude')))
    if seaice_source == 'nsidc':
        mean_seaice = seaice_area_mean(seaice[seaicename],1)
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(mean_seaice.time.values.astype(float), mean_seaice)
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
    if seaice_source == 'ecmwf':
        ln2 = ax2.plot(seaice.time, seaice[seaicename].mean(dim = ('longitude', 'latitude')), label = 'SIC', color = '#177E89')
    if seaice_source == 'nsidc':
        ln2 = ax2.plot(mean_seaice.time, mean_seaice, label = 'SIC', color = '#177E89')
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
    plt.savefig(imagefolder + f'/SIC_{indexname}_{filename}_{seaice_source}' + '.pdf')
    plt.show()


############ SIC Comparison ###############

def plot_all_sic_sic(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
    """Plots all the index timeseries.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
        indicies (list, optional): what indicies to load.
    """
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        plot_sic_sic_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, spatial_resolution = n, detrend = dt == 'detrended')

def plot_sic_sic_timeseries(anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, imagefolder = 'images/timeseries/SIC/'):
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

    filename = f'{temp_decomp}_{temporal_resolution}_{spatial_resolution}_{dt}'
    indicies = xr.open_dataset(output_folder + 'ERA5/SIC/' + filename +'.nc')[filename].mean(dim = ('longitude', 'latitude'))
    data = indicies.copy()
    data = data.loc[data.time.dt.year >= 1979]
    seaicename = f'{temp_decomp}_{temporal_resolution}_{spatial_resolution}_{dt}'
    seaice = xr.open_dataset(output_folder + 'SIC/' + seaicename +'.nc')[seaicename]


    seaice = seaice_area_mean(seaice,1)
    seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice.time.values.astype(float), seaice)
    data_m, data_b, data_r_value, data_p_value, data_std_err = scipy.stats.linregress(data.time.values.astype(float), data)

    title = temp_decomp.capitalize() + ' '

    if detrend:
        title += dt + ' '

    title += temporal_resolution
    title += f' mean ERA5 and SIC'
    fig, ax = plt.subplots()
    ax2 = plt.twinx(ax)
    ax2.plot([],[])

    if anomlous or detrend: ax.axhline(0, alpha = 0.5)

    ln1 = ax.plot(data.time, data, label = f'ERA5', color = '#EA1B10')
    ax.plot(data.time, data_m * data.time.values.astype(float) + data_b, color = '#EA1B10')
    ln2 = ax2.plot(seaice.time, seaice, label = 'SIC', color = '#177E89')
    ax2.plot(seaice.time, seaice_m * seaice.time.values.astype(float) + seaice_b, color = '#177E89')

    if anomlous or detrend:
        yabs_max = abs(max(ax2.get_ylim(), key=abs))
        ax2.set_ylim(ymin=-yabs_max, ymax=yabs_max)

        yabs_max = abs(max(ax.get_ylim(), key=abs))
        ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)

    # ylabels
    ax.set_ylabel(f'ECMWF')
    ax2.set_ylabel(f'Mean SIC')

    # legend
    lines = ln1 + ln2
    labels = [line.get_label() for line in lines]
    plt.legend(lines,labels,bbox_to_anchor=(0.99, -0.15), ncol = 2, loc = 'upper right')

    plt.title(title)
    plt.savefig(imagefolder + f'/SIC_ERA5_{filename}' + '.pdf')
    plt.show()
################################################
#               Correlations                   #
################################################

############ Single Correlations ###############

def plot_single_correlations(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO'], seaice_source = 'nsidc'):
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
        plot_single_correlation(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/correlations/single/', n = 1, seaice_source = seaice_source)

def plot_single_correlation(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/correlations/single/', n = 5, seaice_source = 'nsidc'):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    if seaice_source == 'ecmwf':
        filename = f'processed_data/ERA5/correlations/single/corr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'
    else:
        filename = f'processed_data/correlations/single/corr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    if seaice_source == 'ecmwf':
        p_dataset = xr.open_dataset(f'processed_data/ERA5/correlations/single/pval_{temp_decomp}_{temporal_resolution}_{detrend}_{n}.nc')
    else:
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
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}_{seaice_source}' + '.pdf')
    plt.show()


def gen_single_correlation_table(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO'], seaice_source = 'nsidc'):
    n = 1
    correlations = pd.DataFrame(columns = indicies)
    pvalues = pd.DataFrame(columns = indicies)
    for temp_res, temp_decomp, dt in itertools.product(temporal_resolution, temporal_decomposition, detrend):
        filename = f'processed_data/correlations/single/corr_{temp_decomp}_{temp_res}_{dt}_{n}'

        dataset = xr.open_dataset(filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])

        index_name = f'{temp_decomp} {temp_res}'
        if dt == 'detrended': index_name += f' {dt}'

        correlations.loc[index_name,indicies] = values



        filename = f'processed_data/correlations/single/pval_{temp_decomp}_{temp_res}_{dt}_{n}'

        dataset = xr.open_dataset(filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])

        pvalues.loc[index_name,indicies] = values
    correlations.to_csv(f'images/correlations/single/correlations_{seaice_source}.csv')
    pvalues.to_csv(f'images/correlations/single/pvalues_{seaice_source}.csv')
    return correlations, pvalues


############ Spatial Correlations ###############

def plot_spatial_correlations(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI'], seaice_source='nsidc'):
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
        plot_spatial_correlation(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/correlations/spatial/', n = 1,seaice_source = seaice_source)

def plot_spatial_correlation(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/correlations/spatial/', n = 5, seaice_source='nsidc'):
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

    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, ax = plt.subplots(2,2,subplot_kw={'projection': ccrs.SouthPolarStereo()}, figsize = (5,5))
    ax = ax.flatten()
    for i in range(len(indicies)):
        if seaice_source == 'ecmwf':
            contor = ax[i].contourf(dataset.longitude, dataset.latitude, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.PlateCarree())
        if seaice_source == 'nsidc':
            contor = ax[i].contourf(dataset.x, dataset.y, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.SouthPolarStereo())

        ax[i].set_axis_off()
        ax[i].set_title(indicies[i])
        ax[i].coastlines()
    fig.suptitle(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
    cbar.set_label('Correlation')
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}_{seaice_source}' + '.pdf')
    plt.show()


def plot_spatial_correlations_with_significance(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI'], seaice_source = 'nsidc'):
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
        plot_spatial_correlation_with_significance(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/correlations/spatial/', n = 1, seaice_source = seaice_source)

def plot_spatial_correlation_with_significance(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/correlations/spatial/', n = 5, seaice_source = 'nsidc'):
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

    divnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    fig, ax = plt.subplots(2,2,subplot_kw={'projection': ccrs.SouthPolarStereo()}, figsize = (5,5))
    ax = ax.flatten()
    for i in range(len(indicies)):
        if seaice_source == 'ecmwf':
            contor = ax[i].contourf(dataset.longitude, dataset.latitude, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.PlateCarree())
            cs = ax[i].contourf(dataset.longitude, dataset.latitude, p_values[i], levels=[0,0.05,1], colors='none', hatches=[None, '//////'], transform = ccrs.PlateCarree())
        if seaice_source == 'nsidc':
            contor = ax[i].contourf(dataset.x, dataset.y, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.SouthPolarStereo())
            cs = ax[i].contourf(dataset.x, dataset.y, p_values[i], levels=[0,0.05,1], colors='none', hatches=[None, '//////'], transform = ccrs.SouthPolarStereo())
        # cs = ax[i].contour(dataset.x, dataset.y, p_values[i], levels=[0.05], colors='k', transform = ccrs.SouthPolarStereo(), linewidths=0.5, alpha = 1)
        ax[i].set_axis_off()
        ax[i].set_title(indicies[i])
        ax[i].coastlines()
    fig.suptitle(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
    cbar.set_label('Correlation')
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}_{seaice_source}' + '.pdf')
    plt.show()


################################################
#               Regressions                    #
################################################

############ Single Regressions ################

def plot_all_regression_scatter(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/regressions/single/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO'],seaice_source='nsidc'):
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
        plot_regression_scatter(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/regressions/single/', n = 1,seaice_source=seaice_source)

def plot_regression_scatter(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/regressions/single/', n = 5,seaice_source='nsidc'):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/regressions/single/regr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'



    seaicename = f'{temp_decomp}_{temporal_resolution}_{n}_{detrend}'
    seaice_data = xr.open_dataset('processed_data/SIC/' + seaicename +'.nc')
    seaice_data = seaice_data[seaicename]

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    b_dataset = xr.open_dataset(f'processed_data/regressions/single/bval_{temp_decomp}_{temporal_resolution}_{detrend}_{n}.nc')
    b_values  = np.array([b_dataset[i].values for i in b_dataset])


    index_data = []
    for indexname in indicies:
        filename = f'{indexname}_{temp_decomp}_{temporal_resolution}_{detrend}'
        index_data += [xr.open_dataset('processed_data/INDICIES/' + filename +'.nc')[indexname]]

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' indicies correlated with SIC'

    fig, ax = plt.subplots(2,2, figsize = (5,6))
    ax = ax.flatten()

    times = list(set.intersection(set(seaice_data.time.values), *(set(index_data[i].time.values)for i in range(len(indicies)))))

    seaice_data = seaice_area_mean(seaice_data.sel(time=times).sortby('time'), 1)
    index_data = [ind.sel(time=times).sortby('time') for ind in index_data]

    for ind in index_data:
        ind = (ind - ind.mean()) 
        ind =  ind / ind.std()

    # seaice_data = (seaice_data - seaice_data.mean()) 
    # seaice_data =  seaice_data / seaice_data.std()
    # Plotting
    for i in range(len(indicies)):
        xlength  = 1.25 * max(-min(index_data[i]),max(index_data[i]))
        ylength  = 1.25 * max(-min(seaice_data),max(seaice_data))
        ax[i].scatter(index_data[i], seaice_data, c = seaice_data.time)
        ax[i].axhline(0, alpha = 0.5)
        ax[i].axvline(0, alpha = 0.5)
        ax[i].set_xlim(-xlength,xlength)
        ax[i].set_ylim(-ylength,ylength)
        ax[i].set_xlabel(indicies[i])

        yfit = values[i] * np.array([-xlength,xlength]) + b_values[i]
        ax[i].plot(np.array([-xlength,xlength]), yfit, color = 'black')
    fig.suptitle(title)
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}_{seaice_source}' + '.pdf')
    plt.show()

def gen_single_regression_table(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
    n = 1
    regressions = pd.DataFrame(columns = indicies)
    pvalues     = pd.DataFrame(columns = indicies)
    rvalues     = pd.DataFrame(columns = indicies)
    bvalues     = pd.DataFrame(columns = indicies)
    stderr      = pd.DataFrame(columns = indicies)
    for temp_res, temp_decomp, dt in itertools.product(temporal_resolution, temporal_decomposition, detrend):
            
        folder = 'processed_data/regressions/single/'
        filename = f'{temp_decomp}_{temp_res}_{dt}_{n}'

        dataset = xr.open_dataset(folder + 'regr_' + filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])
        index_name = f'{temp_decomp} {temp_res}'
        if dt == 'detrended': index_name += f' {dt}'
        regressions.loc[index_name,indicies] = values

        dataset = xr.open_dataset(folder + 'pval_' + filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])
        index_name = f'{temp_decomp} {temp_res}'
        if dt == 'detrended': index_name += f' {dt}'
        pvalues.loc[index_name,indicies] = values

        dataset = xr.open_dataset(folder + 'rval_' + filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])
        index_name = f'{temp_decomp} {temp_res}'
        if dt == 'detrended': index_name += f' {dt}'
        rvalues.loc[index_name,indicies] = values

        dataset = xr.open_dataset(folder + 'bval_' + filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])
        index_name = f'{temp_decomp} {temp_res}'
        if dt == 'detrended': index_name += f' {dt}'
        bvalues.loc[index_name,indicies] = values

        dataset = xr.open_dataset(folder + 'std_err_' + filename + '.nc')
        indicies = np.array([i for i in dataset])
        values   = np.array([dataset[i].values for i in dataset])
        index_name = f'{temp_decomp} {temp_res}'
        if dt == 'detrended': index_name += f' {dt}'
        stderr.loc[index_name,indicies] = values

    regressions.to_csv('images/regressions/single/regressions.csv')
    pvalues.to_csv('images/regressions/single/pvalues.csv')
    rvalues.to_csv('images/regressions/single/rvalues.csv')
    bvalues.to_csv('images/regressions/single/bvalues.csv')
    stderr.to_csv('images/regressions/single/stderr.csv')
    return regressions, pvalues, rvalues, bvalues, stderr


############ Single Spatial Regressions ################

def plot_all_regression_spatial(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/regressions/spatial/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
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
        plot_regression_spatial(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/regressions/spatial/', n = 1)

def plot_regression_spatial(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/regressions/spatial/', n = 5):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/regressions/spatial/regr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'

    dataset = xr.open_dataset(filename + '.nc')
    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    dataset = dataset*area
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' SIC regressed against'

    max_ = max([max(value.max(),-value.min()) for value in  values])
    # max_ = 2
    divnorm = TwoSlopeNorm(vmin=-max_, vcenter=0, vmax=max_)
    fig, ax = plt.subplots(2,2,subplot_kw={'projection': ccrs.SouthPolarStereo()}, figsize = (5,5))
    ax = ax.flatten()

    # Plotting
    for i in range(len(indicies)):
        contor = ax[i].contourf(dataset.x, dataset.y, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.SouthPolarStereo())
        ax[i].coastlines()
        ax[i].set_axis_off()
        ax[i].set_title(indicies[i])  
    fig.suptitle(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
    cbar.set_label('Correlation')
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}' + '.pdf')
    plt.show()


############ Multiple Regressions ################


def plot_all_regression_multiple_scatter(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/regressions/single_multiple/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
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
        plot_regression_multiple_scatter(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/regressions/single_multiple/', n = 1)

def plot_regression_multiple_scatter(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/regressions/single_multiple/', n = 5):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/regressions/single_multiple/regr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'



    seaicename = f'{temp_decomp}_{temporal_resolution}_{n}_{detrend}'
    seaice_data = xr.open_dataset('processed_data/SIC/' + seaicename +'.nc')
    seaice_data = seaice_data[seaicename]
    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    seaice_data = seaice_data * area

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    b_dataset = xr.open_dataset(f'processed_data/regressions/single_multiple/bval_{temp_decomp}_{temporal_resolution}_{detrend}_{n}.nc')
    b_values  = np.array([b_dataset[i].values for i in b_dataset])


    index_data = []
    for indexname in indicies:
        filename = f'{indexname}_{temp_decomp}_{temporal_resolution}_{detrend}'
        index_data += [xr.open_dataset('processed_data/INDICIES/' + filename +'.nc')[indexname]]

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' indicies correlated with SIC'

    fig, ax = plt.subplots(2,2, figsize = (5,6))
    ax = ax.flatten()

    times = list(set.intersection(set(seaice_data.time.values), *(set(index_data[i].time.values)for i in range(len(indicies)))))

    seaice_data = seaice_area_mean(seaice_data.sel(time=times).sortby('time'), 1)
    index_data = [ind.sel(time=times).sortby('time') for ind in index_data]

    for ind in index_data:
        ind = (ind - ind.mean()) 
        ind =  ind / ind.std()

    # seaice_data = (seaice_data - seaice_data.mean()) 
    # seaice_data =  seaice_data / seaice_data.std()
    # Plotting
    for i in range(len(indicies)):
        xlength  = 1.25 * max(-min(index_data[i]),max(index_data[i]))
        ylength  = 1.25 * max(-min(seaice_data),max(seaice_data))
        ax[i].scatter(index_data[i], seaice_data, c = seaice_data.time)
        ax[i].axhline(0, alpha = 0.5)
        ax[i].axvline(0, alpha = 0.5)
        ax[i].set_xlim(-xlength,xlength)
        ax[i].set_ylim(-ylength,ylength)
        ax[i].set_xlabel(indicies[i])

        yfit = values[i] * np.array([-xlength,xlength]) + b_values[i]
        ax[i].plot(np.array([-xlength,xlength]), yfit, color = 'black')
    fig.suptitle(title)
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}_{seaice_source}' + '.pdf')
    plt.show()

############ Multiple Spatial Regressions ################

def plot_all_regression_multiple_spatial(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/regressions/spatial_multiple/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
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
        plot_regression_multiple_spatial(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/regressions/spatial_multiple/', n = 1)

def plot_regression_multiple_spatial(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/regressions/spatial_multiple/', n = 5):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/regressions/spatial_multiple/regr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])

    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    dataset = dataset * area

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' SIC regressed against indicies'

    max_ = max([max(value.max(),-value.min()) for value in  values])
    # max_ = 1
    divnorm = TwoSlopeNorm(vmin=-max_, vcenter=0, vmax=max_)
    fig, ax = plt.subplots(1,5,subplot_kw={'projection': ccrs.SouthPolarStereo()}, figsize = (2.5*5,3))
    ax = ax.flatten()

    # Plotting
    for i in range(len(indicies)):
        contor = ax[i].contourf(dataset.x, dataset.y, values[i], cmap = 'RdBu', norm = divnorm, transform=ccrs.SouthPolarStereo())
        ax[i].coastlines()
        ax[i].set_axis_off()
        ax[i].set_title(indicies[i])  
    fig.suptitle(title)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
    cbar.set_label('Regression coefficients')
    plt.savefig(imagefolder + f'{temp_decomp}_{temporal_resolution}_{detrend}_{n}' + '.pdf')
    plt.show()

############ Multiple Spatial Contribution Regressions ################

def plot_all_regression_multiple_contribution_spatial(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI', 'ENSO']):
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
        plot_regression_multiple_contribution_spatial(anomlous = temp_decomp, temporal_resolution = temp_res, detrend = dt, temp_decomp = temp_decomp, imagefolder = 'images/correlations/single/', n = 1)

def plot_regression_multiple_contribution_spatial(anomlous = False, temporal_resolution = 'monthly', detrend = 'raw', temp_decomp = 'anomalous', imagefolder = 'images/correlations/spatial/', n = 5):
    """Summary
    
    Args:
        anomlous (bool, optional): Description
        temporal_resolution (str, optional): Description
        detrend (bool, optional): Description
        imagefolder (str, optional): Description
        n (int, optional): Description
    """
    filename = f'processed_data/regressions/spatial_multiple/regr_{temp_decomp}_{temporal_resolution}_{detrend}_{n}'

    dataset = xr.open_dataset(filename + '.nc')
    indicies = np.array([i for i in dataset])
    values   = np.array([dataset[i].values for i in dataset])


    # area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    # dataset = dataset * area

    index_data = {}
    for indexname in indicies[:-1]:
        filename = f'{indexname}_{temp_decomp}_{temporal_resolution}_{detrend}'
        index_data[indexname] = xr.open_dataset('processed_data/INDICIES/' + filename +'.nc')[indexname]
        index_data[indexname] = (index_data[indexname] - index_data[indexname].mean()) 
        index_data[indexname] =  index_data[indexname] / index_data[indexname].std()
        
    newdata = {} 
    for indexname in indicies[:-1]:
        a = scipy.stats.linregress(index_data[indexname].time.values.astype(float), index_data[indexname])
        newdata[indexname] = a[0] * dataset[indexname] * 24*60*60*365e9

    title = temp_decomp.capitalize() + ' '
    if detrend == 'detrended':
        title += detrend + ' '

    title += temporal_resolution
    title += f' SIC trend contributions'

    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    # Plotting
    for i in range(len(indicies)-1):
        indexname = indicies[i]
        newdata[indexname] = newdata[indexname] * area / 250
        newdata[indexname] = newdata[indexname].where(newdata[indexname] !=0)

    max_ = max([max(newdata[indexname].max(),-newdata[indexname].min()) for indexname in indicies[:-1]])
    # max_ = 1
    divnorm = TwoSlopeNorm(vmin=-max_, vcenter=0, vmax=max_)
    fig, ax = plt.subplots(2,2,subplot_kw={'projection': ccrs.SouthPolarStereo()}, figsize = (5,5))
    ax = ax.flatten()

    # Plotting
    for i in range(len(indicies)-1):
        indexname = indicies[i]
        contor = ax[i].contourf(dataset.x, dataset.y, newdata[indexname], cmap = 'RdBu', norm = divnorm, transform=ccrs.SouthPolarStereo())
        ax[i].coastlines()
        ax[i].set_axis_off()
        ax[i].set_title(indicies[i])  
    fig.suptitle(title)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(cm.ScalarMappable(norm=divnorm, cmap='RdBu'), cax=cbar_ax, shrink=0.88)
    cbar.set_label('Trend contributions (km$^2$ yr$^{-1}$)')
    plt.show()

############ SIE Trends ################

def plot_all_seaice_trends(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    """Plots all of the timeseries for Antarctic SIC.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
    """
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        plot_seaice_trend(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, spatial_resolution = n, detrend = dt == 'detrended',seaice_source=seaice_source)

def plot_seaice_trend(anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    """Plots a timeseries for Antarctic SIC.
    
    Args:
        anomlous (bool, optional): If the data is anomalous.
        temporal_resolution (str, optional): What temporal resolution to load.
        spatial_resolution (int, optional): What spatial resolution to load.
        detrend (bool, optional): Wheather to load detrended data or not.
        imagefolder (str, optional): Where to save output image.
    """
    output_folder = 'processed_data/SIC/'
    if seaice_source == 'ecmwf':
        output_folder = 'processed_data/ERA5/SIC/'

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
    title += ' SIE trends'


    seaicename = f'{temp_decomp}_{temporal_resolution}_{spatial_resolution}_{dt}'
    seaice = xr.open_dataset(output_folder + seaicename +'.nc')

    if seaice_source == 'nsidc':
        seaice = seaice/250
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = xr.apply_ufunc(scipy.stats.linregress, seaice[seaicename].time.values.astype(float), seaice[seaicename], input_core_dims=[['time'],['time']], vectorize=True, dask='parallelized', output_dtypes=[float]*5, output_core_dims=[[]]*5)
    if seaice_source =='ecmwf':
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice[seaicename].time.values.astype(float), seaice[seaicename])
    
    seaice_m = seaice_m * 1e9 * 60 * 60 * 24 * 365
    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    seaice_m = seaice_m*area
    seaice_m = seaice_m.where(seaice_m != 0)
    # seaice_m = seaice_m.where(seaice_p_value <= 0.05)
    max_ = max(seaice_m.max(),-seaice_m.min())
    # max_ = 1
    divnorm = TwoSlopeNorm(vmin=-max_, vcenter=0, vmax=max_)
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111, projection = ccrs.SouthPolarStereo())
    # Plotting
    contor = ax.contourf(seaice_m.x, seaice_m.y, seaice_m, cmap = 'RdBu', levels = 100, norm = divnorm, transform=ccrs.SouthPolarStereo())
    ax.coastlines()
    ax.set_axis_off()
    cbar = plt.colorbar(contor)
    cbar.set_label('Trend in SIE (km$^2$ yr$^{-1}$)')
    plt.title(title)
    plt.show()


def plot_all_subplot_trends(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    """Plots all of the timeseries for Antarctic SIC.
    
    Args:
        resolutions (list of int): What spatial resolutions to load.
        temporal_resolution (list of str): What temporal resolutions to load.
        temporal_decomposition (list of str): Anomalous or not.
        detrend (list of bool): detrended or not.
        imagefolder (str, optional): Folder to save output images to.
    """
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        plot_seaice_trend(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, spatial_resolution = n, detrend = dt == 'detrended',seaice_source=seaice_source)

def plot_subplot_trend(anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, imagefolder = 'images/timeseries/SIC/',seaice_source='nsidc'):
    """Plots a timeseries for Antarctic SIC.
    
    Args:
        anomlous (bool, optional): If the data is anomalous.
        temporal_resolution (str, optional): What temporal resolution to load.
        spatial_resolution (int, optional): What spatial resolution to load.
        detrend (bool, optional): Wheather to load detrended data or not.
        imagefolder (str, optional): Where to save output image.
    """
    output_folder = 'processed_data/SIC/'
    if seaice_source == 'ecmwf':
        output_folder = 'processed_data/ERA5/SIC/'

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
    title += ' SIE trends'


    seaicename = f'{temp_decomp}_{temporal_resolution}_{spatial_resolution}_{dt}'
    seaice = xr.open_dataset(output_folder + seaicename +'.nc')

    if seaice_source == 'nsidc':
        seaice = seaice/250
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = xr.apply_ufunc(scipy.stats.linregress, seaice[seaicename].time.values.astype(float), seaice[seaicename], input_core_dims=[['time'],['time']], vectorize=True, dask='parallelized', output_dtypes=[float]*5, output_core_dims=[[]]*5)
    if seaice_source =='ecmwf':
        seaice_m, seaice_b, seaice_r_value, seaice_p_value, seaice_std_err = scipy.stats.linregress(seaice[seaicename].time.values.astype(float), seaice[seaicename])
    
    seaice_m = seaice_m * 1e9 * 60 * 60 * 24 * 365
    area = xr.open_dataset('data/area_files/processed_nsidc.nc').area
    seaice_m = seaice_m*area
    seaice_m = seaice_m.where(seaice_m != 0)
    # seaice_m = seaice_m.where(seaice_p_value <= 0.05)
    max_ = max(seaice_m.max(),-seaice_m.min())
    # max_ = 1
    divnorm = TwoSlopeNorm(vmin=-max_, vcenter=0, vmax=max_)
    fig = plt.figure(projection = ccrs.SouthPolarStereo(), figsize = (5,5))
    ax = fig.add_subplot(111)

    # Plotting
    contor = ax.contourf(seaice_m.x, seaice_m.y, seaice_m, cmap = 'RdBu', levels = 100, norm = divnorm, transform=ccrs.SouthPolarStereo())
    ax.coastlines()
    ax.set_axis_off()
    cbar = plt.colorbar(contor)
    cbar.set_label('Trend in SIE (km$^2$ yr$^{-1}$)')
    plt.title(title)
    plt.show()