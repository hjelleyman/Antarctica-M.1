import xarray as xr
import matplotlib.pyplot as plt
import itertools
import glob
from modules.misc import seaice_area_mean

################################################
#                Time series                   #
################################################


################### seaice #####################

def plot_all_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/SIC/'):
    for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
        plot_seaice_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, spatial_resolution = n, detrend = dt == 'detrended')

def plot_seaice_timeseries(anomlous = False, temporal_resolution = 'monthly', spatial_resolution = 1, detrend = False, imagefolder = 'images/timeseries/SIC/'):

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
    
    plt.plot(seaice.time, seaice[seaicename].mean(dim = ('x', 'y')))
    plt.title(title)
    plt.savefig(imagefolder + seaicename+'.pdf')
    plt.show()

################## indicies #####################

def plot_all_indicies(resolutions, temporal_resolution, temporal_decomposition, detrend, imagefolder = 'images/timeseries/INDICIES/', indicies = ['SAM', 'IPO', 'DMI']):
    for temp_res, temp_decomp, dt, indexname in itertools.product(temporal_resolution, temporal_decomposition, detrend, indicies):
        plot_index_timeseries(anomlous = 'anomalous' == temp_decomp, temporal_resolution = temp_res, detrend = dt == 'detrended', indexname = indexname)

def plot_index_timeseries(anomlous = False, temporal_resolution = 'monthly', detrend = False, imagefolder = 'images/timeseries/INDICIES/', indexname = 'SAM'):

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
    title = temp_decomp.capitalize() + ' '

    if detrend:
        title += dt + ' '

    title += temporal_resolution
    title += f' mean {indexname}'
    plt.plot(data.time, data)
    plt.title(title)
    plt.savefig(imagefolder + f'{indexname}_{filename}' + '.pdf')
    plt.show()