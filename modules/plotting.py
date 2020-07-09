import xarray as xr
import matplotlib.pyplot as plt
import itertools
from modules.misc import seaice_area_mean

################################################
#                Time series                   #
################################################

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