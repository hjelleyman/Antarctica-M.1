# Loading modules
# dataprocessing can be accessed with dp
# plotting can be accessed with plot
from modules import *

import itertools
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# What data to load
load_seaice   = True
load_indicies = True
load_ERA5     = True

# What indicies and variables
indicies  = ['SAM','IPO', 'DMI', 'ENSO']
variables = ['t2m']

# Resolutions to save data as.
resolutions = [1,5]
n = 5

# temporal averages
temporal_resolution = ['monthly', 'seasonal', 'annual']

# temporal_breakdown
temporal_decomposition = ['raw', 'anomalous']

# detrending
detrend = ['raw', 'detrended']

# # Generate a processor object
# processor = dp.dataprocessor(rawdatafolder = 'data/', processeddatafolder = 'processed_data/')

# # Load in datasets
# processor.load_data(load_seaice   = load_seaice,
#                     load_indicies = load_indicies,
#                     load_ERA5     = load_ERA5,
#                     indicies      = indicies,
#                     variables     = variables)

# # Change resolution of data
# processor.decompose_and_save(resolutions            = resolutions,
#                              temporal_resolution    = temporal_resolution,
#                              temporal_decomposition = temporal_decomposition,
#                              detrend                = detrend)

# seaice_source = 'ecmwf'
# for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
#     print(n, temp_res, temp_decomp, dt)
#     correlator = corr.correlator(process_seaice = load_seaice,
#                                  process_indicies = load_indicies,
#                                  indicies = indicies,
#                                  anomlous = temp_decomp == 'anomalous',
#                                  temporal_resolution = temp_res,
#                                  spatial_resolution = n,
#                                  detrend = dt == 'detrended',
#                                  outputfolder = 'processed_data/correlations/',
#                                  input_folder = 'processed_data/',
#                                  seaice_source = seaice_source)
#     print('    Computing correlation for mean SIC')
#     correlator.correlate_mean_sic_indicies()
#     print('    Computing spatial correlations')
#     correlator.correlate_spatial_sic_indicies()
#     print('    Saving to file')
#     correlator.save_data()

# for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
#     print(n, temp_res, temp_decomp, dt)
#     regressor = regr.regressor(process_seaice = load_seaice,
#                                  process_indicies = load_indicies,
#                                  indicies = indicies,
#                                  anomlous = temp_decomp == 'anomalous',
#                                  temporal_resolution = temp_res,
#                                  spatial_resolution = n,
#                                  detrend = dt == 'detrended',
#                                  outputfolder = 'processed_data/regressions/',
#                                  input_folder = 'processed_data/',
#                                  seaice_source = seaice_source)
#     print('    Computing regression for mean SIC')
#     regressor.regress_mean_sic_indicies()
#     print('    Computing spatial regression')
#     regressor.regress_spatial_sic_indicies()
#     print('    Computing comprehensive regression for mean SIC')
#     regressor.multiple_regression()
#     print('    Computing spatial comprehensive regression')
#     regressor.multiple_spatial_regression()
#     print('    Saving results')
#     regressor.save_data()

# seaice_source = 'nsidc'
# for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
#     print(n, temp_res, temp_decomp, dt)
#     correlator = corr.correlator(process_seaice = load_seaice,
#                                  process_indicies = load_indicies,
#                                  indicies = indicies,
#                                  anomlous = temp_decomp == 'anomalous',
#                                  temporal_resolution = temp_res,
#                                  spatial_resolution = n,
#                                  detrend = dt == 'detrended',
#                                  outputfolder = 'processed_data/correlations/',
#                                  input_folder = 'processed_data/',
#                                  seaice_source = seaice_source)
#     print('    Computing correlation for mean SIC')
#     correlator.correlate_mean_sic_indicies()
#     print('    Computing spatial correlations')
#     correlator.correlate_spatial_sic_indicies()
#     print('    Saving to file')
#     correlator.save_data()

# for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
#     print(n, temp_res, temp_decomp, dt)
#     regressor = regr.regressor(process_seaice = load_seaice,
#                                  process_indicies = load_indicies,
#                                  indicies = indicies,
#                                  anomlous = temp_decomp == 'anomalous',
#                                  temporal_resolution = temp_res,
#                                  spatial_resolution = n,
#                                  detrend = dt == 'detrended',
#                                  outputfolder = 'processed_data/regressions/',
#                                  input_folder = 'processed_data/',
#                                  seaice_source = seaice_source)
#     print('    Computing regression for mean SIC')
#     regressor.regress_mean_sic_indicies()
#     print('    Computing spatial regression')
#     regressor.regress_spatial_sic_indicies()
#     print('    Computing comprehensive regression for mean SIC')
#     regressor.multiple_regression()
#     print('    Computing spatial comprehensive regression')
#     regressor.multiple_spatial_regression()
#     print('    Saving results')
#     regressor.save_data()

seaice_source = 'nsidc'
# Plot seaice time series
plt.style.use('stylesheets/timeseries.mplstyle')
plot.plot_all_seaice(resolutions, temporal_resolution, temporal_decomposition, detrend, seaice_source = seaice_source)
plt.close()

# Plot index time series
plt.style.use('stylesheets/timeseries.mplstyle')
plot.plot_all_indicies(resolutions, temporal_resolution, temporal_decomposition, detrend, indicies = indicies)
plt.close()

# Plot index time series with SIC
plt.style.use('stylesheets/timeseries.mplstyle')
plot.plot_all_indicies_sic(resolutions, temporal_resolution, temporal_decomposition, detrend, indicies, seaice_source = seaice_source)
plt.close()

# Plot seaice time series
plt.style.use('stylesheets/timeseries.mplstyle')
plot.plot_all_sic_sic(resolutions, temporal_resolution, temporal_decomposition, detrend)
plt.close()

plt.style.use('stylesheets/timeseries.mplstyle')
plot.plot_single_correlations(resolutions, temporal_resolution, temporal_decomposition, detrend, seaice_source = seaice_source)
plt.close()

correlations, pvalues = plot.gen_single_correlation_table(resolutions, temporal_resolution, temporal_decomposition, detrend, seaice_source = seaice_source)

plt.style.use('stylesheets/contour.mplstyle')
plot.plot_spatial_correlations(resolutions, temporal_resolution, temporal_decomposition, detrend, seaice_source=seaice_source)
plt.close()

plt.style.use('stylesheets/contour.mplstyle')
plot.plot_spatial_correlations_with_significance(resolutions, temporal_resolution, temporal_decomposition, detrend, seaice_source = seaice_source)
plt.close()

plt.style.use('stylesheets/scatter.mplstyle')
plot.plot_all_regression_scatter(resolutions, temporal_resolution, temporal_decomposition, detrend, seaice_source = seaice_source)
plt.close()

regressions, pvals, rvalues, bvalues, stderr = plot.gen_single_regression_table(resolutions, temporal_resolution, temporal_decomposition, detrend)

plt.style.use('stylesheets/contour.mplstyle')
plot.plot_all_regression_spatial(resolutions, temporal_resolution, temporal_decomposition, detrend)
plt.close()

plt.style.use('stylesheets/contour.mplstyle')
plot.plot_all_regression_multiple_spatial(resolutions, temporal_resolution, temporal_decomposition, detrend)
plt.close()

plt.style.use('stylesheets/contour.mplstyle')
plot.plot_all_regression_multiple_contribution_spatial(resolutions, temporal_resolution, temporal_decomposition, detrend)
plt.close()
