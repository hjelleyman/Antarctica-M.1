"""Contains classes and functions required for data processing.
"""
# Loading relevant modules
import xarray as xr
import numpy  as np
import glob   as glob
import datetime
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.signal

# For printing headings
from modules.misc import print_heading


class dataprocessor(object):
	"""Dataprocessor contains the data for processing and coordinates it's standardisation. It contains seaice data from NSIDC, ERA5 data of a variety of variables and index datasets.
	each of these need the following functions to work in this class. (Required if more data is added to the system at a later date)
	
	load_data()              - to load data in.
	temporal_decomposition() - to split into raw, seasonal cycle and anomalous data.
	save_data()              - to save data to folder.
	
	
	Attributes
	----------
	indicies : list
		Which indicies to process.
	load_ERA5 : bool
		Should data from the ERA5 dataset be processed.
	load_indicies : bool
		Should data from the index datasets be processed.
	load_seaice : bool
		Are we processing seaice data.
	processeddatafolder : str
		File path for output processed data.
	rawdatafolder : str
		File path for source data.
	seaice_data : object
		Object containing seaice data.
	variables : list
		Which Era5 variables to load.
	"""
	def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processed_data/'):
		"""Generates a dataprocessor object.
		
		Parameters
		----------
		rawdatafolder : str, optional
			Path to raw data.
		processeddatafolder : str, optional
			Path to output data.
		"""
		heading = "Generating a data processor"
		print_heading(heading)

		# Saving datafolder paths to object
		self.rawdatafolder = rawdatafolder
		self.processeddatafolder = processeddatafolder

	def load_data(self, load_seaice = False, load_indicies = False, load_ERA5 = False, indicies = ['SAM'], variables = ['t2m']):
		"""Adds raw data to the processor object.
		
		Parameters
		----------
		load_seaice : bool, optional
			Decides if we should load seaice data.
		load_indicies : bool, optional
			Decides if we should load index data.
		load_ERA5 : bool, optional
			Description
		indicies : list, optional
			Which indicies to load as index data.
		variables : list, optional
			which era5 variables to load.
		
		Deleted Parameters
		------------------
		n : int, optional
		    Spatial resolution parameter.
		"""

		# Setting which datasets to load for processing
		self.load_seaice   = load_seaice
		self.load_indicies = load_indicies
		self.load_ERA5     = load_ERA5

		# For datasets with multiple variables, which should be loaded.
		self.indicies      = indicies
		self.variables     = variables


		if self.load_seaice:
			heading = "Loading seaice data from NSIDC"
			print_heading(heading)
			self.seaice_data = seaice_data(rawdatafolder       = self.rawdatafolder,
										   processeddatafolder = self.processeddatafolder)
			self.seaice_data.load_data()

		if self.load_indicies:
			heading = f"Loading index data"
			print_heading(heading)
			self.index_data = index_data(rawdatafolder       = self.rawdatafolder,
										 processeddatafolder = self.processeddatafolder,
										 indicies = self.indicies)
			self.index_data.load_data()

		if self.load_ERA5:
			heading = f"Loading ECMWF ERA5 data"
			print_heading(heading)

	def decompose_and_save(self, resolutions = [1,5,10,20], temporal_resolution = ['monthly', 'seasonally', 'annually'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
		"""Summary
		
		Parameters
		----------
		resolutions : list, optional
		    Description
		temporal_resolution : list, optional
		    Description
		temporal_decomp : list, optional
		    Description
		"""
		if self.load_seaice:
			self.seaice_data.decompose_and_save(resolutions = resolutions, temporal_resolution = temporal_resolution, temporal_decomposition = temporal_decomposition, detrend = detrend)
			
		if self.load_indicies:
			self.index_data.decompose_and_save(temporal_resolution = temporal_resolution, temporal_decomposition = temporal_decomposition, detrend = detrend)
			
	


class seaice_data:
	"""Class for seaice data.
	
	Attributes
	----------
	data : xarray DataArray
		The data for seaice.
	files : list
		list of seaice raw data files.
	output_folder : str
		File path for output data folder.
	source_folder : str
		File path for source data folder.
	
	Deleted Attributes
	------------------
	n : int
	    spatial resolution parameter.
	"""
	
	def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processeddata/', n = 5):
		"""Loads the raw data.
		
		Parameters
		----------
		rawdatafolder : str, optional
			File path for raw data.
		processeddatafolder : str, optional
			File path for processed data.
		n : int, optional
			Spatial resolution parameter.
		"""

		self.source_folder = rawdatafolder + 'SIC-monthly/'
		self.output_folder = processeddatafolder + 'SIC/'
		self.files         = glob.glob(self.source_folder+'*.bin')

	def load_data(self):
		"""Iterates over seaice files and loads as an object.
		"""
		data      = []
		dates     = []
		errorlist = []
		sic_files = self.files
		n         = 1
		for file in sic_files:
			date  = file.split('_')[-4]
			try:
				data += [self.readfile(file)[::n,::n]]
			except ValueError:
				print(file)
				data += [data[-1]]
				errorlist += [(date,file)]

			# try:
			#     date = datetime.datetime.strptime(date, '%Y%m%d')
			# except:
			date = datetime.datetime.strptime(date, '%Y%m')
			dates += [date]
		for date, file in errorlist:
			i = int(np.where(np.array(files) == file)[0])
			data[i] = (data[i-1]+data[i+1])/2
		
		data = np.array(data, dtype = float)
		
		x = 10*np.arange(-395000,395000,2500)[::n]
		y = 10*np.arange(435000,-395000,-2500)[::n]
		x,y = np.meshgrid(x,y)
		
		sie = data[0]
		
		x_coastlines = x.flatten()[sie.flatten()==253]
		y_coastlines = y.flatten()[sie.flatten()==253]
		
		seaice = xr.DataArray(data, 
							  coords={'time': dates,
									  'x': 10*np.arange(-395000, 395000, 2500)[::n],
									  'y': 10*np.arange( 435000,-395000,-2500)[::n]},
							  dims=['time', 'y', 'x'])
		seaice.rename('seaice_concentration')

		self.data = seaice
		self.data = self.data.sortby('time')

	def decompose_and_save(self, resolutions = [1,5,10,20], temporal_resolution = ['monthly', 'seasonally', 'annually'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
		"""Break the data into different temporal splits.
		"""
		dataset = xr.Dataset({'source':self.data.copy()})

		dataset.to_netcdf(self.output_folder+'source.nc')

		heading = 'Splitting the seaice data up'
		print_heading(heading) 

		for n, temp_res, temp_decomp, dt in itertools.product(resolutions, temporal_resolution, temporal_decomposition, detrend):
			print(n, temp_res, temp_decomp, dt)
			# Spatial resolution fix.
			new_data = dataset.source.loc[:,::n,::n].copy()

			# Temporal interpolation for missing data.
			new_data = new_data.resample(time = '1MS').fillna(np.nan)
			new_data = new_data.sortby(new_data.time)
			new_data = new_data.groupby('time.month').apply(lambda group: group.sortby(group.time).interp(method='linear'))


			# Detrend
			if 'detrended' == dt:
				new_data = new_data.sortby(new_data.time)
				new_data = detrend_data(new_data)

			# If anomalous remove seasonal cycle
			if temp_decomp == 'anomalous':
				climatology = new_data.groupby("time.month").mean("time")
				new_data = new_data.groupby("time.month") - climatology


			# temporal averaging
			if temp_res == 'seasonally':
				new_data = new_data.resample(time="QS-DEC").mean()

			elif temp_res == 'annually':
				new_data = new_data.resample(time="YS").mean()
			# plt.plot(new_data.mean(dim = ('x','y')))
			# plt.show()

			# dataset = xr.Dataset({'source':self.data.copy()})
			# dataset[f'{temp_decomp}_{temp_res}_{n}'] = new_data

			new_data.name = f'{temp_decomp}_{temp_res}_{n}_{dt}'
			new_data.to_netcdf(self.output_folder + new_data.name +'.nc')

		# self.data = dataset
				
		print_heading('DONE')



	def readfile(self, file):
		"""Reads a binary data file and returns the numpy data array.
		
		Parameters
		----------
		file : str
			File path.
		
		Returns
		-------
		data = (numpy array) data contained in the file.
		"""
		with open(file, "rb") as binary_file:
			# Seek a specific position in the file and read N bytes
			binary_file.seek(300, 0)  # Go to beginning of the file
			data = binary_file.read()         # data array
			data = np.array(list(data)).reshape(332, 316)
		return data

class index_data:

	"""Class for index data.
	
	Attributes
	----------
	indicies : list
	    Which indicies to load.
	output_folder : str
	    Path to output folder.
	source_folder : str
	    Path to source folder.
	"""

	def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processeddata/', indicies = ['SAM']):
		"""Loads the raw data.
		
		Parameters
		----------
		rawdatafolder : str, optional
			File path for raw data.
		processeddatafolder : str, optional
			File path for processed data.
		indicies : list, optional
		    which indicies to load.
		"""

		self.source_folder = rawdatafolder + 'indicies/'
		self.output_folder = processeddatafolder + 'INDICIES/'

		self.indicies = indicies

	def load_data(self):
		"""Summary
		"""
		self.data = xr.Dataset()
		if 'DMI' in self.indicies:
			dmi = xr.open_dataset('Data/Indicies/dmi.nc')
			self.data['DMI'] = dmi.DMI

		if 'SAM' in self.indicies:
			sam = np.genfromtxt('Data/Indicies/newsam.1957.2007.txt', skip_header =1, skip_footer = 1)[:,1:]

			index = range(1957, 2020)
			columns = range(1,13)

			sam = pd.DataFrame(data = sam, columns = columns, index = index)
			sam = sam.stack().reset_index()
			sam.columns = ['year', 'month', 'SAM']
			sam['time'] = pd.to_datetime(sam.year*100+sam.month,format='%Y%m')
			sam = sam.set_index('time').SAM
			sam = xr.DataArray(sam)
			self.data['SAM'] = sam

		if 'IPO' in self.indicies:
			ipo = np.genfromtxt('Data/Indicies/tpi.timeseries.ersstv5.data', skip_header = 1, skip_footer = 11)[:,1:]

			index = range(1854, 2021)
			columns = range(1,13)

			ipo = pd.DataFrame(data = ipo, columns = columns, index = index)
			ipo = ipo.stack().reset_index()
			ipo.columns = ['year', 'month', 'IPO']
			ipo['time'] = pd.to_datetime(ipo.year*100+ipo.month,format='%Y%m')
			ipo = ipo.set_index('time').IPO
			ipo = ipo[ipo>-10]
			ipo = xr.DataArray(ipo)

	def decompose_and_save(self, temporal_resolution = ['monthly', 'seasonally', 'annually'], temporal_decomposition = ['raw', 'anomalous'], detrend = ['raw', 'detrended']):
		"""Break the data into different temporal splits.
		"""

		heading = 'Splitting the index data up'
		print_heading(heading) 

		for temp_res, temp_decomp, dt in itertools.product(temporal_resolution, temporal_decomposition, detrend):
			print(temp_res, temp_decomp, dt)
			# Spatial resolution fix.
			new_data = self.data.copy()

			# Temporal interpolation for missing data.
			new_data = new_data.resample(time = '1MS').fillna(np.nan)
			new_data = new_data.sortby(new_data.time)
			new_data = new_data.groupby('time.month').apply(lambda group: group.sortby(group.time).interp(method='linear'))


			# Detrend
			if 'detrended' == dt:
				for index in new_data:
					subdata = new_data[index].copy()
					subdata = subdata.sortby(subdata.time)
					subdata = subdata.dropna(dim='time')
					subdata = detrend_data(subdata)
					new_data[index] = subdata

			# If anomalous remove seasonal cycle
			if temp_decomp == 'anomalous':
				climatology = new_data.groupby("time.month").mean("time")
				new_data = new_data.groupby("time.month") - climatology


			# temporal averaging
			if temp_res == 'seasonally':
				new_data = new_data.resample(time="QS-DEC").mean()

			elif temp_res == 'annually':
				new_data = new_data.resample(time="YS").mean()
			# plt.plot(new_data.mean(dim = ('x','y')))
			# plt.show()

			# dataset = xr.Dataset({'source':self.data.copy()})
			# dataset[f'{temp_decomp}_{temp_res}_{n}'] = new_data

			new_dataname = f'{temp_decomp}_{temp_res}_{dt}'
			new_data.to_netcdf(self.output_folder + new_dataname +'.nc')

		# self.data = dataset
				
		print_heading('DONE')




class era5_data:

	"""Class for index data.
	
	Attributes
	----------
	output_folder : str
	    Path to output folder.
	source_folder : str
	    Path to source folder.
	variables : list
	    Which variables to load.
	
	Deleted Attributes
	------------------
	n : int
	    Spatial resolution parameter.
	"""

	def __init__(self, rawdatafolder = 'data/', processeddatafolder = 'processeddata/', variables = ['SAM']):
		"""Loads the raw data.
		
		Parameters
		----------
		rawdatafolder : str, optional
			File path for raw data.
		processeddatafolder : str, optional
			File path for processed data.
		variables : list, optional
		    which variables to laod.
		n : int, optional
			Spatial resolution parameter.
		"""

		self.source_folder = rawdatafolder + 'ECMWF/'
		self.output_folder = processeddatafolder + 'ERA5/'

		self.variables     = variables

	def load_data():
		"""Summary
		"""
		pass
	
	def decompose_and_save():
		"""Summary
		"""
		pass

def detrend_data(t):
    return xr.apply_ufunc(scipy.signal.detrend, t,
                          input_core_dims=[['time']],
                          vectorize=True, # !Important!
                          dask='parallelized',
                          output_core_dims=[['time']],
                          )