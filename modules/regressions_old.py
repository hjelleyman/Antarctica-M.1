"""
This script contains functions needed for regressions.
"""
import xarray as xr
import scipy
import numpy as np

from Modules.preparingdata import detrend

import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

class regressions:

	"""
	Object for regression data. Used for regression analysis.
	
	Attributes:
		anomalous (bool): Whether to decompose data.
		datafolder (str): Location where source data is located.
		derrivative (bool): Whether to derrive data.
		detrended (bool): Whether to detrend data.
		DMI (xarray DataArray): DMI timeseries.
		figure (TYPE): Description
		imagefolder (str): Directory for output images to be stored.
		IPO (xarray DataArray): IPO timeseries.
		maxyear (TYPE): Description
		minyear (TYPE): Description
		mode (str): Temporal resolution parameter.
		n (int): Spatial resolution parameter.
		SAM (xarray DataArray): SAM timeseries.
		seaice_data (xarray DataArray): Data for the seaice in Atarctica.
		seaice_file (str): File location of the seaice raw data.
		seaice_mean (xarray DataArray): Timeseries of mean Antarctic SIE.
	
	Deleted Attributes:
		index_data (xarray DataArray): Data for the index(es) being investigated.
	"""

	def __init__(self, datafolder, imagefolder):
		"""Creates the regressions object.
		
		Args:
			datafolder (str): Location where source data is located
			imagefolder (str): Directory for output images to be stored.
		"""
		self.datafolder  = datafolder
		self.imagefolder = imagefolder

	def load_data(self, n=5, mode='monthly', anomalous = False, detrended = False, derrivative = False, indicies = ['DMI', 'SAM', 'IPO'], minyear = 1970, maxyear = 2020):
		"""Loads all the data at once.
		
		Args:
			n (int, optional): Spatial resolution parameter.
			mode (str, optional): Temporal resolution parameter.
			anomalous (bool, optional): Indicates if using anomalous data.
			detrended (bool, optional): Indicates if using detrended data.
			derrivative (bool, optional): Indicates if derrivative.
			indicies (list, optional): Which indicies to load.
			minyear (int, optional): Description
			maxyear (int, optional): Description
		"""

		self.anomalous   = anomalous
		self.detrended   = detrended
		self.derrivative = derrivative
		self.n           = n
		self.mode        = mode
		self.minyear     = minyear
		self.maxyear     = maxyear
		self.load_seaice_data(n=n,mode = mode, anomalous = anomalous, detrended = detrended, derrivative = derrivative)
		self.load_index_data(indicies, mode)

		self.seaice_data = self.seaice_data.sel(time=slice(f'{self.minyear}-01-01', f'{self.maxyear+1}-01-01')).copy()
		self.SAM         = self.SAM.sel(time=slice(f'{self.minyear}-01-01', f'{self.maxyear}-12-01')).copy()
		self.DMI         = self.DMI.sel(time=slice(f'{self.minyear}-01-01', f'{self.maxyear}-12-01')).copy()
		self.IPO         = self.IPO.sel(time=slice(f'{self.minyear}-01-01', f'{self.maxyear}-12-01')).copy()

		if self.detrended:
			self.DMI = detrend(self.DMI)
			self.SAM = detrend(self.SAM)
			self.IPO = detrend(self.IPO)

		self.DMI = (self.DMI - self.DMI.mean()) 
		self.DMI =  self.DMI / self.DMI.std()
		self.IPO = (self.IPO - self.IPO.mean()) 
		self.IPO =  self.IPO / self.IPO.std()
		self.SAM = (self.SAM - self.SAM.mean()) 
		self.SAM =  self.SAM / self.SAM.std()

		self.seaice_data = (self.seaice_data - self.seaice_data.mean()) 
		self.seaice_data =  self.seaice_data / self.seaice_data.std()



	def load_seaice_data(self, n=5, mode='monthly', anomalous = False, detrended = False, derrivative = False):
		"""Loads sea ice data.
		
		Args:
			n (int, optional): Spatial resolution parameter.
			mode (str, optional): Temporal resolution parameter.
			anomalous (bool, optional): Indicates if using anomalous data.
			detrended (bool, optional): Indicates if using detrended data.
			derrivative (bool, optional): Indicates if derrivative.
		"""

		if derrivative:
			derr = '_derrivative'
		else:
			derr = ''
		if anomalous:
			self.seaice_file = self.datafolder + f'seaice{derr}_anomalous_n{n}_{mode}_s.nc'
			if detrended:
				self.seaice_file = self.datafolder + f'seaice{derr}_anomalous_n{n}_{mode}_s_detrended.nc'
		else:
			self.seaice_file = self.datafolder + f'seaice{derr}_raw_n{n}_{mode}_s.nc'
			if detrended:
				self.seaice_file = self.datafolder + f'seaice{derr}_raw_n{n}_{mode}_s_detrended.nc'

		seaice = xr.open_dataset(self.seaice_file).__xarray_dataarray_variable__

		seaice = seaice.rename('SIC')

		self.seaice_data = seaice
		self.seaice_mean = self.seaice_data.mean(dim = ('x','y'))


	def load_index_data(self, indicies = ['DMI', 'SAM', 'IPO'], mode = 'monthly'):
		"""Loads index data.
		
		Args:
			indicies (list, optional): Indicies to be loaded.
			mode (str, optional): Temporal resolution parameter.
		"""

		if 'DMI' in indicies:
			self.DMI = xr.open_dataset('Data/Indicies/dmi.nc').DMI
			if mode == 'annually':
				self.DMI = self.DMI.resample(time = 'Y').mean()

		if 'SAM' in indicies:
			self.SAM = xr.open_dataset('Data/Indicies/sam.nc').SAM
			if mode == 'annually':
				self.SAM = self.SAM.resample(time = 'Y').mean()

		if 'IPO' in indicies:
			self.IPO = xr.open_dataset('Data/Indicies/ipo.nc').IPO
			if mode == 'annually':
				self.IPO = self.IPO.resample(time = 'Y').mean()


	def plot_scatter(self):
		"""Plots a scatter plot of sea ice against the index data.
		"""

		variables = [self.SAM.copy(), self.DMI.copy(), self.IPO.copy()]
		indicies  = ['SAM', 'DMI', 'IPO']

		fig, AX = plt.subplots(1,3, constrained_layout = True, figsize = (7, 7/3), sharey = True)

		AX[0].set_ylabel('SIE')

		for i in range(3):
			ax       = AX[i]
			seaice   = self.seaice_mean.copy()
			variable = variables[i]
			times    = list(set(seaice.time.values) & set(variable.time.values))			
			seaice   = seaice.loc[times]
			variable = variable.loc[times]

			# yl      = seaice.std()

			# # normalisation
			# seaice   = seaice / yl

			xlength  = 1.25 * max(-min(variable),max(variable))
			ylength  = 1.25 * max(-min(seaice),max(seaice))

			# Plotting
			ax.scatter(variable, seaice, c = seaice.time)
			ax.axhline(0, alpha = 0.5)
			ax.axvline(0, alpha = 0.5)
			ax.set_xlim(-xlength,xlength)
			ax.set_ylim(-ylength,ylength)
			ax.set_xlabel(indicies[i])

			# line of best fit
			m, b, r_value, p_value, std_err = scipy.stats.linregress(variable, seaice)
			ax.plot(np.array([-xlength,xlength]), m*np.array([-xlength,xlength]) + b, color = 'black')

			# correlation
			corr, pval = scipy.stats.pearsonr(variable, seaice)

			# Add metrics to plot
			# ax.text(-xlength*0.9,ylength*0.5, f'm = {m:.2f}\nR^2 = {r_value**2:.2f}\ncorr = {corr:.2f}\np-val = {pval:.2f}')


			# Labels
			if self.mode == 'annually':
				for x,y in zip(variable, seaice):

					label = str(x.time.dt.year.values)

					# this method is called for each point
					ax.annotate(label, # this is the text
								 (x.values,y.values), # this is the point to label
								 textcoords="offset points", # how to position the text
								 xytext=(5,5), # distance from text to points (x,y)
								 ha='center',   # horizontal alignment can be left, right or center
								 alpha = 0.3)
		fig.suptitle('Indicies and SIE in Antarctica')



		self.figure = [fig,AX]


		imagefile = namefile(self.imagefolder, "scatter", self.derrivative, self.detrended, self.anomalous, self.n, self.mode, self.minyear, self.maxyear)

		plt.savefig(imagefile, transparent = True)
		plt.show()


	def multiple_regression(self):
		"""Performs a multiple regression analysis on the mean sea ice.
		"""
		def linear_model(x, a, b, c, d):
			"""Summary
			
			Args:
				x (TYPE): Description
				a (TYPE): Description
				b (TYPE): Description
				c (TYPE): Description
				d (TYPE): Description
			
			Returns:
				TYPE: Description
			"""
			return d + a*x[0] + b*x[1] + c*x[2]

		variables = [self.SAM.copy(), self.DMI.copy(), self.IPO.copy()]
		seaice = self.seaice_mean.copy()

		times = list(set(seaice.time.values) & set(variables[0].time.values) & set(variables[1].time.values) & set(variables[2].time.values))

		seaice = seaice.loc[times]

		variables = [v.loc[times] for v in variables]
		seaice = seaice.sortby('time')
		variables = [v.sortby('time') for v in variables]

		# normalisation
		# for i in range(3):
		# 	variable = variables[i]
		# 	# xl       = variable.std()
		# 	# variable = variable / xl

		# 	variables[i] = variable

		p0 = [.12, -.03, -.04, 0]

		params, covariances = scipy.optimize.curve_fit(linear_model, variables, seaice, p0)
		print(params)

		plt.plot(seaice.time, linear_model(variables,*params)-seaice ,'o')
		plt.axhline(0)

		if self.derrivative:
			derr = '_derrivative'
		else:
			derr = ''
		if self.anomalous:
			imagefile = self.imagefolder + f'/multivariate_residual{derr}_anomalous_n{self.n}_{self.mode}_{self.minyear}_{self.maxyear}.pdf'
			if self.detrended:
				imagefile = self.imagefolder + f'/multivariate_residual{derr}_anomalous_n{self.n}_{self.mode}_detrended_{self.minyear}_{self.maxyear}.pdf'
		else:
			imagefile = self.imagefolder + f'/multivariate_residual{derr}_raw_n{self.n}_{self.mode}_{self.minyear}_{self.maxyear}.pdf'
			if self.detrended:
				imagefile = self.imagefolder + f'/multivariate_residual{derr}_raw_n{self.n}_{self.mode}_detrended_{self.minyear}_{self.maxyear}.pdf'

		plt.savefig(imagefile, transparent = True)
		plt.show()

		plt.plot(seaice.time, linear_model(variables,*params), label = 'model prediction')
		plt.plot(seaice.time, seaice, label = 'seaice')
		plt.axhline(0)
		plt.legend()
		
		imagefile = namefile(self.imagefolder, "multivariate_model", self.derrivative, self.detrended, self.anomalous, self.n, self.mode, self.minyear, self.maxyear)

		plt.savefig(imagefile, transparent = True)
		plt.show()


	def single_spatial_regression(self):
		"""Summary
		"""
		seaice = self.seaice_data
		variables = [self.SAM.copy(), self.DMI.copy(), self.IPO.copy()]
		indicies  = ['SAM', 'DMI', 'IPO']

		fig, AX = plt.subplots(1,3, constrained_layout = True, figsize = (11, 11/3))

		divnorm = colors.TwoSlopeNorm(vcenter=0, vmin = -1.3, vmax=1.3)
		for i in range(3):
			ax = AX[i]
			variable = variables[i]
			times    = list(set(seaice.time.values) & set(variable.time.values))			
			seaice   = seaice.loc[times]
			variable = variable.loc[times]


			# line of best fit
			m, b, r_value, p_value, std_err = xr.apply_ufunc(scipy.stats.linregress, variable, seaice,
															  input_core_dims=[['time'], ['time']],
															  vectorize=True, # !Important!
															  dask='parallelized',
															  output_dtypes=[float]*5,
															  output_core_dims=[[]]*5
															  )

			img = ax.imshow(m, cmap='RdBu_r', norm=divnorm)
			ax.set_title(indicies[i])
			divider = make_axes_locatable(AX[i])
			cax = divider.append_axes("right", size="5%", pad=0.05)
			cbar = plt.colorbar(img, cax=cax)
		cbar.set_label('regression coefficient')
		plt.suptitle('Single regression coefficients between SIE and different Indicies')
		AX[0].axis('off')
		AX[1].axis('off')
		AX[2].axis('off')

		imagefile = namefile(self.imagefolder, "spatial_multiple", self.derrivative, self.detrended, self.anomalous, self.n, self.mode, self.minyear, self.maxyear)

		plt.savefig(imagefile, transparent = True)

		plt.show()

	def multiple_spatial_regression(self):
		"""Performs a multiple regression analysis on the mean sea ice.
		"""
		def linear_model(x, a, b, c, d):
			"""Summary
			
			Args:
				x (TYPE): Description
				a (TYPE): Description
				b (TYPE): Description
				c (TYPE): Description
				d (TYPE): Description
			
			Returns:
				TYPE: Description
			"""
			return d + a*x[0] + b*x[1] + c*x[2]

		variables = [self.SAM.copy(), self.DMI.copy(), self.IPO.copy()]
		indicies  = ['SAM', 'DMI', 'IPO']
		seaice = self.seaice_data.copy()
		times = list(set(seaice.time.values) & set(variables[0].time.values) & set(variables[1].time.values) & set(variables[2].time.values))

		seaice = seaice.loc[times]

		variables = [v.loc[times] for v in variables]
		seaice = seaice.sortby('time')
		variables = [v.sortby('time') for v in variables]
		newvariables = xr.Dataset({v.name:v for v in variables})

		# normalisation
		# xl       = max(newvariables.max(), -newvariables.min())
		# newvariables = newvariables / xl


		p0 = np.array([.12, -.03, -.04, 0])

		newvariables = newvariables.to_array(dim='variable')


		def apply_curvefit(linear_model, newvariables, seaice):
			"""Applies the linear fitting to the data and returns the parameters.
			
			Args:
				linear_model (TYPE): The plotting model to use
				newvariables (TYPE): The variables to go in the model
				seaice (TYPE): The dependant variable.
			
			Returns:
				TYPE: Description
			"""
			params, covariances = scipy.optimize.curve_fit(linear_model, newvariables.transpose(), seaice)
			a, b, c, d  = params
			return a, b, c, d

		params = xr.apply_ufunc(apply_curvefit, 
								  linear_model, newvariables, seaice,
								  input_core_dims=[[], ['time','variable'], ['time']],
								  vectorize=True, # !Important!
								  # dask='parallelized',
								  output_dtypes=[float]*4,
								  output_core_dims=[[]]*4
								  )

		self.params = params
		

		fig, AX = plt.subplots(1,4, constrained_layout = True, figsize = (11, 11/3))

		max_  = max(p.max().values for p in params)
		min_  = min(p.min().values for p in params)
		divnorm = colors.TwoSlopeNorm(vcenter = 0, vmax = max_, vmin = min_)
		for i in range(4):
			ax = AX[i]
			img = ax.imshow(params[i], cmap = 'RdBu_r', norm = divnorm)
			ax.set_title(['a','b','c','d'][i])
		divider = make_axes_locatable(AX[3])
		cax = divider.append_axes("right", size="5%", pad=0.05)
		cbar = plt.colorbar(img, cax=cax)
		cbar.set_label('regression coefficient')
		plt.suptitle('Multiple regression coefficients between SIE and different Indicies\nmodel:\t (a $\\times$ SAM) + (b $\\times$ DMI) + (c $\\times$ IPO) + d')
		AX[0].axis('off')
		AX[1].axis('off')
		AX[2].axis('off')
		AX[3].axis('off')

		imagefile = namefile(self.imagefolder, 
								"spatial_multiple", 
								self.derrivative, 
								self.detrended, 
								self.anomalous, 
								self.n, 
								self.mode, 
								self.minyear, 
								self.maxyear)
		plt.savefig(imagefile, transparent = True)

		plt.show()


	def index_contribution(self):
		variables = [self.SAM.copy(), self.DMI.copy(), self.IPO.copy()]
		indicies  = ['SAM', 'DMI', 'IPO']
		seaice = self.seaice_data.copy()
		times = list(set(seaice.time.values) & set(variables[0].time.values) & set(variables[1].time.values) & set(variables[2].time.values))

		seaice = seaice.loc[times]

		variables = [v.loc[times] for v in variables]
		seaice = seaice.sortby('time')
		variables = [v.sortby('time') for v in variables]
		newvariables = xr.Dataset({v.name:v for v in variables})

		params = self.params

		fig, AX = plt.subplots(1,3, constrained_layout = True, figsize = (11, 11/3))

		# max_  = max(p.max().values for p in params)
		# min_  = min(p.min().values for p in params)
		divnorm = colors.TwoSlopeNorm(vcenter = 0)
		for i in range(3):
			ax = AX[i]
			divnorm = colors.TwoSlopeNorm(vcenter = 0)
			img = ax.imshow((params[i]*variables[i]).mean('time'), cmap = 'RdBu_r', norm = divnorm)
			ax.set_title(['a*SAM','b*DMI','c*IPO'][i])
			divider = make_axes_locatable(AX[i])
			cax = divider.append_axes("right", size="5%", pad=0.05)
			cbar = plt.colorbar(img, cax=cax)
		cbar.set_label('contribution')
		plt.suptitle('Contributions of different Indicies')
		AX[0].axis('off')
		AX[1].axis('off')
		AX[2].axis('off')

		# if self.derrivative:
		# 	derr = '_derrivative'
		# else:
		# 	derr = ''
		# if self.anomalous:
		# 	imagefile = self.imagefolder + f'/index_contribution{derr}_anomalous_n{self.n}_{self.mode}_{self.minyear}_{self.maxyear}.pdf'
		# 	if self.detrended:
		# 		imagefile = self.imagefolder + f'/index_contribution{derr}_anomalous_n{self.n}_{self.mode}_detrended_{self.minyear}_{self.maxyear}.pdf'
		# else:
		# 	imagefile = self.imagefolder + f'/index_contribution{derr}_raw_n{self.n}_{self.mode}_{self.minyear}_{self.maxyear}.pdf'
		# 	if self.detrended:
		# 		imagefile = self.imagefolder + f'/index_contribution{derr}_raw_n{self.n}_{self.mode}_detrended_{self.minyear}_{self.maxyear}.pdf'


		imagefile = namefile(self.imagefolder, 
								"index_contribution", 
								self.derrivative, 
								self.detrended, 
								self.anomalous, 
								self.n, 
								self.mode, 
								self.minyear, 
								self.maxyear)
		
		plt.savefig(imagefile, transparent = True)

		plt.show()
	

	def print_heading(self):
		size = os.get_terminal_size()

		sidespace = size[0]//4

		rows = ["Doing Regression Analysis for:"]
		if self.anomalous:
			rows += ["Anomalous"]
		else:
			rows += ["Raw"]
		if self.detrended:
			rows += ["Detrended"]
		rows += [f"{self.mode} data"]
		rows += [f"Resolution = {self.n}"]
		rows += [f"{self.minyear} - {self.maxyear}"]



		print('-'*((size[0]-2*sidespace)//2 * 2))
		for row in rows:
			if len(row)%2==1:
				row = ' '+row
			print(' '*((size[0]-2*sidespace-2-len(row))//2)+row+' '*((size[0]-2*sidespace-2-len(row))//2)+' ')
		print('-'*((size[0]-2*sidespace)//2 * 2))


	def spatial_model_verification(self):
		variables = [self.SAM.copy(), self.DMI.copy(), self.IPO.copy()]
		indicies  = ['SAM', 'DMI', 'IPO']
		seaice = self.seaice_data.copy()
		times = list(set(seaice.time.values) & set(variables[0].time.values) & set(variables[1].time.values) & set(variables[2].time.values))

		seaice = seaice.loc[times]

		variables = [v.loc[times] for v in variables]
		seaice = seaice.sortby('time')
		variables = [v.sortby('time') for v in variables]
		newvariables = xr.Dataset({v.name:v for v in variables})

		def linear_model(x, a, b, c, d):
			"""Summary
			
			Args:
				x (TYPE): Description
				a (TYPE): Description
				b (TYPE): Description
				c (TYPE): Description
				d (TYPE): Description
			
			Returns:
				TYPE: Description
			"""
			return d + a*x[0] + b*x[1] + c*x[2]

		expected = linear_model(variables, *self.params)
		plt.plot(expected.mean(dim = ('x','y')))
		plt.plot(seaice.mean(dim=('x','y')))
		plt.show()


def namefile(folder, classification, derrivative, detrended, anomalous, n, mode, minyear, maxyear):

	if derrivative:
			derr = '_derrivative'
	else:
		derr = ''
	if anomalous:
		filename     = f'{folder}/{classification}{derr}_anomalous_n{n}_{mode}_{minyear}_{maxyear}.pdf'
		if detrended:
			filename = f'{folder}/{classification}{derr}_anomalous_n{n}_{mode}_detrended_{minyear}_{maxyear}.pdf'
	else:
		filename     = f'{folder}/{classification}{derr}_raw_n{n}_{mode}_{minyear}_{maxyear}.pdf'
		if detrended:
			filename = f'{folder}/{classification}{derr}_raw_n{n}_{mode}_detrended_{minyear}_{maxyear}.pdf'

	return filename