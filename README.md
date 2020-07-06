# Antarctica-M.1
The latest version of code for my MSc looking at sea ice in Antarctica. Thesis can be found at this [link](https://github.com/hjelleyman/Masters-Thesis)

## Notebooks

These are for running the code and printing outputs. They are found in this folder and are as follows.

### Dataprep

This notebook is primarily used to prep the data for further analysis. It also contains plots of the different timeseries being investigated.

### Correlations

This notebook is used to generate results for the correlation analysis.

### Regressions

This notebook is used to generate results for the regression analysis.

### Composites

This notebook is used to generate results for the composite analysis.

## Scripts

These can be found in the **modules** folder and contain the bulk of the code. They are as follows.

### Dataprocessing (dp)

Contains the class dataprocesser which is used to process data.

### Correlations (corr)

Contains the class correlator which is used to correlate processed data.

### Regreessions (regr)

Contains the class regressor which is used to regress data.

### Composites (comp)

Contains the class compositor which is used to generate composites.

### Plotting (plot)

Contains functions for plotting so style can be kept consistant throught the entire thesis.