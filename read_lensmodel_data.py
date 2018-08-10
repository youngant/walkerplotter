import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
import glob
import os
from walkerplotter import *

def read_tmpfile_data(filename, parmnames):
	'''
	Reads the pixsrc file and returns a pandas dataframe containing the
	model parameters and corresponding chi-squared.
	'''
	data = np.loadtxt(filename)

	# Set data array to have parameters and chisq as columns
	nparms = len(parmnames) - 1
	models = data[:,:nparms]
	chisq  = data[:,-2,None]  # None makes the array have shape Nx1
	#chisq  = data[:,9,None]  # None makes the array have shape Nx1

	data = np.append(models, chisq, 1)

	# Put data into pandas dataframe
	models_df = pd.DataFrame(data=data, columns=parmnames)

	return models_df


def read_files(parmlist, directory = './', basename='best'):
	'''
	Read lensmodel/pixsrc MCMC output files and return a WalkerPlotter object.

	parmstoshow: a list of the indices of parameter values that are in the chain
				 data. Indices should start numbering from 0. The available
				 parameters are (0): 'b', (1): 'RA', (2): 'DEC', (3): 'e',
				 (4): 'theta_e', (5): 'gamma', (6); 'theta_gamma', (7): 's'.
				 The chisq data will always be passed to the object.
	'''

	# Get the list of files in the directory
	files = glob.glob(directory + basename + '-*.tmp')

	# Split the directory path from the file names
	dir_name = os.path.dirname(files[0])
	fn_list  = []
	for file in files:
		fn_list.append(os.path.basename(file))

	# Sort the file names
	fn_list.sort(key = lambda x: int(x[5:-4]))

	# Replace the file list with the sorted file names
	for i in range(0, len(fn_list)):
		files[i] = dir_name + '/' + fn_list[i]

	col_names 		= ['b', 'x', 'y', 'RA', 'DEC', 'e', 'theta_e', 'gamma', 
					   'theta_gamma', 's', 'e_x', 'e_y', 'gamma_x', 'gamma_y', 
					   'alpha', 'chisq']
	col_names_latex = [r'$b$', r'$x$', r'$y$', r'RA', r'DEC', r'$e$', 
					   r'$\theta_e$', r'$\gamma$', r'$\theta_{\gamma}$', r'$s$',
					   r'$e_x$', r'$e_y$', r'$\gamma_x$', r'$\gamma_y$', 
					   r'$\alpha$', r'$\chi^2$']

	# Make sure that chisq is in the list
	if 'chisq' not in parmlist:
		parmlist.append('chisq')

	parms_idx = []
	for parm in parmlist:
		parms_idx.append(col_names.index(parm))

	# Pick out the data to include names for.
	col_names_included = []
	col_names_latex_included = []
	for idx in parms_idx:
		col_names_included.append(col_names[idx])
		col_names_latex_included.append(col_names_latex[idx])

	# Read in the data and make a list of it.
	walkers = []
	for filename in files:
		walkers.append(read_tmpfile_data(filename, col_names_included))

	walk = WalkerPlotter(walkers, col_names_included,
						 parm_labels_latex = col_names_latex_included)

	return walk


def read_multiple_mcmc_folders(parmlist, directory = './', basename='mcmc'):
	'''
	Reads in the data from multiple MCMC runs in separate folders.

	This needs to be done because lensmodel does not properly take advantage of
	multi-thread CPUs.

	The contents of the directory folder need to only be identical folders each
	containing an MCMC run. The only difference between folders should be the
	random seed and the output files.
	'''

	# Get a folder list (This will be unsorted, but that should be fine.)
	files = glob.glob(directory + '{0}-*/{0}-*-*.tmp'.format(basename))

	col_names 		= ['b', 'RA', 'DEC', 'e', 'theta_e', 'gamma', 'theta_gamma',
					   's', 'e_x', 'e_y', 'gamma_x', 'gamma_y', 'chisq']
	col_names_latex = [r'$b$', r'RA', r'DEC', r'$e$', r'$\theta_e$', \
					   r'$\gamma$', r'$\theta_{\gamma}$', r'$s$', r'$e_x$',
					   r'$e_y$', r'$\gamma_x$', r'$\gamma_y$', r'$\chi^2$']

	# Make sure that chisq is in the list
	if 'chisq' not in parmlist:
		parmlist.append('chisq')

	parms_idx = []
	for parm in parmlist:
		parms_idx.append(col_names.index(parm))

	# Pick out the data to include names for.
	col_names_included = []
	col_names_latex_included = []
	for idx in parms_idx:
		col_names_included.append(col_names[idx])
		col_names_latex_included.append(col_names_latex[idx])

	# Read in the data and make a list of it.
	walkers = []
	for filename in files:
		walkers.append(read_tmpfile_data(filename, col_names_included))

	walk = WalkerPlotter(walkers, col_names_included,
						 parm_labels_latex = col_names_latex_included)

	return walk