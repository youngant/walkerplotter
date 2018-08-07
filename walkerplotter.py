'''
This is basically a version 2.0 of walkerplotter.py

This script defines the WalkerPlotter class which holds and can plot and manipulate
the data from a set of MCMC chains (walkers).
'''

import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
import warnings

class WalkerPlotter:
	'''
	Holds all of the data for plotting MCMC walkers in a high-dimensional
	parameter space and performs the plotting in a corner-style plot as well
	as some other formats.

	Class variables:
		walkers : A list of pandas DataFrame objects. Each item in the list
				  holds all of the parameters for each walker step including the
				  chi^2, which is the last column of the table.

		parm_labels : A list of strings containing each of the parameter labels.

		parm_labels_latex :	...

		nwalkers : length of the walkers object

		nmods :	number of models that were tested (sum over all walkers)

		nparms : number of parameters for each model

	'''

	def __init__(self, walkers, parm_labels, parm_labels_latex=None):
		'''
		Initialize the WalkerPlotter object.

		Parameters
		----------
		walkers: list
			Each item in the list must be a pandas DataFrame containing the data
			for a single MCMC chain. The chains do not need to be the same 
			length. For a model with N parameters, the DataFrame should have N+1
			columns. The first N columns should contain the model parameters, 
			and the last column should give the chi^2 for each model. Each row 
			should give the data for a single model evaluation.

		parm_labels : list
			Each item in the list should be a string giving the name of the parameter in the corresponding column of the DataFrames in walkers.

		parm_labels_latex : list, optional, default: None
			Each item in the list should be a LaTeX string containing the 
			formatted label for each parameter. Each entry should match the
			corresponding entry in parm_labels.
		'''
		self._walkers 	  = walkers
		self._parm_labels = parm_labels

		if parm_labels_latex is None:
			self._parm_labels_latex = parm_labels
		else:
			self._parm_labels_latex = parm_labels_latex

		# Record useful information about the object
		self._nwalkers = len(walkers)
		self._nmods    = np.sum(np.array([walker.shape[0] for walker in \
								self._walkers]))
		self._nparms   = walkers[0].shape[1]

		print('Plotting {} models.'.format(self.nmods))

		# Set the colors for the walker lines.
		colormap = plt.cm.nipy_spectral  # gist_ncar, nipy_spectral, Set1,Paired
		self._colors = [colormap(x) for x in np.linspace(0.1, 1, self.nwalkers)]

	@property
	def walkers(self):
		return self._walkers
	@property
	def parm_labels(self):
		return self._parm_labels
	@property
	def parm_labels_latex(self):
		return self._parm_labels_latex
	@property
	def nwalkers(self):
		return self._nwalkers
	@property
	def nmods(self):
		return self._nmods
	@property
	def nparms(self):
		return self._nparms
	@property
	def fig(self):
		return self._fig
	@property
	def axarr(self):
		return self._axarr
	@property
	def colors(self):
		return self._colors
	@property
	def all_walkers(self):
		return pd.concat(self._walkers)



	def return_chain_2parms(self, n, col_i, col_j):
		'''
		Returns the values of parameters col_i and col_j for chain n.
		
		Parameters
		----------
		n : int
			Index of the chain.

		col_i : int
			Index of the first parameter to be returned.

		col_j : int
			Index of the second parameter to be returned.

		Returns
		-------
		parm_i : numpy.ndarray
			Array containing all of the values of parameter col_i for chain n.

		parm_j : numpy.ndarray
			Array containing all of the values of parameter col_j for chain n.
		'''
		parm_i = self.walkers[n].iloc[:,col_i].as_matrix()
		parm_j = self.walkers[n].iloc[:,col_j].as_matrix()

		return parm_i, parm_j


	def return_chain_parms(self, n, *cols):
		'''
		Returns the values for each parameter in *cols for chain n.

		Parameters
		----------
		n : int
			Index of the chain.

		cols : ints
			An arbitrary number of integers specifying indices of the parameters
			to be returned.

		Returns
		-------
		parms : tuple
			A tuple containing a numpy ndarray of parameter values for each 
			value passed in cols. 
:		'''
		parms = tuple(self.walkers[n].iloc[:,col].as_matrix() for col in cols)
		
		if len(cols) == 0:
			print('You probably wanted to give the indices of some parameters.')
		else:
			return parms


	def plot_walkers(self, i, j, idxs=None, ax=None, figsize=None, savefig=None, plotendsonly=False):
		'''
		Plots the values of parameters i and j for each chain with a different 
		color. Starting and ending points are also marked. 

		Parameters
		----------
		i : int
			Index of the parameter plotted on the x-axis.
		
		j : int
			Index of the parameter plotted on the y-axis.
		
		idxs : int, list, tuple, or None, optional, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

		ax : Axis or None, optional, default: None
			If None, a new Figure and Axis will be created. Otherwise, the 
			figure will be plotted to the provided Axis. Should be used when 
			adding the axes as a subplot to a figure.

		figsize: tuple of integers, optional, default: None
			The figsize argument passed to pyplot.

		savefig: str, optional, default: None
			If None, no figure is saved. Otherwise, must be a string specifying 
			the format for the figure to be saved in, e.g., 'png', 'pdf', etc.

		plotendsonly : bool, optional, default: False
			Whether or not to only plot starting and ending points of the chains.
		'''
		noax = ax is None 
		if noax:
			fig = plt.figure(figsize=figsize)
			ax  = fig.gca()

		# Check/set up the list of walkers to iterate over for plotting
		if idxs is None:
			idxs = list(range(0, self.nwalkers))
		elif isinstance(idxs, (int, list, tuple, np.ndarray)):
			if isinstance(idxs, int):
				idxs = [idxs]
			else:
				# Check that each item in the list is an int
				for n in idxs:
					if not isinstance(n, int):
						raise TypeError('idxs must be an int or a list/tuple/1d numpy array of ints.')
			
			# Check that the indices are within the correct range.
			for n in idxs:
				if n > (self.nwalkers - 1):
					raise ValueError('Index {} in idxs must be less than self.nwalkers - 1 = {}.'.format(n,self.nwalkers-1))
		else:
			raise TypeError('idxs must be an int or list of ints.')

		if plotendsonly is False:
			# Plot the line
			for n in idxs:
				parm_i, parm_j = self.return_chain_2parms(n, i, j)
				ax.plot(parm_i, parm_j, color=self.colors[n], linewidth=0.8)

		# Mark the starting and ending points on top
		for n in idxs:
			parm_i, parm_j = self.return_chain_2parms(n, i, j)
			ax.plot(parm_i[0], parm_j[0], 'k.')
			ax.plot(parm_i[-1], parm_j[-1], 'kx')

		# Add information if only a single plot is being made
		if noax:
			# Set labels
			ax.set_xlabel(self.walkers[0].columns.values[i])
			ax.set_ylabel(self.walkers[0].columns.values[j])

			# Set legend
			black_circle = ax.plot([], [], 'k.', label='Start')
			black_x = ax.plot([], [], 'kx', label='Stop')
			plt.legend()

			# TODO: Add a list of the walkers which were plotted to the figure.

		if savefig is not None:
			if not isinstance(savefig, str):
				raise TypeError('savefig must be either None or a string.')
			plt.savefig('histogram_{}.{}'.format(i, savefig), format=savefig)


	def plot_histogram(self, i, bins=20, idxs=None, ax=None, figsize=None, savefig=None):
		'''
		Plots a histogram of the ith parameter using the data from all chains.

		Parameters
		----------
		i : int
			The index of the parameter to be plotted.

		bins : int, optional, default: 20
			The number of bins for the histogram.

		idxs : int, list, tuple, or None, optional, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

		ax : Axis or None, optional, default: None
			If None, a new Figure and Axis will be created. Otherwise, the 
			figure will be plotted to the provided Axis. Should be used when 
			adding the axes as a subplot to a figure.

		figsize: tuple of integers, optional, default: None
			The figsize argument passed to pyplot.

		savefig: str, optional, default: None
			If None, no figure is saved. Otherwise, must be a string specifying 
			the format for the figure to be saved in, e.g., 'png', 'pdf', etc.
		'''
		noax = ax is None 
		if noax:
			fig = plt.figure(figsize=figsize)
			ax  = fig.gca()

		# Check/set up the list of walkers to iterate over for plotting
		if idxs is None:
			idxs = list(range(0, self.nwalkers))
		elif isinstance(idxs, (int, list, tuple, np.ndarray)):
			if isinstance(idxs, int):
				idxs = [idxs]
			else:
				# Check that each item in the list is an int
				for n in idxs:
					if not isinstance(n, int):
						raise TypeError('idxs must be an int or a list/tuple/1d numpy array of ints.')
			
			# Check that the indices are within the correct range.
			for n in idxs:
				if n > (self.nwalkers - 1):
					raise ValueError('Index {} in idxs must be less than self.nwalkers - 1 = {}.'.format(n,self.nwalkers-1))
		else:
			raise TypeError('idxs must be an int or list of ints.')

		# Make the list of all data points.
		pdata = []
		for n in idxs:
			pdata = np.append(pdata, self.walkers[n].iloc[:,i].as_matrix())

		ax.hist(pdata, bins=bins, histtype='step', color='black')

		if noax:
			# Set label
			ax.set_xlabel(self.walkers[0].columns.values[i])

		if savefig is not None:
			if not isinstance(savefig, str):
				raise TypeError('savefig must be either None or a string.')
			plt.savefig('histogram_{}.{}'.format(i, savefig), format=savefig)

	def plot_parameter(self, i, idxs=None, ax=None, figsize=None, savefig=None):
		'''
		Plots the values of a single parameter as a function of step number for 
		each chain.

		Parameters
		----------
		idxs : int, list, tuple, or None, optional, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

		ax : Axis or None, optional, default: None
			If None, a new Figure and Axis will be created. Otherwise, the 
			figure will be plotted to the provided Axis. Should be used when 
			adding the axes as a subplot to a figure.

		figsize: tuple of integers, optional, default: None
			The figsize argument passed to pyplot.

		savefig: str, optional, default: None
			If None, no figure is saved. Otherwise, must be a string specifying 
			the format for the figure to be saved in, e.g., 'png', 'pdf', etc.
		'''
		noax = ax is None 
		if noax:
			fig = plt.figure(figsize=figsize)
			ax  = fig.gca()
			ax.set_xlabel('Number of steps', fontsize=12)

		# Check/set up the list of walkers to iterate over for plotting
		if idxs is None:
			idxs = list(range(0, self.nwalkers))
		elif isinstance(idxs, (int, list, tuple, np.ndarray)):
			if isinstance(idxs, int):
				idxs = [idxs]
			else:
				# Check that each item in the list is an int
				for n in idxs:
					if not isinstance(n, int):
						raise TypeError('idxs must be an int or a list/tuple/1d numpy array of ints.')
			
			# Check that the indices are within the correct range.
			for n in idxs:
				if n > (self.nwalkers - 1):
					raise ValueError('Index {} in idxs must be less than self.nwalkers - 1 = {}.'.format(n,self.nwalkers-1))
		else:
			raise TypeError('idxs must be an int or list of ints.')

		# Loop over each walker and plot the data points
		for n in idxs:
			walker = self.walkers[n]
			steps = range(1, walker.shape[0]+1)
			data  = walker.iloc[:,i].as_matrix()
			ax.plot(steps, data, color=self.colors[n], linewidth=0.8)

		# Loop over each walker and mark the starting and ending points
		for n in idxs:
			walker = self.walkers[n]
			steps = range(1, walker.shape[0]+1)
			data  = walker.iloc[:,i].as_matrix()

			ax.plot(steps[0], data[0], color=self.colors[n], marker='o')    # Start
			ax.plot(steps[-1], data[-1], color=self.colors[n], marker='D')  # Stop

		ax.set_ylabel(self.parm_labels_latex[i], fontsize=12)


	def plot_corner_walkers(self, idxs=None, showhist=False, savefig = None, dpi = 'auto', plotendsonly=False):
		'''
		Plots the all of the parameter values for each chain in a different 
		color in a corner-style plot. Starting and ending points are also 
		marked. Unfortunately, the later-plotted chains usually cover the 
		earlier-plotted ones when there are more than a few chains.

		Parameters
		----------
		idxs : int, list, tuple, or None, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

		showhist : bool, optional, default: False
			Whether or not to show the histograms of each parameter in the plot.

		savefig: str or None, optional, default: None
			If None, no figure is saved. Otherwise, must be a string specifying 
			the format for the figure to be saved in, e.g., 'png', 'pdf', etc.

		dpi : str or int, optional, default: 'auto'
			If str, must be 'auto'; automatically chooses the DPI of the figure
			based on the number of parameters being plotted.
			If int, gives the DPI of the figure.

		plotendsonly : bool, optional, default: False
			Whether or not to only plot starting and ending points of the chains.

		Returns
		-------
		fig : Figure
			The figure which was plotted.
		'''
		# Most of what's below for setting up the axes was adapted from the 
		# corner source code.
		# Some magic numbers for pretty axis layout.
		K = self._nparms - 1  # -1 excludes chi^2
		factor = 2.0

		if showhist is False:
			K += -1

		lbdim = 0.2 * factor
		trdim = 0.5 * factor

		whspace = 0.05
		plotdim = factor * K + factor * (K - 1.) * whspace
		dim = lbdim + plotdim + trdim

		fig, axes = plt.subplots(K, K, figsize=(dim,dim))

		try:
			axes = np.array(fig.axes).reshape((K, K))
		except:
			raise ValueError("Provided figure has {0} axes, but data has "
                 "dimensions K={1}".format(len(fig.axes), K))

		lb = lbdim / dim
		tr = (lbdim + plotdim) / dim
		fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                			wspace=whspace, hspace=whspace)
		for i in range(0, K):
			for j in range(i+1, K):
				axes[i][j].axis('off')

		# Check/set up the list of walkers to iterate over for plotting
		if idxs is None:
			idxs = list(range(0, self.nwalkers))
		elif isinstance(idxs, (int, list, tuple, np.ndarray)):
			if isinstance(idxs, int):
				idxs = [idxs]
			else:
				# Check that each item in the list is an int
				for n in idxs:
					if not isinstance(n, int):
						raise TypeError('idxs must be an int or a list/tuple/1d numpy array of ints.')
			
			# Check that the indices are within the correct range.
			for n in idxs:
				if n > (self.nwalkers - 1):
					raise ValueError('Index {} in idxs must be less than self.nwalkers - 1 = {}.'.format(n,self.nwalkers-1))
		else:
			raise TypeError('idxs must be an int or list of ints.')

		if showhist is True:
			# Add the histograms to the grid of axes
			for i in range(0, K):
				self.plot_histogram(i, ax=axes[i][i], idxs=idxs)

		# Need to add 1 to all of the inner (i) loops below if the histograms 
		# are not being shown
		s = int(not showhist)

		# Add the walker plots to the grid of axes
		for j in range(0, K):
			for i in range(0, j+s):
				self.plot_walkers(i, j+s, ax=axes[j][i], idxs=idxs, 
								  plotendsonly=plotendsonly)

		# Remove the ylabels on all but leftmost plots
		for j in range(0, K):
			for i in range(1, j+s):
				axes[j][i].set_yticklabels([])
		# Remove the xlabels on all but bottommost plots
		for j in range(0, K-1):
			for i in range(0, j):
				axes[j][i].set_xticklabels([])

		# Set the x and y axis labels
		for i in range(0, K):
			# Set x and y axis labels
			if showhist is True:
				axes[i][0].set_ylabel(self.parm_labels_latex[i], fontsize=16)
			else: 
				axes[i][0].set_ylabel(self.parm_labels_latex[i+1], fontsize=16)
			axes[-1][i].set_xlabel(self.parm_labels_latex[i], fontsize=16)

			# Set tick label size
			axes[i][0].yaxis.set_tick_params(labelsize=14)
			axes[-1][i].xaxis.set_tick_params(labelsize=14)

			# Make the tick labels slanted
			[ax.set_rotation(45) for ax in axes[i][0].get_yticklabels()]
			[ax.set_rotation(45) for ax in axes[-1][i].get_xticklabels()]


		# Show legend
		axes[0][K-1].plot([], [], 'k.', label='Start')
		axes[0][K-1].plot([], [], 'kx', label='Stop')
		axes[0][K-1].legend(fontsize=20)

		if dpi == 'auto':
			dpi = 50 * self.nparms
		elif not isinstance(dpi, int):
			warnings.warn('dpi should be an integer or \'auto\'.')
		else:
			dpi = None

		if savefig is not None:
			plt.savefig('cornerwalkers.{}'.format(savefig), format=savefig, dpi = dpi)

		return fig


	def plot_1d_parameters(self, idxs=None, figsize=None, savefig=None):
		'''
		Plots a vertical series of plots showing each parameter as a function of
		step number.

		Parameters
		----------
		idxs : int, list, tuple, or None, optional, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

		figsize: tuple of integers, optional, default: None
			The figsize argument passed to pyplot. If None, the figsize is 
			chosen automatically based on the number of parameters being plotted.

		savefig: str, optional, default: None
			If None, no figure is saved. Otherwise, must be a string specifying 
			the format for the figure to be saved in, e.g., 'png', 'pdf', etc.
		'''
		if figsize is None:
			figsize = (10, 3*self.nparms)
		fig, ax_list = plt.subplots(self.nparms-1, 1, figsize=figsize, sharex=True)

		# Loop over each parameter (-1 excludes chi^2)
		for i in range(0, self.nparms-1):
			ax= ax_list[i]

			# Make to plot for that parameter
			self.plot_parameter(i, idxs=idxs, ax=ax)

			plt.subplots_adjust(hspace=0)

		ax.set_xlabel('Number of steps', fontsize=12)

		if savefig is not None:
			plt.savefig('parameters.{}'.format(savefig), format=savefig)


	def plot_chisq(self, nsteps=None, idxs = None, figsize=None, logscale=False):
		'''
		Plots the chi^2 for each chain as a function of step number.

		Parameters
		----------
		nsteps : int or None, optional, default: None
			The number of steps to cut off from the beginning of the chain.
		
		idxs : int, list, tuple, or None, optional, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

		figsize: tuple of integers, optional, default: None
			The figsize argument passed to pyplot.

		logscale : bool, optional, default: False
			Whether or not the y-axis is a log scale.
		'''

		fig = plt.figure(figsize = figsize)
		ax  = fig.gca()


		# Check/set up the list of walkers to iterate over for plotting
		if idxs is None:
			idxs = list(range(0, self.nwalkers))
		elif isinstance(idxs, (int, list, tuple, np.ndarray)):
			if isinstance(idxs, int):
				idxs = [idxs]
			else:
				# Check that each item in the list is an int
				for n in idxs:
					if not isinstance(n, int):
						raise TypeError('idxs must be an int or a list/tuple/1d numpy array of ints.')
			
			# Check that the indices are within the correct range.
			for n in idxs:
				if n > (self.nwalkers - 1):
					raise ValueError('Index {} in idxs must be less than self.nwalkers - 1 = {}.'.format(n,self.nwalkers-1))
		else:
			raise TypeError('idxs must be an int or list of ints.')


		for n in idxs:
			walker = self.walkers[n]
			steps = range(1, walker.shape[0]+1)
			data_chisq  = walker.iloc[:,-1].as_matrix()

			steps      = steps[nsteps:]
			data_chisq = data_chisq[nsteps:]

			# Plot the line
			ax.plot(steps, data_chisq, color=self.colors[n])

			# Mark the starting and ending points
			ax.plot(steps[0], data_chisq[0], color=self.colors[n], marker='o')    # Start
			ax.plot(steps[-1], data_chisq[-1], color=self.colors[n], marker='D')  # Stop

		ax.set_ylabel(r'$\chi^2$')
		ax.set_xlabel('Number of steps')

		if logscale:
			plt.yscale('log')

		if nsteps is not None:
			ax.set_title(r'$\chi^2$ with first {} steps cut off'.format(nsteps))
		else:
			ax.set_title(r'$\chi^2$ with no steps cut off')


	def corner(self, npoints=None, idxs=None, plot_contours=True, 
			   no_fill_contours=False, truths=None, savefig=None):
		'''
		Makes a corner plot of the data from the chains. The first npoints from 
		the beginning of each chain should be dropped for the plot to be 
		meaningful.

		Parameters
		----------
		npoints : int or None, optional, default: None
			The number of points to be dropped from the beginning of each chain 
			before plotting.

		idxs : int, list, tuple, or None, optional, default: None
 			Gives the index/indices of the walkers to be plotted. If None, all 
 			of the walkers are plotted.

 		plot_contours : bool, optional, default: True
 			The plot_contours argument passed to corner: "Draw contours for 
 			dense regions of the plot."

 		no_fill_contours : bool, optional, default: False
 			The no_fill_contours argument passed to corner: "Add no filling at 
 			all to the contours (unlike setting fill_contours=False, which still
 			adds a white fill at the densest points)."

 		truths : iterable or None, optional, default: None
 			The truths argument passed to corner: "A list of reference values 
 			to indicate on the plots. Individual values can be omitted by using 
 			None."

 		savefig: str, optional, default: None
			If None, no figure is saved. Otherwise, must be a string specifying 
			the format for the figure to be saved in, e.g., 'png', 'pdf', etc.

		Returns
		-------
		data_all : DataFrame
			A pandas DataFrame containing all of the data for all of the walkers 
			that were plotted. The initial points in each chain that were 
			removed for plotting are also removed.
		'''
		# Check/set up the list of walkers to iterate over for plotting
		if idxs is None:
			idxs = list(range(0, self.nwalkers))
		elif isinstance(idxs, (int, list, tuple, np.ndarray)):
			if isinstance(idxs, int):
				idxs = [idxs]
			else:
				# Check that each item in the list is an int
				for n in idxs:
					if not isinstance(n, int):
						raise TypeError('idxs must be an int or a list/tuple/1d numpy array of ints.')
			
			# Check that the indices are within the correct range.
			for n in idxs:
				if n > (self.nwalkers - 1):
					raise ValueError('Index {} in idxs must be less than self.nwalkers - 1 = {}.'.format(n,self.nwalkers-1))
		else:
			raise TypeError('idxs must be an int or list of ints.')


		# Get only the walkers in idxs and trim off the firth npoints points
		walkers = [self.walkers[n][npoints:] for n in idxs]

		data_all = pd.concat(walkers)

		# Drop the chi^2 column from the data
		data_all = data_all.drop('chisq', axis=1)

		labels   = self.parm_labels_latex[0:-1]

		# Plot the data
		fig = corner.corner(data_all, plot_contours=plot_contours,
					  no_fill_contours=no_fill_contours, labels=labels,
                      truths=truths, label_kwargs={'fontsize' : 16})

		title = 'Lens model posterior distribution (removed first {} points from chains)'.format(npoints)
		fig.suptitle(title, fontsize=20, y=1.02)

		if savefig is not None:
			plt.savefig('parameters.{}'.format(savefig), format=savefig)

		return data_all

	def drop_chains(self, idxs):
		'''
		Drops the chains with the given indices.

		Parameters
		----------
		idxs : int, list, or tuple
 			Gives the index/indices of the walkers to be dropped.
		'''
		# Remove the walkers (starting from the end of the list)
		for n in idxs[::-1]:
			del(self._walkers[n])

		# Update the number of walkers
		self._nwalkers = len(self._walkers)
		print('{} walkers remaining.'.format(self.nwalkers))

		# Update the color list
		colormap = plt.cm.nipy_spectral
		self._colors = [colormap(x) for x in np.linspace(0.1, 1, self.nwalkers)]


	def best_fit(self):
		'''
		Gets all of the models with the lowest chi^2.

		Returns
		-------
		mindata : DataFrame
			Pandas DataFrame containing the model(s) with the lowest chi^2. If 
			there are multiple models with the same chi^2, the DataFrame will 
			have as many rows.
		'''
		alldata = pd.concat(self.walkers)
		mindata = alldata[alldata['chisq'] == min(alldata['chisq'])]

		return mindata


	def plot_chisq_histogram(self, figsize=None):
		'''
		Plots a histogram of the lowest chi^2 value of each chain.

		Parameters
		----------
		figsize: tuple of integers, optional, default: None
			The figsize argument passed to pyplot.
		'''
		chisq_list = [walker[walker['chisq'] == min(walker['chisq'])].tail(1).values[0][-1] for walker in self.walkers]

		fig = plt.figure(figsize=figsize)
		ax  = fig.gca()

		ax.hist(chisq_list, histtype='step', color='black')
		ax.set_xlabel(r'Best-fit $\chi^2$', fontsize=12)
		ax.set_ylabel('Number of walkers', fontsize=12)

		# plt.show()