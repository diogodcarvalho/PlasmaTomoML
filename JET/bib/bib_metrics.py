
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import  compare_mse, compare_nrmse, compare_ssim, compare_psnr
from bib_geom import R_MIN,R_MAX,Z_MIN,Z_MAX

def compare_rmse_pixel(imageA,imageB):
	"""
	Calculate root mean squared error pixelwise
	Inputs:
		imageA, imageB - images to compare (can be arrays of images)
	Outputs:
		err - rmse value
	"""

	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]

	err = np.sqrt((imageA-imageB)**2)

	return err

def compare_mre_pixel(imageA,imageB):
	"""
	Calculate mean relative error pixelwise
	Inputs:
		imageA, imageB - images to compare
	Outputs:
		err - mre value
	"""
	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]

	err = (imageA-imageB)/imageA

	return np.asarray(err)

def compare_all_metrics(imageA,imageB,mean = False):
	"""
	Calculate ssim, mse, psnr and nrmse values 
	Inputs:
		imageA, imageB - images to compare (can be arrays of images)
		mean - if True mean is done over images (in the case we have an array)
	Outputs:
		ssim,mse,psnr,nrmse
	"""
	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]


	ssim  = [compare_ssim(imageA[i,:,:]/np.max(imageA[i,:,:]),imageB[i,:,:]/np.max(imageA[i,:,:])) for i in range(imageA.shape[0])]
	mse   = [compare_mse(imageA[i,:,:],imageB[i,:,:]) for i in range(imageA.shape[0])]
	psnr  = [compare_psnr(imageA[i,:,:],imageB[i,:,:],\
		data_range= np.max(imageA[i,:,:])-np.min(imageA[i,:,:])) for i in range(imageA.shape[0])]
	nrmse = [compare_nrmse(imageA[i,:,:],imageB[i,:,:]) for i in range(imageA.shape[0])]

	if mean:
		ssim = np.mean(ssim)
		mse = np.mean(mse)
		psnr = np.mean(psnr)
		nrmse = np.mean(nrmse)

	return np.asarray(ssim), np.asarray(mse), np.asarray(psnr), np.asarray(nrmse)

def compare_all_metrics_pixel(imageA,imageB,mean = False):
	"""
	Calculate rmse, amre and mre values 
	Inputs:
		imageA, imageB - images to compare (can be arrays of images)
		mean - if True mean is done over images (in the case we have an array)
	Outputs:
		rmse_pixel, amre_pixel, mre_pixel
	"""
	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]

	rmse_pixel = compare_rmse_pixel(imageA,imageB)
	mre_pixel = compare_mre_pixel(imageA,imageB)

	if mean:
		rmse_pixel = np.mean(rmse_pixel, axis = 0)
		amre_pixel = np.mean(np.abs(mre_pixel), axis = 0)
		mre_pixel = np.mean(mre_pixel, axis = 0)

	return rmse_pixel,amre_pixel,mre_pixel

def plot_metric(metric, i_train, i_valid, i_test, title, log = False):
	"""
	Plot metric values over the data-set
	Inputs:
		metric - (array) metric values
		i_train,i_valid,i_test - indeces of training/validation/test set
		title - title to give to plot
		log - if True log scale in vertical axis is used
	"""
	plt.figure()
	plt.title(title)
	plt.plot(range(len(i_train)), metric[i_train], 'r.', label = 'train')
	plt.plot(range(len(i_train), len(i_train) + len(i_valid)), metric[i_valid], 'b.', label = 'valid')
	plt.plot(range(len(i_train) + len(i_valid), len(metric)), metric[i_test], 'g.', label = 'test')	
	plt.xlabel('# Reconstruction')
	if log:
		plt.yscale('log', nonposy='clip')
	plt.legend()
	plt.savefig(title + '.png',dpi=300,bbox_inches='tight')
	plt.close()

	print title 
	print 'train : %10.3f %10.3f' % (np.mean(metric[i_train]), np.std(metric[i_train]))
	print 'valid : %10.3f %10.3f' % (np.mean(metric[i_valid]), np.std(metric[i_valid]))
	print 'test  : %10.3f %10.3f' % (np.mean(metric[i_test]), np.std(metric[i_test]))

def plot_metric_pixel(metric, title, clb_legend = None):
	"""
	Plot and generate *.png image of metric pixelwise values
	Inputs:
		metric - (2D array) metric values
		title - title to give to plot
		clb_legend - units of colorbar 
	"""
	plt.figure()
	plt.imshow(metric,vmin = np.min((0,np.min(metric))), vmax = np.max(metric), origin = 'lower', extent = [R_MIN, R_MAX, Z_MIN, Z_MAX])
	plt.title(title)
	plt.xlabel('R (m)')
	plt.ylabel('Z (m)')
	clb = plt.colorbar(format='%.2f')
	if clb_legend:
		clb.ax.set_title(clb_legend)
	plt.savefig(title + '.png',dpi=300,bbox_inches='tight')
	plt.close()


def plot_metric_pixel_multi(metric, i_train, i_valid, i_test, title, clb_legend = None):	
	"""
	Plot and generate *.png image of metric pixelwise values over the data-set
	Inputs:
		metric - (array) metric values
		i_train,i_valid,i_test - indeces of training/validation/test set
		title - title to give to plot
		log - if True log scale in vertical axis is used
	"""
	plot_metric_pixel(metric[i_train], title + '_train', clb_legend)
	plot_metric_pixel(metric[i_valid], title + '_valid', clb_legend)
	plot_metric_pixel(metric[i_test], title + '_testn', clb_legend)


