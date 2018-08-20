from skimage.measure import  compare_mse, compare_nrmse, compare_ssim, compare_psnr
from bib_geom import R_MIN,R_MAX,Z_MIN,Z_MAX

import matplotlib.pyplot as plt
import numpy as np

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	mse = compare_mse(imageA, imageB)
	nrmse = compare_nrmse(imageA, imageB)
	psnr = compare_psnr(imageA, imageB, data_range= np.max(imageA)-np.min(imageA))
	ssim,_ = compare_ssim(imageA, imageB,  full=True)
 
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f, NRMSE: %.2f, PSNR: %.2f" % (mse, ssim, nrmse, psnr))
 
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, origin = 'lower', vmin = 0, vmax = np.max(imageA), cmap = 'inferno')
	plt.axis("off")
 
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, origin = 'lower', vmin = 0, vmax = np.max(imageA), cmap = 'inferno')
	plt.axis("off")
	
	# show the images
	plt.savefig(title + '.png')
	#plt.show()

def compare_mae_pixel(imageA, imageB):

	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]

	err = np.absolute(imageA-imageB)

	return err

def compare_rmse_pixel(imageA,imageB):

	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]

	err = np.sqrt((imageA-imageB)**2)

	return err

def compare_mre_pixel(imageA,imageB):

	if imageA.ndim == 2:
		imageA = imageA[np.newaxis,:]
		imageB = imageB[np.newaxis,:]

	err = (imageA-imageB)/imageA

	return np.asarray(err)

def compare_all_metrics(imageA,imageB,mean = False):

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

	plot_metric_pixel(metric[i_train], title + '_train', clb_legend)
	plot_metric_pixel(metric[i_valid], title + '_valid', clb_legend)
	plot_metric_pixel(metric[i_test], title + '_testn', clb_legend)
	
# ----------------------------------------------------------------------------
# UNDER WORK

# skimage.measure already has this ones implementd
#
# def compare_mse(imageA, imageB):
# 	err = np.sum((imageA - imageB)** 2)
# 	err /= float(imageA.shape[0] * imageA.shape[1])
# 	return err

# def compare_nmse(imageA,imageB):
# 	err = compare_mse(imageA,imageB)
# 	err /= (np.mean(imageA)*np.mean(imageB))
# 	return err

# def compare_psnr(imageA, imageB):

# 	err = 20*np.log10(np.max(imageA)/np.sqrt(compare_mse(imageA,imageB)))
# 	return err