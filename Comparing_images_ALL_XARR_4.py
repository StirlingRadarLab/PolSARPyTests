# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:43:13 2023

@author: Armando
"""

import sys
sys.path.insert(0, '/home/am221/C/Programs/Python_Lib')
import os

import xarray as xr

import SAR_Utilities as sar

from pathlib import Path 
import numpy as np
import matplotlib.pyplot as plt
# import scipy.linalg as ln
# from scipy import signal
# import scipy as sp
# import spectral.io.envi as envi
import pandas as pd

import tkinter as tk

import matplotlib
matplotlib.use('Agg')   # non-GUI backend (no windows open)

plt.close('all')



#%% FILENAMES

# IN CASE YOU NEED TO HARDCODE
#################
# col = 1248
# row = 18432
# dtypePSP = '<f4' # this tells that the type is float
#################


# directoris of images
# psp are envi files
path_PSP_short = Path("/home/am221/C/Data/PolSARPy/Output_PSP")
# py are zarray files
path_Py_short =  Path("/home/am221/C/Data/PolSARPy/Output_Py")

# general directory for saving imaghes
path_save_short = Path("/home/am221/C/Data/PolSARPy/Save")



# defining window sizes for visualisation
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
dpi = 100  # dots per inch; you can adjust
    
# set global font sizes
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})




#%% SELECT THE PROCEDURE TO TEST

# # automatically finding out the algorithms to test
# dir_list = os.listdir(path_PSP_short)

# # ### TEST
dir_list = [ 
            'h_a_alpha_decomposition',
            'Yamaguchi4_Y4R',
            'Yamaguchi4_Y4O',
            'Yamaguchi4_S4R',
            'Freeman',
            'LeeRefined',
            'PWF',
            'TSVM',
            'VanZyl',
            'Cameron'
            ]

dir_list = [ 
            'Cameron',
            ]

array_pro = dir_list




#%%

def Create_Cube(path_Py, path_PSP, name_routine):
    """
    This module creates the stack of images from the routines
    path_Py_short: filepath of py files
    name_routine: selection of the routine considered
    
    """    
###########################################################
###########################################################
    # OPENING THE ZARRAY
###########################################################
###########################################################
    ds = xr.open_zarr(path_Py)

    print(ds.description)
    print(ds.data_vars)

    num_dim = len(ds.data_vars)

    array_file_Py = list(ds.data_vars)

    ds[array_file_Py[0]].shape
    col = ds[array_file_Py[0]].shape[0]
    row = ds[array_file_Py[0]].shape[1]


    num_file = len(array_file_Py)

    
    num_dim = num_file
    
    
    # adding an extra dimension for RGB in case we have less than 2 images
    if num_file == 2 : 
        num_dim = 3
        array_file_Py.append('empty')

    elif num_file == 1 : 
        num_dim = 3
        array_file_Py.append('empty')
        array_file_Py.append('empty')

        
    imgPy  = np.zeros([col, row, num_dim], dtype=np.complex64)


################### LEE REFINED #################    
    if name_routine == "LeeRefined":
    
        imgPy[:, :, 0] = ds['m11'].values
        imgPy[:, :, 1] = ds['m22'].values
        imgPy[:, :, 2] = ds['m33'].values
        imgPy[:, :, 3] = ds['m12'].values
        imgPy[:, :, 4] = ds['m13'].values
        imgPy[:, :, 5] = ds['m23'].values
        array_file_Py = ['m11', 'm22', 'm33', 'm12', 'm13', 'm23']

################### YAMAGUCHI #################
    elif name_routine in ("Yamaguchi4_Y4R", "Yamaguchi4_Y4O", "Yamaguchi4_S4R"):
    
        imgPy[:, :, 0] = ds['odd'].values
        imgPy[:, :, 1] = ds['double'].values
        imgPy[:, :, 2] = ds['volume'].values
        imgPy[:, :, 3] = ds['helix'].values
        array_file_Py = ['odd', 'double', 'volume', 'helix']

#################### PWF ###########################
    elif name_routine == "PWF":
        imgPy[:, :, 0] = ds['pwf'].values

################### FREEMAN3 #################
    elif name_routine == "Freeman":
    
        imgPy[:, :, 0] = ds['odd'].values
        imgPy[:, :, 1] = ds['double'].values
        imgPy[:, :, 2] = ds['volume'].values
        array_file_Py = ['odd', 'double', 'volume']

################### VanZyl #################
    elif name_routine == "VanZyl":
    
        imgPy[:, :, 0] = ds['odd'].values
        imgPy[:, :, 1] = ds['double'].values
        imgPy[:, :, 2] = ds['volume'].values
        array_file_Py = ['odd', 'double', 'volume']
        
################### TSVM #################
    elif name_routine == "TSVM":
    
        imgPy[:, :, 0] = ds['alpha_s'].values
        imgPy[:, :, 1] = ds['alpha_s1'].values
        imgPy[:, :, 2] = ds['alpha_s2'].values
        imgPy[:, :, 3] = ds['alpha_s3'].values
        imgPy[:, :, 4] = ds['phi_s'].values
        imgPy[:, :, 5] = ds['tau_m'].values
        imgPy[:, :, 6] = ds['psi'].values

        array_file_Py = ['alpha_s', 'alpha_s1', 'alpha_s2', 'alpha_s3',
                         'phi_s', 'tau_m', 'psi',]
        
#################### PWF ###########################
    elif name_routine == "Cameron":
        imgPy[:, :, 0] = ds['cameron'].values
                
###########################################################
###########################################################
    # OPENING THE POLSARPRO    
###########################################################
###########################################################
    # path_PSP = path_PSP_short / name_routine
    
    # get the filenames
    array_file_PSP = os.listdir(path_PSP)
    array_file_PSP = [f[:-4] for f in array_file_PSP if f.endswith('.bin')]
    

    # read the config file·
    with open(path_PSP / "config.txt", "r") as f:
        lines = f.read().splitlines()
        
    for l, line in enumerate(lines):
        if line.strip() == "Nrow":
            col = int(lines[l + 1].strip())
        elif line.strip() == "Ncol":
            row = int(lines[l + 1].strip())
            
            
    # adding an extra dimension for RGB in case we have less than 2 images
    # empty = np.zeros([col,row])
    if num_file == 2 : 
        num_dim = 3
        array_file_PSP.append('empty')

    elif num_file == 1 : 
        num_dim = 3
        array_file_PSP.append('empty')
        array_file_PSP.append('empty')


    #%% procesing the single images
    imgPSP = np.zeros((col, row, num_dim), dtype=np.complex64)
    
    
#############################################################################
#####################   LEE REFINED  ########################################
    if name_routine == "LeeRefined":

        img_temp = Open_PSP_Image(path_PSP / 'T11.bin', col, row, dtypePSP)
        imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))

        img_temp = Open_PSP_Image(path_PSP / 'T22.bin', col, row, dtypePSP)
        imgPSP[:,:,1] = np.transpose(np.nan_to_num(img_temp))

        img_temp = Open_PSP_Image(path_PSP / 'T33.bin', col, row, dtypePSP)
        imgPSP[:,:,2] = np.transpose(np.nan_to_num(img_temp))

        img_temp1 = Open_PSP_Image(path_PSP / 'T12_real.bin', col, row, dtypePSP)
        img_temp1 = np.transpose(np.nan_to_num(img_temp1))
        img_temp2 = Open_PSP_Image(path_PSP / 'T12_imag.bin', col, row, dtypePSP)
        img_temp2 = np.transpose(np.nan_to_num(img_temp2))
        imgPSP[:,:,3] = img_temp1 + 1j*img_temp2
        
        img_temp1 = Open_PSP_Image(path_PSP / 'T13_real.bin', col, row, dtypePSP)
        img_temp1 = np.transpose(np.nan_to_num(img_temp1))
        img_temp2 = Open_PSP_Image(path_PSP / 'T13_imag.bin', col, row, dtypePSP)
        img_temp2 = np.transpose(np.nan_to_num(img_temp2))
        imgPSP[:,:,4] = img_temp1 + 1j*img_temp2

        img_temp1 = Open_PSP_Image(path_PSP / 'T23_real.bin', col, row, dtypePSP)
        img_temp1 = np.transpose(np.nan_to_num(img_temp1))
        img_temp2 = Open_PSP_Image(path_PSP / 'T23_imag.bin', col, row, dtypePSP)
        img_temp2 = np.transpose(np.nan_to_num(img_temp2))
        imgPSP[:,:,5] = img_temp1 + 1j*img_temp2
        array_file_PSP = ['T11', 'T22', 'T33', 'T12', 'T13', 'T23']


################### YAMAGUCHI #################
    elif name_routine in ("Yamaguchi4_Y4R", "Yamaguchi4_Y4O", "Yamaguchi4_S4R"):
        img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Odd.bin'), col, row, dtypePSP)
        imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))

        img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Dbl.bin'), col, row, dtypePSP)
        imgPSP[:,:,1] = np.transpose(np.nan_to_num(img_temp))

        img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Vol.bin'), col, row, dtypePSP)
        imgPSP[:,:,2] = np.transpose(np.nan_to_num(img_temp))

        img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Hlx.bin'), col, row, dtypePSP)
        imgPSP[:,:,3] = np.transpose(np.nan_to_num(img_temp))
        if name_routine == "Yamaguchi4_Y4R":
            array_file_PSP = ['Y4R_Odd', 'Y4R_Dbl', 'Y4R_Vol', 'Y4R_Hlx']
        elif name_routine == "Yamaguchi4_Y4O":
            array_file_PSP = ['Y4O_Odd', 'Y4O_Dbl', 'Y4O_Vol', 'Y4O_Hlx']
        elif name_routine == "Yamaguchi4_S4R":
            array_file_PSP = ['S4R_Odd', 'S4R_Dbl', 'S4R_Vol', 'S4R_Hlx']

################### PWF #################
    elif name_routine == "PWF":
        img_temp = Open_PSP_Image(path_PSP / ('PWF.bin'), col, row, dtypePSP)
        imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))


################### FREEMAN3 #################
    elif name_routine == "Freeman":
       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Odd.bin'), col, row, dtypePSP)
       imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Dbl.bin'), col, row, dtypePSP)
       imgPSP[:,:,1] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_Vol.bin'), col, row, dtypePSP)
       imgPSP[:,:,2] = np.transpose(np.nan_to_num(img_temp))

       array_file_PSP = ['F3_Odd', 'F3_Dbl', 'F3_Vol']        


################### VanZyl #################
    elif name_routine == "VanZyl":
       img_temp = Open_PSP_Image(path_PSP / (name_routine + '3_Odd.bin'), col, row, dtypePSP)
       imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '3_Dbl.bin'), col, row, dtypePSP)
       imgPSP[:,:,1] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '3_Vol.bin'), col, row, dtypePSP)
       imgPSP[:,:,2] = np.transpose(np.nan_to_num(img_temp))

       array_file_PSP = ['VZ_Odd', 'VZ_Dbl', 'VZ_Vol']     
       
       
################### TSVM #################
    elif name_routine == "TSVM":
       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_alpha_s.bin'), col, row, dtypePSP)
       imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_alpha_s1.bin'), col, row, dtypePSP)
       imgPSP[:,:,1] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_alpha_s2.bin'), col, row, dtypePSP)
       imgPSP[:,:,2] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_alpha_s3.bin'), col, row, dtypePSP)
       imgPSP[:,:,3] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_phi_s.bin'), col, row, dtypePSP)
       imgPSP[:,:,4] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_tau_m.bin'), col, row, dtypePSP)
       imgPSP[:,:,5] = np.transpose(np.nan_to_num(img_temp))

       img_temp = Open_PSP_Image(path_PSP / (name_routine + '_psi.bin'), col, row, dtypePSP)
       imgPSP[:,:,6] = np.transpose(np.nan_to_num(img_temp))

       array_file_PSP = ['TSVM_alpha_s', 'TSVM_alpha_s1', 'TSVM_alpha_s2', 'TSVM_alpha_s3',
                         'TSVM_phi_s', 'TSVM_tau_m', 'TSVM_psi']        

################### PWF #################
    elif name_routine == "Cameron":
        img_temp = Open_PSP_Image(path_PSP / ('Cameron.bin'), col, row, dtypePSP)
        imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))

        
    return imgPy, imgPSP, array_file_Py, array_file_PSP 


       

#%%
def Open_PSP_Image(filename, col, row, dtype):
    """
    This module open binary images 
    The parameters of the functions can be read in the ENVI header file .hdr
    col: number of columns in the image, if left empty takes the values directly from the header
    row: number of raw in the image, if left empty takes the values directly from the header
    dtype: date type, if left empty takes the values directly from the header 
    """
# #     # UNCOMMENT the following 3 lines if you manage to install Spectral 
#     lib = envi.open(str(filename) + '.hdr')
#     header =  np.array([lib.ncols, lib.nrows])
#     datatype = lib.dtype

    # COMMENT the following 2 lines if you manage to install Spectral
    header =  np.array([row, col])
    datatype = dtype
    
    # Opening the first image
    f = open(filename / '', 'rb')
    img = np.fromfile(f, dtype=datatype)
    
    # The following line is to define the order of pixels in the image.
    # 'C' exploits the order format used in C
    # 'F' exploits the order format used in Fortran
    # If you processed the images in C you can use the following line. If you run this turorial again 
    # using  SNAP with Window the order is normally 'F'. If you see that your images look "scrambled"
    # try using the other format
#     img = img.reshape(header,order='C').astype(dtype)
    img = img.reshape(header,order='F').astype(dtype)

    return(img)



#%%
def plot_RGB(T22,T33,T11, title, fact = 1.5, path_out = ''):
    # First we need to create a 3D container of the image
    sizeI = np.shape(T11)            # this evaluates the size of each image
    dim1 = sizeI[0]       
    dim2 = sizeI[1]   
    iRGB = np.zeros([dim1, dim2, 3])    # Create an empty 3D array (full of zeros) 

    # It is important to scale the image "grey levels" properly so that we can see details.
    # We can therefore decide a scaling factor that can be used to improve the contrast 
    # fact = 1.5       # experiment with different values as well 

    # The R (red) colour 
    iRGB[:,:,0] = np.abs(T22)/(np.abs(T22).mean()*fact)
    # the G (green) colour
    iRGB[:,:,1] = np.abs(T33)/(np.abs(T33).mean()*fact)
    # the Blue colour
    iRGB[:,:,2] = np.abs(T11)/(np.abs(T11).mean()*fact)
    # abs is for the magnitude. mean() calculates the mean of the image is pointing to.

    # In case the pixels in the contained are above the scale (saturated) we need to set them to the maximum 
    # value 1. Otherwise larger values can create artifacts when using the imshow function.
    iRGB[np.abs(iRGB) > 1] = 1

    # When making a new figure, we first need to open an empty "figure object".
    # We can use the matplotlib.pyplot library to create an object called fig. 
    fig = plt.figure()      # we number this figure "1"

    plt.title(title)      # this defines the title 

    # To show the RGB inside this figure we can use the imshow function.
    plt.imshow(iRGB)
    
    if path_out != '':
        fig.savefig(path_out, bbox_inches='tight', pad_inches=0)
        


def vis4(img1, img2, img3, img4, 
         title1 = 'Image 1', 
         title2 = 'Image 2',
         title3 = 'Image 3', 
         title4 = 'Image 4', 
         scale1 = [], 
         scale2 = [],
         scale3 = [], 
         scale4 = [],  
         path_out = '', 
         outall = [],
         colormap1 = 'gray',
         colormap2 = 'gray',
         colormap3 = 'gray',
         colormap4 = 'gray'):
    # Visualise any two images. We need to tell it the scaling or it uses the 
    # default for magnitude images
    if scale1 == []:
       scale1 = (0, np.abs(img1).mean()*2)
    if scale2 == []:
       scale2 = (0, np.abs(img2).mean()*2)
    if scale3 == []:
       scale3 = (0, np.abs(img3).mean()*2)
    if scale4 == []:
       scale4 = (0, np.abs(img4).mean()*2)

    fig, [(ax1, ax2), (ay1, ay2)] = plt.subplots(2, 2, sharex=True, sharey=True)
    figManager = plt.get_current_fig_manager()
    figManager.full_screen_toggle()
    ax1.set_title(title1)
    cax1 = ax1.imshow(img1, cmap = colormap1, vmin=scale1[0], vmax=scale1[1])
    plt.colorbar(cax1, ax=ax1)
    ax2.set_title(title2)
    cax2 = ax2.imshow(img2, cmap = colormap2, vmin=scale2[0], vmax=scale2[1])
    plt.colorbar(cax2, ax=ax2)
    ay1.set_title(title3)
    cax3 = ay1.imshow(img3, cmap = colormap3, vmin=scale3[0], vmax=scale3[1])
    plt.colorbar(cax3, ax=ay1)
    ay2.set_title(title4)
    cax4 = ay2.imshow(img4, cmap = colormap4, vmin=scale4[0], vmax=scale4[1])
    plt.colorbar(cax4, ax=ay2)
    
    if path_out != '':
        fig.savefig(path_out)
#    if flag[1] == 1:
#        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig.savefig(out1, bbox_inches=extent)

    return(fig)







#%% OPENING IMAGES



# OPENING PSP IMAGES 
dtypePSP = '<f4' # this tells that the type is float

num_pro = len(array_pro)

for i in range(num_pro):
    
    # cleaning from the previous figures
    plt.close()

    
    # Path for saving files
    path_save = path_save_short / ('Results_' + array_pro[i])
    os.makedirs(path_save, exist_ok=True)
    
    
    print('Processing ' + array_pro[i] + '...')


    path_Py = path_Py_short/array_pro[i]
    path_PSP = path_PSP_short/array_pro[i]
    [imgPy_full, 
     imgPSP_full, 
     array_file_Py, 
     array_file_PSP ] = Create_Cube(path_Py, 
                                    path_PSP, 
                                    array_pro[i])
    
    ###############################################
    ######   FOR SMALLER PORTIONS #################
    flag_area = "small"
    flag_area = "full"
    
    if flag_area == "small":
        imgPy = imgPy_full[100:1100, 100:11000, :]                                    
        imgPSP = imgPSP_full[100:1100, 100:11000, :]    
    else:
        imgPy = imgPy_full          
        imgPSP = imgPSP_full
    
    num_file = len(array_file_Py)

                    
    # ploting the RGB for the elements
    plot_RGB(imgPSP[:,:,1], imgPSP[:,:,2], imgPSP[:,:,0], 
             str(array_pro[i]) + ' RGB: PolSARpro', fact = 1.5, 
             path_out =path_save / 'RGB_PSP')
    plot_RGB(imgPy[:,:,1], imgPy[:,:,2], imgPy[:,:,0], 
             str(array_pro[i]) + ' RGB: Python', fact = 1.5, 
             path_out = path_save / 'RGB_Py')

    plt.figure()
    plt.imshow(np.abs(imgPSP[:,:,0]), vmin = 0, vmax= 5*np.nanmean(np.abs(imgPSP[:,:,0])))

    plt.figure()
    plt.imshow(np.abs(imgPy[:,:,0]), vmin = 0, vmax= 5*np.nanmean(np.abs(imgPy[:,:,0])))


#%% in the following one can select the images to visualise individually, 
# this may crush when running all together and it is suggested to have iii_select = []
    iii_select = [0, 1, 2]
    colormap = ['gray', 'gray', 'gray', 'gray', 'gray']
    
    for iii in iii_select: 
    
        fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
        scale = (0, np.abs(imgPy[:,:,iii]).mean()*2)
        plt.title('Python ' + str(array_pro[i]) + ': ' + array_file_PSP[iii]) 
        scale2 = (0, np.abs(imgPy[:,:,0]).mean()*2)
        im = plt.imshow(np.abs(imgPy[:,:,iii]), cmap = colormap[iii], vmin=scale[0], vmax=scale[1])
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.savefig(path_save / ('Py_' + array_file_Py[iii]), bbox_inches='tight', pad_inches=0)


        fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
        scale = (0, np.abs(imgPSP[:,:,iii]).mean()*2)
        plt.title('PolSARPro ' + str(array_pro[i]) + ': ' + array_file_PSP[iii]) 
        scale2 = (0, np.abs(imgPSP[:,:,0]).mean()*2)
        im = plt.imshow(np.abs(imgPSP[:,:,iii]), cmap = colormap[iii], vmin=scale[0], vmax=scale[1])
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.savefig(path_save / ('PSP_' + array_file_PSP[iii]), bbox_inches='tight', pad_inches=0)
        
        plt.close(fig)


    #%% calculating the differences    
    ##################################################################
    Diff = imgPSP - imgPy

    # plotting differences
    plot_RGB(Diff[:,:,1].real, Diff[:,:,2].real, Diff[:,:,0].real, 'RGB: Difference images')
    
    # sar.vis4(Diff[:,:,0].real, Diff[:,:,1].real, Diff[:,:,2].real, Diff[:,:,3].real, 
    #          title1 = 'Difference Comp1', 
    #          title2 = 'Difference Comp2',
    #          title3 = 'Difference Comp3',
    #          title4 = 'Difference Comp4', 
    #          scale1 = [0, np.mean(abs(Diff[:,:,0]))], 
    #          scale2 = [0, np.mean(abs(Diff[:,:,1]))],
    #          scale3 = [0, np.mean(abs(Diff[:,:,2]))], 
    #          scale4 = [0, np.mean(abs(Diff[:,:,3]))],  
    #          flag = 1, 
    #          outall = path_save / ('Diff_' + array_file_PSP[iii]),
    #          colormap1 = 'jet',
    #          colormap2 = 'jet',
    #          colormap3 = 'jet',
    #          colormap4 = 'gray')



    imgPSP_mean = np.zeros([num_file])
    imgPy_mean = np.zeros([num_file])
    Diff_mean = np.zeros([num_file])
    RMSE_mean = np.zeros([num_file])
    Norm_RMSE_mean = np.zeros([num_file])
    Norm_diff_mean = np.zeros([num_file])
    Norm_abs_diff_mean = np.zeros([num_file])

    #%% Plotting histograms
    ##################################################################
    min_freq = 1000
    
    for ii in range(num_file):

        # histogram of images
        Py1D = np.ravel(np.abs(imgPy[:,:,ii]))
        p9999 = np.percentile(Py1D, 99.99)
        Py1D = Py1D[Py1D <= p9999]
        hist = np.histogram(Py1D, bins=100)
        val = hist[1]
        fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
        plt.title('Histogram of ' + str(array_file_PSP[ii]) + ' Python')
        plt.plot(val[:-1], hist[0])
        plt.axis([np.min(hist[1]), p9999, 0, min_freq])
        plt.savefig(path_save / ('Histogram_' + str(array_file_PSP[ii]) + '_Python'))
    
        PSP1D = np.ravel(np.abs(imgPSP[:,:,ii]))
        p9999 = np.percentile(PSP1D, 99.99)
        PSP1D = PSP1D[PSP1D <= p9999]
        hist = np.histogram(PSP1D, bins=100)
        val = hist[1]
        fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
        plt.title('Histogram of ' + str(array_file_PSP[ii]) + ' POLSARpro')
        plt.plot(val[:-1], hist[0])
        plt.axis([np.min(hist[1]), np.max(hist[1]), 0, min_freq])
        plt.savefig(path_save / ('Histogram_' + str(array_file_PSP[ii]) + '_POLSARpro'))
        
        # histogram of differences
        Diff1D = np.ravel(Diff[:,:,ii])
        Diff1D = np.clip(Diff1D, np.nanmin(Diff1D), np.nanmax(Diff1D))
        Diff1D = Diff1D[np.isfinite(Diff1D)]
        hist = np.histogram(Diff1D, bins=100)
        # hist = np.histogram(Diff1D, bins=100)
        val = hist[1]
        fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
        plt.title('Histogram of DIFFERENCE ' + str(array_file_PSP[ii]) + ' Python')
        plt.plot(val[:-1], hist[0])
        plt.axis([np.min(hist[1]), np.max(hist[1]), 0, min_freq])
        plt.savefig(path_save / ('Histogram_DIFFERENCE_' + str(array_file_PSP[ii]) + '_Python'))      

        plt.close(fig)

#%% Mean difference
    ##################################################################

        imgPSP_mean[ii] = np.nanmean(np.abs(imgPSP[:,:, ii]))
        imgPy_mean[ii]  = np.nanmean(np.abs(imgPy[:,:, ii]))
    
        Diff_mean[ii] = np.nanmean(np.abs(Diff[:,:,ii]))
    
        RMSE_mean[ii] = np.sqrt(np.nanmean( (imgPSP[:,:,ii] - imgPy[:,:,ii])**2 ) )
    
        Norm_RMSE_mean[ii] = np.sqrt(np.nanmean( (imgPSP[:,:,ii] - imgPy[:,:,ii])**2 / (imgPSP[:,:,ii] + imgPy[:,:,ii])**2 ) )
    
        Norm_diff_mean[ii] = np.nanmean( (imgPSP[:,:,ii] - imgPy[:,:,ii]) / (imgPSP[:,:,ii] + imgPy[:,:,ii]) ) 
    
        Norm_abs_diff_mean[ii] = np.nanmean( np.abs(imgPSP[:,:,ii] - imgPy[:,:,ii]) / (imgPSP[:,:,ii] + imgPy[:,:,ii]) )


    # # Normalising the error
    # Norm_Ground_RMSE = Ground_RMSE/np.mean(GroundPSP) 
    # Norm_Volume_RMSE = Volume_RMSE/np.mean(VolumePSP) 


    for ii in range(num_file):

        # Diff_array = [GroundD_mean, VolumeD_mean, GroundD_std, VolumeD_std, Ground_RMSE, Volume_RMSE]
        Diff_array = [RMSE_mean[ii], Norm_diff_mean[ii] ]
        xaxis_array = [str(array_file_PSP[ii])+' RMSE', 
                       str(array_file_PSP[ii])+' NDI',
                       ]

 
    fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
    plt.title(str(array_pro[i]) + ': Mean of Differences')
    plt.plot(xaxis_array, Diff_array)
    fig.savefig(path_save / 'Mean_Differences')          

    plt.close(fig)

    # Save the figure in full screen
    # fig.savefig('C:\\MyC\\Funds\\ESA\\PolSARpy\\Reports\\Code test\\full_screen_figure.png', bbox_inches='tight')


    #%% Percentile
    ############################################
    
    Diff_Perc50 = np.zeros([num_file])
    Diff_Perc90 = np.zeros([num_file])
    Diff_Perc99 = np.zeros([num_file])
    Perc_array = np.empty((0,))
    Perc_Norm_array = np.empty((0,))
    xaxis_array = np.empty((0,))

    for ii in range(num_file):
    
        Diff_Perc50[ii] = np.percentile(np.abs(Diff[:,:,ii]), 50)
        Diff_Perc90[ii] = np.percentile(np.abs(Diff[:,:,ii]), 90)
        Diff_Perc99[ii] = np.percentile(np.abs(Diff[:,:,ii]), 99)
        
    
        Perc_array = np.append(Perc_array, [Diff_Perc50[ii], Diff_Perc90[ii], 
                                            Diff_Perc99[ii]] )
        Perc_Norm_array = np.append(Perc_Norm_array, [Diff_Perc50[ii]/imgPy_mean[ii], 
                                                      Diff_Perc90[ii]/imgPy_mean[ii], 
                                                      Diff_Perc99[ii]/imgPy_mean[ii]] )

        xaxis_array = np.append(xaxis_array, [str(array_file_PSP[ii])+' 50th', str(array_file_PSP[ii])+' 90th', str(array_file_PSP[ii])+' 99th'])

        # Perc_Norm_array = [Diff_Perc50[0]/imgPy_mean[0], Diff_Perc90[0]/imgPy_mean[0], 
        #                    Diff_Perc99[0]/imgPy_mean[0],
        #                    Diff_Perc50[1]/imgPy_mean[1], Diff_Perc90[1]/imgPy_mean[1], 
        #                    Diff_Perc99[1]/imgPy_mean[1], 
        #                    Diff_Perc50[2]/imgPy_mean[2], Diff_Perc90[2]/imgPy_mean[2], 
        #                    Diff_Perc99[2]/imgPy_mean[2]]                    
        # xaxis_array = [str(array_file[0])+' 50th', str(array_file[0])+' 90th', str(array_file[0])+' 99th', 
        #                str(array_file[1])+' 50th', str(array_file[1])+' 90th', str(array_file[1])+' 99th', 
        #                str(array_file[2])+' 50th', str(array_file[2])+' 90th', str(array_file[2])+' 99th']

    fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
    # figManager = plt.get_current_fig_manager()
    # figManager.full_screen_toggle()
    plt.title(str(array_pro[i]) + ': Percentiles')
    plt.plot(xaxis_array, Perc_array)
    plt.xticks(rotation=45)  # ← rotate x-axis labels by 45 degrees
    fig.savefig(path_save / 'Percentiles')          
    
    
    fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
    # figManager = plt.get_current_fig_manager()
    # figManager.full_screen_toggle()
    plt.title(str(array_pro[i]) + ': Normalised Percentiles (over mean)')
    plt.plot(xaxis_array, Perc_Norm_array)
    plt.xticks(rotation=45)  # ← rotate x-axis labels by 45 degrees
    fig.savefig(path_save /'Normalised_Percentiles_(over_mean)')     

    plt.close(fig)



    #%% saving all in excell
    ##########################################
    mean = {}
    perc = {}
    perc_norm = {}
    diverg = {}
    # adding items to the dictionary using a loop
    for ii in range(num_file):
        
        mean.update( {str(array_file_PSP[ii])+' PSP mean': [imgPSP_mean[ii]], 
                      str(array_file_PSP[ii])+' Py mean': [imgPy_mean[ii]]
                     } )
        
        diverg.update( {str(array_file_PSP[ii])+' Diff mean': [Diff_mean[ii]], 
                        str(array_file_PSP[ii])+' RMSE mean': [RMSE_mean[ii]],
                        str(array_file_PSP[ii])+' Norm RMSE mean': [Norm_RMSE_mean[ii]],
                        str(array_file_PSP[ii])+' Norm diff mean': [Norm_diff_mean[ii]],
                        str(array_file_PSP[ii])+' Norm abs diff mean': [Norm_abs_diff_mean[ii]]
                   } )
        
        perc.update( {str(array_file_PSP[ii])+' 50th': [Diff_Perc50[ii]], 
                      str(array_file_PSP[ii])+' 90th': [Diff_Perc90[ii]],
                      str(array_file_PSP[ii])+' 99th': [Diff_Perc99[ii]]
                     } )

        perc_norm.update( {str(array_file_PSP[ii])+' 50th': [Diff_Perc50[ii]/imgPy_mean[ii]], 
                           str(array_file_PSP[ii])+' 90th': [Diff_Perc90[ii]/imgPy_mean[ii]], 
                           str(array_file_PSP[ii])+' 99th': [Diff_Perc99[ii]/imgPy_mean[ii]], 
                   } )        
        

    df_mean   = pd.DataFrame(data=mean)
    df_diverg = pd.DataFrame(data=diverg)
    df_perc   = pd.DataFrame(data=perc)
    df_perc_norm = pd.DataFrame(data=perc_norm)


    # path = 'C:\\MyC\\Funds\\ESA\\PolSARpy\\Reports\\Code test\\'
    df_mean.to_excel(path_save / (str(array_pro[i]) + '_stats.xlsx'), sheet_name='Mean') 
    
    with pd.ExcelWriter(path_save / (str(array_pro[i]) + '_stats.xlsx'),
                    mode='a') as writer:  
        df_diverg.to_excel(writer, sheet_name='Divergences')
        df_perc.to_excel(writer, sheet_name='Percentile')
        df_perc_norm.to_excel(writer, sheet_name='Normalised Percentile')
    
    
# ---------------------------------------------------------
# SAVE EACH SHEET AS ITS OWN CSV
# ---------------------------------------------------------
        
    def transpose_preserve_columns(df, sigfig=4):
        """
        Transpose a dataframe, preserve original column names as first column,
        replace underscores with dots in the first column, 
        and round numeric values to a given number of significant figures.
        """
        # Transpose
        df_transposed = df.T.copy()
        
        # Insert original column names as first column
        df_transposed.insert(0, "Component", df.columns)
        
        # Replace underscores with dots in the first column
        df_transposed["Component"] = df_transposed["Component"].str.replace("_", ".", regex=False)
        
        # Rename second column header to "Values"
        df_transposed.columns.values[1] = "Values"
        
        # Format numeric values to N significant figures
        def format_sig(x):
            try:
                return float(f"{x:.{sigfig}g}")  # 4 significant figures
            except (ValueError, TypeError):
                return x  # leave non-numeric as-is
        
        # Apply formatting to numeric columns only
        df_transposed.iloc[:, 1:] = df_transposed.iloc[:, 1:].applymap(format_sig)
        
        return df_transposed
    
    # Example usage
    dfs = {
        "Mean": df_mean,
        "Divergences": df_diverg,
        "Percentile": df_perc,
        "NormalisedPercentile": df_perc_norm
    }
    
    base_name = f"{array_pro[i]}_stats"
    
    for name, df in dfs.items():
        df_transposed = transpose_preserve_columns(df, sigfig=4)
        df_transposed.to_csv(path_save / f"{base_name}_{name}.csv", index=False)