# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 17:43:13 2023

@author: Armando
"""

import sys
sys.path.insert(0, 'C:\\MyC\\Programs\\Python_Lib\\')
import os

import SAR_Utilities as sar

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln
from scipy import signal
import scipy as sp
#import spectral.io.envi as envi
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

# import seaborn as sns


plt.close('all')





#%%
def Open_ENVI_Image(filename, col, row, dtype):
    """
    This module open binary images 
    The parameters of the functions can be read in the ENVI header file .hdr
    col: number of columns in the image, if left empty takes the values directly from the header
    row: number of raw in the image, if left empty takes the values directly from the header
    dtype: date type, if left empty takes the values directly from the header 
    """
#     # UNCOMMENT the following 3 lines if you manage to install Spectral 
#     lib = envi.open(filename + '.hdr')
#     header =  np.array([lib.ncols, lib.nrows])
#     datatype = lib.dtype

    # COMMENT the following 2 lines if you manage to install Spectral
    header =  np.array([col, row])
    datatype = dtype
    
    # Opening the first image
    f = open(filename + '', 'rb')
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
def Load_One_Date(path, col, row, dtype):
    """
    This module opens the elements of a coherency matrix for a single acquisition.
    INPUT
    path: a string containing the path to the folder where images are contained 
    date: a string containing the date of the acquisition we want to open
    OUTPUT 
    The elements of the coherency matrix stored as separate images
    """
    
    ############### Loading T11 ######################################
    # First we need to identify the name of the file for the HH image 
    fileT11 = "T11"
    T11Full = Open_ENVI_Image(path + fileT11, col, row, dtype) 
    # Full stand for "Entire image"
    # Notice I am calling a function by writing its name and passing the parameters separated by a comma.
    # Always make sure that the order of parameters is consistent with your definition of the function done above. 
    
    ############### Loading T22 ############################
    fileT22 = "T22"
    T22Full = Open_ENVI_Image(path + fileT22, col, row, dtype) 
    
    ############### Loading T33 ############################
    fileT33 = "T33"
    T33Full = Open_ENVI_Image(path + fileT33, col, row, dtype) 
    
    
    
    ############### Loading T12 ############################
    # The off diagonal terms are cross correlation and therefore they are complex
    # numbers. They are stored by SNAP as Real and Imaginay parts. Both are floats
    fileT12_real = "T12_real"
    T12Full_real = Open_ENVI_Image(path + fileT12_real, col, row, dtype) 
    fileT12_imag = "T12_imag"
    T12Full_imag = Open_ENVI_Image(path + fileT12_imag, col, row, dtype) 
    
    # We can now put the Real and Imaginary parts togeter to form the complex number
    T12Full = T12Full_real + 1j*T12Full_imag
    # Since the Real and Imaginary part on their own are redundant, we can remove 
    # them from the RAM memory
    del T12Full_real, T12Full_imag
    
    
    ############### Loading T13 ############################
    fileT13_real = "T13_real"
    T13Full_real = Open_ENVI_Image(path + fileT13_real, col, row, dtype) 
    fileT13_imag = "T13_imag"
    T13Full_imag = Open_ENVI_Image(path + fileT13_imag, col, row, dtype) 
    
    T13Full = T13Full_real + 1j*T13Full_imag
    del T13Full_real, T13Full_imag
    
    
    ############### Loading T23 ############################
    fileT23_real = "T23_real"
    T23Full_real = Open_ENVI_Image(path + fileT23_real, col, row, dtype) 
    fileT23_imag = "T23_imag"
    T23Full_imag = Open_ENVI_Image(path + fileT23_imag, col, row, dtype) 
    
    T23Full = T23Full_real + 1j*T23Full_imag
    del T23Full_real, T23Full_imag
    

    return T11Full, T22Full, T33Full, T12Full, T13Full, T23Full


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
        fig.savefig(path_out)
        


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

#%%

# the following are for the RCM image we are using in this practical. If you reprocess the data and chnage the size of the subset, 
# you will need to modify those numbers the looking at the ENVI text header file {.hdr}. 
# You can open this using any text editor
col = 1248
row = 18432

dtypePPP = '<f4' # this tells that the type is float
dtypePy = '<f4' # this tells that the type is float
# > indicates they are bigger endians

# flag_image = 'PPP'
# flag_image = 'Py'

path_PPP_short = "C:\\MyC\\Data\\PolSARProPy\\Output_PPP\\"
path_Py_short = "C:\\MyC\\Data\\PolSARProPy\\Output_Py\\artifacts\\"

path_save_short = "C:\\MyC\\Funds\\ESA\\PolSARpy\\Reports\\Code test\\"

# automatically finding out the algorithms used
# dir_list = os.listdir(path_PPP_short)

### manually imputting the algorithms used
# dir_list = [
#             'arii_nned_3components_decomposition',
#             'boxcar_filter',
#             'cloude_decomposition',
#             'freeman_2components_decomposition',
#             'freeman_decomposition',
#             'lee_refined_filter',
#             'vanzyl92_3components_decomposition',
#             'wishart_h_a_alpha_classifier',
#             'yamaguchi_3components_decomposition',
#             ]

# ### TEST
dir_list = [
            'wishart_h_a_alpha_classifier',
            'wishart_supervised_classifier',
            'id_class_gen'
            ]

# ############## TO DO
# you also need to do Entropy and Anisotropy for h a alpha
# wishart supervised

# ############# NO RUNNING
# opce
# wishart supervised
# id class gen
# Arii anned


#%% SELECT THE PROCEDURE TO TEST

array_pro = dir_list


num_pro = len(array_pro)

for i in range(num_pro):
    
    plt.close()
    
    path_PPP = path_PPP_short + array_pro[i] + '\\'
    path_Py  = path_Py_short  + array_pro[i] + '\\out\\'
    
    path_save = path_save_short + '\\Results_' + array_pro[i] + '\\'
    os.makedirs(path_save, exist_ok=True)
    
    # get the filenames
    array_file_ext = os.listdir(path_Py_short + array_pro[i] + '\\out\\')
    # remove the config file if present
    if 'config.txt' in array_file_ext: array_file_ext.remove(array_file_ext[0])
    # remove the extensions
    array_file = [os.path.splitext(file)[0] for file in array_file_ext]


    num_file = len(array_file)
    print('Processing ' + array_pro[i] + '...')
    
    num_dim = num_file
    
    # adding an extra dimension for RGB in case we have less than 2 images
    if num_file < 3 : 
        num_dim = 3
        array_file.append('empty')
    



    #%% procesing the single images
    imgPPP = np.zeros([col, row, num_dim])
    imgPy  = np.zeros([col, row, num_dim])
    empty  = np.zeros([col, row])
    
    for ii in range(num_file):
        
        imgPPP[:,:,ii] = Open_ENVI_Image(path_PPP + array_file[ii] + '.bin', col, row, dtypePPP) 
        imgPPP[:,:,ii]  = np.nan_to_num(imgPPP[:,:,ii])
        
        imgPy[:,:,ii]  = Open_ENVI_Image(path_Py  + array_file[ii] + '.bin', col, row, dtypePy) 
        imgPy[:,:,ii]  = np.nan_to_num(imgPy[:,:,ii])
        
        if num_file < 3 :
            imgPPP[:,:,2] = empty
            imgPy[:,:,2]  = empty

    # creating confusion matrices
        imgPPP_flat = imgPPP[:,:,ii].flatten()
        imgPy_flat  = imgPy[:,:,ii].flatten()
        
        # Compute the confusion matrix
        # classes = np.unique(np.concatenate((A_flat, B_flat)))
        # cm = confusion_matrix(imgPPP_flat, imgPy_flat, labels=classes)
        cm = confusion_matrix(imgPPP_flat, imgPy_flat)
        
        plt.figure(figsize=(6, 5))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.imshow(cm)
        plt.show()
        
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"Accuracy: {accuracy:.4f}")

        # Calculate metrics
        # accuracy = accuracy_score(imgPPP_flat, imgPy_flat)
        report = classification_report(imgPPP_flat, imgPy_flat, 
                                       output_dict=True, digits=4)
        
        # print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(report)

        
        df_cm = pd.DataFrame(cm)
        df_cm.to_excel(path_save + str(array_file[ii]) + '_stats.xlsx', 
                     sheet_name = 'Confusion Matrix') 

        df_rep = pd.DataFrame(report).transpose()
        with pd.ExcelWriter(path_save + str(array_file[ii]) + '_stats.xlsx',
                    mode='a') as writer:
            df_rep.to_excel(writer, sheet_name='Report')


  