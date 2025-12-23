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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score

import tkinter as tk

import matplotlib
# matplotlib.use('Agg')   # non-GUI backend (no windows open)

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
dir_list = [ 'Cameron'
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
    
       
    imgPy  = np.zeros([col, row, num_dim])

       
#################### Cameron ###########################
    if name_routine == "Cameron":
        imgPy[:, :, 0] = ds['cameron'].values
        array_file_Py = ['cameron']


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
            


    #%% procesing the single images
    imgPSP = np.zeros((col, row, num_dim))
    
    

################### Cameron #################
    if name_routine == "Cameron":
        
        img_temp = Open_PSP_Image(path_PSP / ('Cameron.bin'), col, row, dtypePSP)
        imgPSP[:,:,0] = np.transpose(np.nan_to_num(img_temp))
        array_file_PSP = ['Cameron']        

        
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


    max_class = np.max(imgPSP).real

    fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
    plt.title('Python ' + str(array_pro[i]) + ': ' + array_file_PSP[0]) 
    im = plt.imshow(np.abs(imgPy), cmap = 'jet', vmin=0, vmax=max_class)
    plt.savefig(path_save / ('Py_' + array_file_Py[0]), bbox_inches='tight', pad_inches=0)

    fig = plt.figure(figsize=(screen_width / dpi, screen_height / dpi), dpi=dpi)    
    plt.title('PolSARPro ' + str(array_pro[i]) + ': ' + array_file_PSP[0]) 
    im = plt.imshow(np.abs(imgPSP), cmap = 'jet', vmin=0, vmax=max_class)
    plt.savefig(path_save / ('PSP_' + array_file_Py[0]), bbox_inches='tight', pad_inches=0)


#%% #######################################
# procesing the single images

    for jj in range(num_file):
        # creating confusion matrices
        imgPPP_flat = imgPSP[:,:,jj].flatten()
        imgPy_flat  = imgPy[:,:,jj].flatten()
        
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
    
        
        file_path = path_save / f"{array_pro[jj]}_stats.xlsx"
        
        df_cm = pd.DataFrame(cm)
        df_rep = pd.DataFrame(report).transpose()
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_cm.to_excel(writer, sheet_name='Confusion Matrix', index=True)
            df_rep.to_excel(writer, sheet_name='Report', index=True)

    
        # saving as csv now
        df_cm.to_csv(
            path_save / f"{array_pro[jj]}_confusion_matrix.csv",
            index=True
        )
        
        df_rep.to_csv(
            path_save / f"{array_pro[jj]}_report.csv",
            index=True
        )
    

