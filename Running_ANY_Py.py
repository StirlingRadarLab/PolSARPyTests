#!/usr/bin/env python
# coding: utf-8

# # Yamaguchi 4-component decomposition

# This notebook shows how to apply the Yamaguchi 4-component decomposition to a PolSARpro NetCDF dataset.

# ## Import packages and set directories

# In[1]:

import sys

flag_routine = "yamaguchi"
# flag_routine = "lee"
# flag_routine = "PWF"
# flag_routine = "Freeman"


if flag_routine == "yamaguchi":
    sys.path.insert(0, '/home/am221/C/Programs/PolSARproPy/Phase_2/Branches/Yamaguchi')
    from polsarpro.decompositions import yamaguchi4

elif flag_routine == "lee":
    sys.path.insert(0, '/home/am221/C/Programs/PolSARproPy/Phase_2/Branches/LeeRefined')
    from polsarpro.speckle_filters import refined_lee

elif flag_routine == "PWF":
    sys.path.insert(0, '/home/am221/C/Programs/PolSARproPy/Phase_2/Branches/PWF')
    from polsarpro.speckle_filters import PWF

elif flag_routine == "Freeman":
    sys.path.insert(0, '/home/am221/C/Programs/PolSARproPy/Phase_2/Branches/Freeman')
    from polsarpro.decompositions import freeman
    

import os 
from pathlib import Path
import xarray as xr
from polsarpro.io import open_netcdf_beam
from polsarpro.util import S_to_T3
from polsarpro.util import S_to_C3

# optional import for progress bar
from dask.diagnostics import ProgressBar

# change to your data paths
# original dataset
input_alos_data = Path("/home/am221/C/Data/PolSARPy/NetCDF/SAN_FRANCISCO_ALOS1_slc.nc")
output_dir = Path("/home/am221/C/Data/PolSARPy/Output_Py")


#%% Load data
# 
# We load the SNAP NetCDF-BEAM dataset using the `open_netcdf_beam` function. 
# To obtain such a dataset, please refer to the "Getting Started" tutorial or the `quickstart-tutorial.ipynb` notebook.


# uncomment to test on S matrix made with SNAP
S = open_netcdf_beam(input_alos_data)


# ## Apply the decomposition
# 
# Let's apply the decomposition and write the result to a NetCDF file.
# Optionally we can use a progress bar to monitor the progress of the computation.

#%% Decide what process to run


# # # ### TEST
if flag_routine == "yamaguchi":
    dir_list = [ 
                'yamaguchi_4components_decomposition'
                ]
elif flag_routine == "lee":
    dir_list = [ 
                'LeeRefined'
                ]    

elif flag_routine == "PWF":
    dir_list = [ 
                'PWF'
                ]    
    
elif flag_routine == "Freeman":
    dir_list = [ 
                'Freeman'
                ]    

array_pro = dir_list

# change to the name of your liking
file_out = output_dir / array_pro[0]

# netcdf writer cannot overwrite
if os.path.isfile(file_out):
    os.remove(file_out)


if flag_routine == "yamaguchi":

    mode = "s4r" # choose mode among "y4o", "y4r", "s4r"
    
    with ProgressBar():
        yamaguchi4(S, boxcar_size=[7, 7], mode=mode).to_zarr(file_out, mode="w")

elif flag_routine == "lee":

    with ProgressBar():
        # convert S to T3
        T3 = S_to_T3(S)
        C3 = S_to_C3(S)
        # apply the filter
        T3_flt = refined_lee(T3, window_size=7, num_looks=1).to_zarr(file_out, mode="w")

        # use custom function that writes complex matrices and preserves chunks
        # polmat_to_netcdf(T3_flt, output_test_dir / f"T3_refined_lee.nc")    


elif flag_routine == "PWF":

    with ProgressBar():
        PWF(S, train_window_size=[9, 9], test_window_size=[1,1]).to_zarr(file_out, mode="w")


elif flag_routine == "Freeman":

    with ProgressBar():
        freeman(S, boxcar_size=[7, 7]).to_zarr(file_out, mode="w")
        
        
# #%% CHECKS IF NEEDED
# # ## Display outputs

# # We open the previously saved dataset:

# res = xr.open_dataset(file_out)

# # odd bounce / surface component
# res.odd.plot.imshow(vmin=0, vmax=1)
# # double bounce
# res.double.plot.imshow(vmin=0, vmax=1)
# # volume component
# res.volume.plot.imshow(vmin=0, vmax=1)
# # helix component
# res.helix.plot.imshow(vmin=0, vmax=1)

# # make an rgb image from diagonal elements
# rgb = xr.concat([T3_flt.m22, T3_flt.m33, T3_flt.m11], dim="band")

# # compute clipping values to handle the high dynamic range
# clip_val = 3 * rgb.mean(dim=("x", "y"))

# # clip, normalize and display the image
# (rgb.clip(0, clip_val) / clip_val).plot.imshow(figsize=(5,8))


