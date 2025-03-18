"""
@author: wassimchakroun, GET

Target: Apply Grad-Cam on the patches of an image and display the heatmaps that highlight the U-Net decision
"""

import random
import asyncio
import resource
import threading
import gc
import pickle

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import earthpy.plot as ep # Simple plot with spatial raster
from rasterio.plot import show, plotting_extent

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU

from core.UNet import UNet
from core.frame_info import adjust_contrast
from prediction import * # Load all functions so far


# Find the convolutional layer that captures high-level features (before the last up-sampling layer)
def unet_conv_layer(model, layer):
    """ Get the values for unet_conv_layer: 
    -using model.summary() to see the names of all layers in the model - set the argument 'summary' of U-Net to True and find manually the layer)
    -searching directly for the conv layer
    Desired layer: Conv layer = conv2d_27 and Shape = (None, 256, 256, 1) """
    # Return the layer given its name
    try:
        return model.get_layer(layer)
    
    except:
        # Raise an explanatory error if the layer isn't found
        raise ValueError("Unable to locate the convolutional layer")


# Process an input image by creating patches
def process_image(config):
    """ Convert an input image, dedicated to be predicted, to patches """
    # Get the file names in the directory tree of prediction
    img_path = ""
    for root, dirs, files in os.walk(config.to_predict_dir):
        
        # Select randomly a single file 
        fl = random.choice(files)

        # Check that the file is an image with extension 'tif'
        if fl.endswith(config.predict_images_file_type) and fl.startswith(config.predict_images_prefix):

            # Create the path to the file
            img_path = os.path.join(root, fl)
        
        # Verify that the image path is not empty and break out of the loop
        if img_path != "":
            print("An image has been randomly selected through this path:", img_path)
            break
                
    # Raise an explanatory error if there are no images in the directory
    if len(files) == 0:
        raise Exception("No images to predict")

    # Open the selected image (Tiled mode allows for reading the image in smaller chunks useful for processing large raster images efficiently : default tiled True / The size of the blocks to be used when reading the raster image: default blockxsize and blockysize 256)
    rast = rasterio.open(img_path) # Corresponding numpy array of shape (c=12, h=10980, w=10980)
    
    # Get the affine transformation matrix used to transform pixel coordinates to map coordinates (and vice versa) 
    transform = rast.transform

    # Get the list of patch offsets for the image
    patch_width, patch_height = config.patch_size
    stride = config.prediction_stride
    offsets = get_patch_offsets(rast, patch_width, patch_height, stride)
    
    # Generate patches from the selected image
    batch, batch_norm, batch_pos, coordinates = [], [], [], []
    big_window = Window(0, 0, rast.width, rast.height)

    # Yield the fixed img_path value
    #yield None, None, None, None, img_path
    
    for col_off, row_off in tqdm(offsets, desc=f"Creating batch of {(patch_width, patch_height, len(config.channel_list))} patches from {len(offsets)} offsets"):

        # Initialise patch with zero padding in case of corner images. The size is based on number of channels
        patch = np.zeros((patch_height, patch_width, np.sum(config.channels_used)))
        patch_norm = np.zeros((patch_height, patch_width, np.sum(config.channels_used)))

        # Load patch window from image
        patch_window = Window(col_off=col_off, row_off=row_off, width=patch_width, height=patch_height).intersection(big_window)
        patch_pos = (patch_window.col_off, patch_window.row_off, patch_window.width, patch_window.height)
        
        # Select only the necessary channels
        temp_im = rast.read(list(np.where(config.channels_used)[0] + 1), window=patch_window)
        
        # Transpose the image to match the required shape order (h, w, c)
        temp_im = temp_im.transpose((1, 2, 0))

        # Normalize the image along the width and height i.e. independently per channel. Ignore nodata for normalization
        temp_im_norm = image_normalize(temp_im, axis=(0, 1), nodata_val=rast.nodatavals[0])

        # Apply contrast adjustment with gamma correction
        #temp_im_norm = adjust_contrast(temp_im_norm, gamma=.001)

        # Add to batch list
        patch[:patch_window.height, :patch_window.width] = temp_im
        patch_norm[:patch_window.height, :patch_window.width] = temp_im_norm
        batch.append(patch)
        batch_norm.append(patch_norm)
        batch_pos.append(patch_pos)

        # Extract the coordinates of the patch (lon_left, lon_right, lat_bottom, lat_top) according to 'extent' argument
        patch_coords = bounds(patch_window, transform=transform)
        patch_coords = (patch_coords[0], patch_coords[-2], patch_coords[1], patch_coords[-1])
        coordinates.append(patch_coords)
        
        # Yield the results one by one
        #yield patch, patch_norm, patch_pos, patch_coords, None

    return batch, batch_norm, batch_pos, coordinates, img_path


# Generate Grad-CAM heatmaps for the U-Net model
def grad_cam(img_patches, model, layer, img_name):
    """ Compute the activation maps through Grad-Cam """
    # Load the conv layer of U-Net
    conv_gcam = unet_conv_layer(model, layer)

    # Create a model with the original inputs as inputs, the conv layer and the original output as the output
    grad_model = keras.Model([model.inputs], [conv_gcam.output, model.output])

    # Create a list to gather the heatmaps of all patches
    heatmaps = []

    print(f"Starting computing Grad-Cam of the image: {img_name}.")
    start = time.time()

    # Iterate over the patches of the image
    for patch in tqdm(img_patches, unit="patches"):

        # Rescale to a range 0-1: scaled patch (WHAT ABOUT - VALUES)
        #patch = patch / patch.max()

        # Expand the dimensions of the patch to match the input shape of the model
        patch = np.expand_dims(patch, axis=0)
        
	# Use gradient tape to monitor the conv layer output and retrive the gradients corresponding to the prediction
        with tf.GradientTape() as tape:

	    # Pass the patch through the base model and get the feature map and the predicted class probabilities
            final_conv_output, predicted_probs = grad_model(patch) # <class 'tensorflow.python.framework.ops.EagerTensor'>, shape=(1, 256, 256, 1), dtype=float32

            # While final_conv_output is not a tf.Variable => we need to set the tape watch it, or else the derivative will be 0
            tape.watch(final_conv_output)
	    
	# Compute the gradient with respect to the feature map of the conv layer
        gradient = tape.gradient(predicted_probs, final_conv_output) # <class 'tensorflow.python.framework.ops.EagerTensor'>, shape=(1, 256, 256, 1), dtype=float32

	# Generate a vector where each entry is the mean intensity of the gradient across the spatial dimensions, over a specific feature map channel (1 channel in our case)
        pooled_grads = tf.reduce_mean(gradient, axis=(1, 2, 3))

        # Multiply the output of the conv layer with the pooled gradients (in other words "how important the channel is")
        heatmap = final_conv_output * pooled_grads
        
        # Remove dimensions of size 1 from the shape
        heatmap = tf.squeeze(heatmap)

	# Normalize the heatmap between 0 & 1 (for visualization purpose) and avoid zero-division
        if tf.reduce_max(heatmap) == 0.:
             heatmap = tf.maximum(heatmap, 0.)
             heatmap = tf.where(tf.equal(heatmap, 0.), tf.fill(heatmap.shape, 1e-8), heatmap)
        else:
             heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
             
	# Add the current heatmap to the list
        heatmaps.append(heatmap.numpy()) # Numpy array of shape (h=256, w=256)

        # Produce the current heatmap of the counter (generators don't store the whole sequence in memory at once)
        #yield heatmap.numpy()

    print(f"Computing completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")

    return heatmaps


# Save Grad-Cam results over the corresponding patches
def save_over_heatmaps(img_patches, img_patches_norm, heatmaps, coords, img_name, path_gcam, path_hmap , alpha=.3):
    """ Create plots of heatmaps over their corresponding patches """
    # Create a collection of patches, heatmaps and coordinates
    collection = zip(img_patches, img_patches_norm, heatmaps, coords)

    # Create directories to store the results of gradcam and the heatmaps respectively
    if not os.path.exists(path_gcam):
        os.makedirs(path_gcam)
        print("Directory '%s' created" % img_name)

    if not os.path.exists(path_hmap):
        os.makedirs(path_hmap)
        print("Directory 'Heatmaps' created")
    
    print(f"Starting saving heatmaps on patches of the image: {img_name}.")
    start = time.time()

    # Iterate over the collection of patches, heatmaps and coordinates
    for indx, (patch, patch_norm, hmap, crd) in tqdm(enumerate(collection)):
        
        # Select RGB bands (for image displaying purpose) and rescale the patch to a range 0-1
        patch_norm_scaled = patch_norm[:, :, :3] / patch_norm[:, :, :3].max() # Already normalized !?
        patch_norm_scaled = patch_norm_scaled[:, :, ::-1] # BGR -> RGB (it was in the reverse order: we tend to see the blue color in images)
        patch_scaled = patch[:, :, :3] / patch[:, :, :3].max()
        patch_scaled = patch_scaled[:, :, ::-1]

        # Select false color infrared bands (B08, B04, B03) and rescale the patch to a range 0-1
        #false_patch_norm_scaled = patch_norm[:, :, 1:4] / patch_norm[:, :, 1:4].max()
        #false_patch_norm_scaled = false_patch_norm_scaled[:, :, ::-1] # GR-NIR -> NIR-RG
        false_patch_scaled = patch[:, :, 1:4] / patch[:, :, 1:4].max()
        false_patch_scaled = false_patch_scaled[:, :, ::-1]

        # Save the heatmaps (for comparison with grad-cam purpose)
        plt.figure() #figsize=[6.4, 4.8]
        #plt.rc_context({"xtick.major.pad": 10}) # Spacing between xticks
        plt.imshow(hmap, extent=crd, cmap='jet')
        plt.title(img_name, fontsize=6) #title() for the subtitle
        plt.suptitle(f"Heatmap_{indx}", fontsize=12, fontweight='bold') #suptitle() for the actual title
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar()
        plt.savefig(f"{path_hmap}/heatmap_{indx}.jpg")

        # Prepare the figure and axes of the plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

        # Set the title and patch data of the first axis
        ax1.set_title('Patch')
        ax1.imshow(patch_scaled, cmap='gray')
        ax1.axis('off')

        # Set the title and normalized patch data of the first axis
        ax2.set_title('Normalized patch')
        ax2.imshow(patch_norm_scaled, cmap='gray')
        ax2.axis('off')

        # Set the title and false color patch data of the second axis
        ax3.set_title('False color patch')
        ax3.imshow(false_patch_scaled, cmap='gray')
        ax3.axis('off')

        # Set the title and patch-heatmap data of the third axis
        ax4.set_title('Grad-Cam')
        ax4.imshow(patch_scaled, cmap='gray') # Normalized patches are the inputs of the Grad-Cam model #patch_norm_scaled
        ax4.imshow(hmap, cmap='jet', alpha=alpha)
        ax4.axis('off')

        # Save the resulting figure
        plt.tight_layout()
        fig.suptitle(img_name, fontweight='bold', y=0.7) # Alternative to coordinates
        fig.set_size_inches([10, 10])
        fig.savefig(f"{path_gcam}/gradcam_{indx}.jpg")

    print(f"Saving Gradcam results completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")


# Merge Grad-Cam results
def merge_heatmaps(patches_pos, heatmaps, img_path, img_name, path_hmap, operator="REPLACE"):
    """ Merge all the heatmaps to compare the result with the prediction of the U-Net """
    # Open the selected image
    rast = rasterio.open(img_path)

    # Get the profile of the raster
    profile = rast.profile.copy() 
    # {'driver': 'GTiff', 'dtype': 'int16', 'nodata': None, 'width': 10980, 'height': 10980, 'count': 12, 'crs': CRS.from_epsg(32630), 'transform': Affine(10.0, 0.0, 499980.0, 0.0, -10.0, 1500000.0), 'tiled': False, 'interleave': 'pixel'}

    # Extract the coordinates of the four corners of the bounding box
    rast_bounds = rast.bounds
    rast_bounds = (rast_bounds[0], rast_bounds[-2], rast_bounds[1], rast_bounds[-1]) # (left, right, bottom, top)

    # Initialize the merged heatmap to zeros, or -1 for MIN operator
    merged_hmap = np.zeros((rast.height, rast.width), dtype=np.float32) # Heatmap values are float
    if operator == "MIN":
        merged_hmap = merged_hmap - 1

    print(f"Starting merging heatmaps of the image: {img_name}.")
    start = time.time()

    # Iterate over the positions of each patch (lon, lat, width, height)
    for i in tqdm(range(len(patches_pos))):

        # Load the patch position and the corresponding heatmap
        (col, row, wi, he) = patches_pos[i]
        hmap = heatmaps[i]

        # Instead of replacing the current values with new values, use the user specified operator (MIN, MAX, REPLACE)
        merged_hmap = add_to_result(merged_hmap, hmap, row, col, he, wi, operator)
    
    # Run once more to process the last partial batch (when image not exactly divisible by N batches)
    #if patches_pos:
        #merged_hmap = add_to_result(merged_hmap, hmap, row, col, he, wi, operator)

    # Rescale the merged heatmap to a range 0-255
    #merged_hmap = np.uint16(merged_hmap * 255)

    # Update the profile of the raster
    profile.update(dtype=merged_hmap.dtype) 

    # Save the merged heatmap (for comparison with the model's predictions)
    plt.figure(figsize=[20, 20])
    plt.imshow(merged_hmap, extent=rast_bounds, cmap='jet')
    plt.tick_params(axis='both', which='major', labelsize=15) # Tick label font size
    plt.title(img_name, fontweight='bold', fontsize=12)
    plt.savefig(f"{path_hmap}/{img_name}(merged_heatmap).jpg")

    # Convert the merged heatmap to a raster
    with rasterio.open(f"{path_hmap}/{img_name}(merged_heatmap).tif", 'w', **profile) as dst:
        
        # Write the merged heatmap to the raster bands
        dst.write(merged_hmap, 1)

    print(f"Merging heatmaps completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")


# Display patches per band
def patches_per_band(img_patches, img_name, path_gcam):
    """ Compare the bands of patches to Grad-Cam results """

    # Create a list of used Sentinel-2 bands
    bands = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

    # Create directory to store the bands of each patch
    path_ppband = os.path.join(path_gcam + '/', "Patches_bands")
    if not os.path.exists(path_ppband):
        os.makedirs(path_ppband)
        print("Directory 'Patches_bands' created")

    print(f"Starting saving bands of the image: {img_name}.")
    start = time.time()

    # Iterate over the patches of the image
    for indx, patch in tqdm(enumerate(img_patches), unit="patches"):

        # Transpose the patch to match the required shape order (c, h, w)
        patch = patch.transpose((2, 0, 1))

        # Check whether the correct number of bands was selected
        if len(bands) != patch.shape[0]:
            raise ValueError("Expects the number of plot titles to equal the number of array raster layers.")

        # Define the number of columns and rows for plot grid
        cols = 3
        rows = 2 #int(np.ceil(patch.shape[0] / cols)) # The smallest integer i, such that i >= x (used if we don't know the number of bands)

        # Get the number of channels
        total_layers = patch.shape[0]

        # Plot all bands
        fig, axs = plt.subplots(rows, cols, figsize=(12, 12))
   
        # Flatten the axis into 1D array and iterate
        axs_ravel = axs.ravel()
        for ax, i in zip(axs_ravel, range(total_layers)):

            # Create a matplotlib figure with an image axis
            ep._plot_image(patch[i], title=bands[i], ax=ax) # cmap (default = "Greys_r")

        # Adjust the padding between and around subplots and save figure
        plt.tight_layout()
        fig.suptitle(img_name, fontweight='bold', y=0.95)
        plt.savefig(f"{path_ppband}/patch_bands_{indx}.jpg")

        # Perform a sanity check
        #print(axs_ravel, axs_ravel.shape)
        #if indx == 3:
            #break

    print(f"Saving bands completed in {str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n")


# Set all functions together
def set_gcam(config):
    """ Execute the whole process: process the image, load last conv layer, compute activation maps, visualize the result and merge heatmaps """
    # Define the start point of memory usage
    start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Current memory usage: {(start_mem / 1024.0 ** 2):.2f} MB") # Convert to megabytes

    # Load and process an original image (selected randomly)
    batch, batch_norm, batch_pos, coords, img_path = process_image(config)

    # Define a new model
    #generic_unet = UNet([config.train_batch_size, *config.patch_size, len(config.channel_list)], [len(config.channel_list)])

    # Load a pretrained model (just perform inference and no further optimization or training)
    pretrained_unet = tf.keras.models.load_model(config.trained_model_path, compile=False)
    #pretrained_unet = tf.keras.models.load_model("/home/wassim/Documents/saved_models/20230420-1330_adam_leakyrelu.h5", compile=False, custom_objects={'LeakyReLU': LeakyReLU})

    # Extract the image name (without extension)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    # Define the paths of Grad-Cam and heatmaps results
    path_gcam = os.path.join("../superimposed_heatmaps/", img_name)
    path_hmap = os.path.join(path_gcam + '/', "Heatmaps")

    # Generate class activation heatmaps
    heatmaps = grad_cam(batch_norm, pretrained_unet, "conv2d_27", img_name)

    # Save the results
    save_over_heatmaps(batch, batch_norm, heatmaps, coords, img_name, path_gcam, path_hmap, alpha=.4)

    # Merge heatmaps of all patches
    merge_heatmaps(batch_pos, heatmaps, img_path, img_name, path_hmap)

    # Delete variables that are no longer needed
    #for var in locals().copy():
        #print(locals()) # del locals()[var]

    # Serialize batch arrays and save them to a file
    """path_batch = path_gcam + '/batch.pkl'
    with open(path_batch, 'wb') as batch_file:
        pickle.dump(batch, batch_file)

    # Load the batch from the saved file
    with open(path_batch, 'rb') as f:
        batch = pickle.load(f)"""

    # Save patches per band
    #patches_per_band(batch, img_name, path_gcam)

    # Compute the total memory usage
    end_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_usage = (end_mem - start_mem) / 1024.0 ** 2
    print(f"Memory usage: {mem_usage:.2f} MB") # The memory usage was about 43.3 MB without the last func --> set the limit usage to 50 MB

