# smooth-segmentation-output
A module to improve the performance of segmentation outputs of remote sensing data


Many segmentation models based on neural networks (i.e. U-Net)  utilizes image patches to make predictions on localized windows, neglecting data near the border of the patches. This can result in high prediction errors near the outside of the window, compounded by the fact that predictions may be concatenated, producing an even more jagged output.

A simple solution is using 2D interpolation between overlapping patches when producing final predictions. This is the approach taken by the current algorithm, which accepts as inputs the image and its size, the size of the windows, the number of times we want to overlap, and the function that performs local predictions.

The implementation of this algorithm splits the image into patches using a 5D NumPy array. Image patches are already a 3D array, but spatial ordering may require two additional dimensions. These patches are then reshaped to 4D, along a single batch_size dimension, to be passed into the neural network's prediction function. Batch predictions can be performed since patches are loaded in memory, assuming enough memory is available. The predictions are then reassembled into a 5D array, which is merged with a spline interpolation to produce a regular 3D image array.

This approach may not be suitable for regular computers if the image is too large, since the 5D array can be too large to fit into CPU RAM (GPU RAM usage is balanced through the use of a batch_size variable within the prediction function).