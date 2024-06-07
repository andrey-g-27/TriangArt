# TriangArt
Program for creating trianguated images from source ones.
Controllable parameters: average triangle size & size variability.

Basic idea is to drop a set of points onto the image, then make them move following image gradient, but with repelling force if they get too close.

## Project files
* `triang_art.py` - main program with GUI.
* `triang_img.py` - module with image processing code.

## Required libraries
PyQt6, NumPy, SciPy, Matplotlib.

## Example
Original image used is by Reurinkan under Creative Commons Attribution 2.0 Generic license.

Original image: 

![Original image.](/ImgTest/Baxoi_County.jpg)

Processed image: 

![Processed image.](/ImgTest/Baxoi_County_triang.png)
