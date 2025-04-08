#!/usr/bin/env python3

import pyvista as pv

import os
#os.environ['PYVISTA_OFF_SCREEN'] = 'true'
#os.environ['VTK_USE_X'] = 'off'
os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkOSOpenGLRenderWindow'		# because of this: https://github.com/pyvista/pyvista/issues/1180
#os.environ['VTK_DEFAULT_OPENGL_WINDOW'] = 'vtkEGLRenderWindow'

import vtk
ren_win = vtk.vtkRenderWindow()
print(ren_win.GetClassName())  # Should output vtkOSOpenGLRenderWindow

'''
# Get system info
import pyvista as pv
print(pv.Report())
'''

'''
import sys
sys.exit(0)
'''

import subprocess

import argparse

import time

import cv2

from pathlib import Path


import cv2
import numpy as np

def bounding_box(img):
	# Apply threshold to create binary mask
	threshold_value = 10
	_, thresh = cv2.threshold(img.astype(np.uint8), threshold_value, 255, cv2.THRESH_BINARY)
	
	# Find contours in the binary image
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	if contours:
		# Find largest contour and its bounding box
		largest_contour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largest_contour)
	else:
		# Fallback to full image if no contours found
		h, w = img.shape[:2]
		x, y = 0, 0
	return x, y, w, h

def crop_and_resize(img, bbox):
	x, y, w, h = bbox

	# Crop and resize
	cropped = img[y:y+h, x:x+w]
	resized = cv2.resize(cropped, (128, 128))

	# Convert grayscale to BGR if necessary
	if len(resized.shape) == 2:
		resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
	if resized.shape[2] == 4:
		resized = resized[:,:,:3]

	return resized

def create_grid_8x8(image_list):
	processed = []
	for img in image_list:
		# Convert to max channel if the image is color
		if len(img.shape) == 3:
			max_channels = np.max(img, axis=2)
		else:
			max_channels = img.copy()

		# Find bounding box
		bbox = bounding_box(max_channels)
		x, y, w, h = bbox
	
		# Crop and resize
		resized = crop_and_resize(img, bbox)
		
		processed.append(resized)
	
	# Create 8x8 grid
	grid = np.zeros((1024, 1024, 3), dtype=np.uint8)
	
	for i, img in enumerate(processed):
		row = i // 8
		col = i % 8
		grid[row*128:(row+1)*128, col*128:(col+1)*128] = img
	
	return grid

def start_plotting(filename, debug_show=False):
	p = pv.Plotter(off_screen=(not debug_show))

	# Read filename
	p.add_mesh(pv.read(filename))

	# Set camera
	p.camera_position = 'xz'

	# Set background
	p.background_color = 'k'

	# Set window size
	p.window_size = (1000, 1000)

	# Set window title
	p.title = f'{filename}'

	# Remove legend
	p.remove_legend()
	# Remove scalar bar
	p.remove_scalar_bar()

	# Show
	if debug_show:
		p.show()

	return p

def rotate(p, elevation, azimuth):
	p.camera.elevation = elevation
	p.camera.azimuth = azimuth

def screenshot(p, out_fn, ext='jpg', quality=90, dontsave=False):
	# Take a screenshot
	outf = f'{out_fn}.{ext}'
	#print(f'Writing output to: {outf}')
	img = p.screenshot(filename=None, transparent_background=True)

	if dontsave:
		return img

	save_image(img, out_fn, ext, quality)

def save_image(img, out_fn, ext='jpg', quality=90):
	if ext == 'jpg':
		#img = img.convert('RGB')
		img = img[:,:,:3]
		cv2.imwrite(out_fn, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
	else:
		cv2.imwrite(out_fn, img)

def get_class_and_id_from_filename(filename, is_test_set=False):
	# 1768-60-6msg_32:TA:r_model1.vtk
	# 2043-81-6qc2_24:X:B8_model1.vtk
	# The first field is the id, the second one is the class
	# Separator is '-'
	fn = str(Path(filename).stem)
	if not is_test_set:
		cls = fn.split('-')[1]
	else:
		cls = 'unk'
	_id = fn.split('-')[0]
	return cls, _id

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'filename', type=str, help='filename to plot',
	)

	# Add debug_show optional and non-positional parameter
	parser.add_argument(
		'--debug_show', action='store_true', help='show plot',
	)

	# Add output directory --out_dir optional and non-positional parameter
	parser.add_argument(
		'--out_dir', type=str, help='output directory', default='/tmp/proteins-screenshots'
	)

	# Add is_test_set optional and non-positional parameter (true or false) to handle absence of class information
	parser.add_argument(
		'--is_test_set', action='store_true', help='is test set', default=False
	)

	args = parser.parse_args()

	filename = args.filename
	cls, _id = get_class_and_id_from_filename(filename, args.is_test_set)
	out_dir  = Path(args.out_dir) / cls
	print(f'{_id} - {cls} - Creating directory: {out_dir}')
	Path(out_dir).mkdir(parents=True, exist_ok=True)

	fn_digits = Path(filename).stem

	img_list = []
	for elevation in range(0, 360, 45):
		for azimuth in range(0, 360, 45):
			p = start_plotting(filename, args.debug_show)
			if args.debug_show or True:
				print(f'Capturing image: {filename = } - {elevation = } - {azimuth = }')
			rotate(p, elevation, azimuth)
			out_fn = Path(out_dir) / f'{Path(filename).stem}-{elevation}-{azimuth}'
			#print(f'Output filename: {out_fn}')
			img = screenshot(p, out_fn, dontsave=True)
			img_list.append(img)

	grid_8x8 = create_grid_8x8(img_list)
	out_fn = Path(out_dir) / f'{Path(filename).stem.replace(":", "+")}-grid-8x8.png'
	print(f'Writing image: {out_fn}')
	save_image(grid_8x8, out_fn, ext='png', quality=100)
