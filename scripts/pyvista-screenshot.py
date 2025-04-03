#!/usr/bin/env python3

import pyvista as pv

'''
# Get system info
import pyvista as pv
print(pv.Report())

import sys
sys.exit(0)
'''

import subprocess

import argparse

import time

import cv2

from pathlib import Path

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

def screenshot(p, out_fn, ext='jpg', quality=90):
	# Take a screenshot
	outf = f'{out_fn}.{ext}'
	#print(f'Writing output to: {outf}')
	img = p.screenshot(filename=None, transparent_background=True)
	if ext == 'jpg':
		#img = img.convert('RGB')
		img = img[:,:,:3]
		cv2.imwrite(outf, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
	else:
		cv2.imwrite(outf, img)

def get_class_and_id_from_filename(filename):
	# 1768-60-6msg_32:TA:r_model1.vtk
	# 2043-81-6qc2_24:X:B8_model1.vtk
	# The first field is the id, the second one is the class
	# Separator is '-'
	fn = str(Path(filename).stem)
	cls = fn.split('-')[1]
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

	args = parser.parse_args()

	filename = args.filename
	cls, _id = get_class_and_id_from_filename(filename)
	out_dir  = Path(args.out_dir) / cls
	print(f'{_id} - {cls} - Creating directory: {out_dir}')
	Path(out_dir).mkdir(parents=True, exist_ok=True)

	for elevation in range(0, 360, 45):
		for azimuth in range(0, 360, 45):
			p = start_plotting(filename, args.debug_show)
			#print(f'{args.debug_show = }, {filename = }, {elevation = }, {azimuth = }')
			rotate(p, elevation, azimuth)
			out_fn = Path(out_dir) / f'{Path(filename).stem}-{elevation}-{azimuth}'
			#print(f'Output filename: {out_fn}')
			screenshot(p, out_fn, ext='jpg', quality=90)
