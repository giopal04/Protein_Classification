#!/usr/bin/env python3

import pyvista as pv

import subprocess

import argparse

import time

def start_plotting(filename):
	p = pv.Plotter(off_screen=True)

	# Read filename
	p.add_mesh(pv.read(filename))

	# Set camera
	p.camera_position = 'xz'

	# Set background
	p.background_color = 'w'

	# Set window size
	p.window_size = (1000, 1000)

	# Set window title
	p.title = f'{filename}'

	# Show
	#p.show()

	return p

def rotate(p, elevation, azimuth):
	p.camera.elevation = elevation
	p.camera.azimuth = azimuth

def screenshot(p, out_fn):
	# Take a screenshot
	p.screenshot(f'{out_fn}.png', transparent_background=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'filename', type=str, help='filename to plot',
	)

	args = parser.parse_args()

	filename = args.filename

	for elevation in range(0, 360, 45):
		for azimuth in range(0, 360, 45):
			p = start_plotting(filename)
			rotate(p, elevation, azimuth)
			screenshot(p, f'{filename}-{elevation}-{azimuth}')
