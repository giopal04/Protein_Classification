#!/usr/bin/env python3

import pyvista as pv

import subprocess

import argparse

import time

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

def screenshot(p, out_fn):
	# Take a screenshot
	p.screenshot(f'{out_fn}.png', transparent_background=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'filename', type=str, help='filename to plot',
	)

	# Add debug_show optional and non-positional parameter
	parser.add_argument(
		'--debug_show', action='store_true', help='show plot',
	)

	args = parser.parse_args()

	filename = args.filename

	for elevation in range(0, 360, 45):
		for azimuth in range(0, 360, 45):
			p = start_plotting(filename, args.debug_show)
			print(f'{args.debug_show = }, {filename = }, {elevation = }, {azimuth = }')
			rotate(p, elevation, azimuth)
			screenshot(p, f'{filename}-{elevation}-{azimuth}')
