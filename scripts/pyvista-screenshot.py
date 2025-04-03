#!/usr/bin/env python3

import pyvista as pv

import subprocess

import argparse

import time

def start_plotting(filename):
	#p = pv.Plotter()
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

	#p.camera.Dolly(1.2)
	'''
	p.camera.Yaw(0)
	p.camera.Pitch(45)
	p.camera.Roll(0)
	'''
	#p.reset_camera()
	#p.camera.elevation = 90
	# Rotate the camera
	#p.camera.roll = 135
	#p.camera.azimuth = 135

	# Show
	#p.show()
	return p

def rotate(p, elevation, azimuth):
	p.camera.elevation = elevation
	p.camera.azimuth = azimuth

def screenshot(p, out_fn):
	p.screenshot(f'{out_fn}.png', transparent_background=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'filename', type=str, help='filename to plot',
	)

	args = parser.parse_args()

	filename = args.filename

	# Take a screenshot
	#p.screenshot(f'{filename}.png', transparent_background=True)

	for elevation in range(0, 360, 45):
		for azimuth in range(0, 360, 45):
			p = start_plotting(filename)
			rotate(p, elevation, azimuth)
			screenshot(p, f'{filename}-{elevation}-{azimuth}')
			#p.camera.elevation = elevation
			#p.camera.azimuth = azimuth
			#time.sleep(0.5)
			#p.export_html('/tmp/render.html')
			#p.screenshot(f'{filename}-{elevation}-{azimuth}.png', transparent_background=True)

			'''
			# Export to HTML
			p.export_html('/tmp/render.html')
			
			# Use shot-scraper to render and screenshot this via a subprocess
			# You can use the --wait flag to deal with screenshotting renders that take a while to render
			subprocess.Popen(f"shot-scraper /tmp/render.html -o /tmp/prot/screenshot-{filename}-{elevation}-{azimuth}.png --wait 5000".split(" "))
			'''
