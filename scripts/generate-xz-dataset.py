#!/usr/bin/env python3

import os, sys
sys.path.append("..")

import csv
import pyvista as pv

from pathlib import Path

from functools import partial
from multiprocessing import Pool

from symmetria.shapes import BenchmarkShape

'''
class BenchmarkShape:
    def __init__(self, points, symmetry_list=None):
        self.points = points  # Geometry (point cloud)
        self.symmetry_list = symmetry_list  # List of SymmetryPlane

    def save_benchmark_shape(self, output_path, prefix, file_fmt='npz', number_fmt='%.18e'):
        pc_path = os.path.join(output_path, prefix[:-1] + '.' + file_fmt)
        if file_fmt == 'npz':
            np.savez_compressed(pc_path, points=self.points, fmt=number_fmt)
        elif file_fmt == 'txt':
            np.savetxt(pc_path, self.points, fmt=number_fmt)
        elif file_fmt == 'gz':
            import gzip
            with gzip.GzipFile(pc_path, "w") as f:
                np.savetxt(f, self.points, fmt=number_fmt)
        elif file_fmt == 'xz':
            import lzma
            with lzma.open(pc_path, 'wt', encoding='ascii') as f:
                np.savetxt(f, self.points, fmt=number_fmt)
'''

def process_file(filename, protein_class, vtk_dir, output_dir):
	if not filename.endswith('.vtk'):
		return
	protein_id = os.path.splitext(filename)[0]
	print(f'Processing: {filename} ({protein_id})')
	if protein_id not in protein_class:
		print(f"Skipping {filename}: protein_id not found in CSV")
		return
	class_id = protein_class[protein_id]
	vtk_path = os.path.join(str(vtk_dir), filename)
	print(f"Reading: {vtk_path} - class_id: {class_id}")
	try:
		mesh = pv.read(vtk_path)
	except Exception as e:
		print(f"Error reading {vtk_path}: {e}")
		return
	points			= mesh.points
	num_points		= points.shape[0]
	# Generate the desired output filename components
	class_id_str		= f"{class_id:02d}"
	num_points_str		= f"{num_points:06d}"
	desired_output_filename	= f"{class_id_str}-{num_points_str}-{protein_id}.xz"
	desired_output_filename	= desired_output_filename.replace(':', '+')
	# Create the prefix to compensate for the save_benchmark_shape's prefix[:-1]
	basename_part		= desired_output_filename[:-3]  # Remove '.xz' part
	prefix_for_save		= basename_part + 'x'  # Add a character to be sliced off
	# Save using the BenchmarkShape's method
	shape			= BenchmarkShape(points)
	shape.save_benchmark_shape(output_dir, prefix_for_save, file_fmt='xz', number_fmt='%.3f')
	print(f"Saved: {desired_output_filename}")

def main():
	base_path  = Path('/mnt/raid1/dataset/shrec-2025-protein-classification/v2-20250331')
	csv_path   = base_path / 'train_set.csv'
	vtk_dir    = base_path / 'train'
	output_dir = base_path / 'train-xz'

	print(f'Reading CSV: {csv_path} and VTK files in: {vtk_dir}')
	print(f'Output directory: {output_dir}')

	# Read CSV into a dictionary mapping protein_id to class_id
	protein_class = {}
	with open(str(csv_path), 'r') as f:
		reader = csv.reader(f)
		next(reader)  # Skip header
		for row in reader:
			protein_id, class_id = row
			protein_class[protein_id] = int(class_id)
			print(f"protein_id: {protein_id} - class_id: {class_id} - {protein_class[protein_id]}")

	# Create output directory if it doesn't exist
	os.makedirs(str(output_dir), exist_ok=True)

	file_list = [filename for filename in os.listdir(str(vtk_dir)) if filename.endswith('.vtk')]
	print(f"Found {len(file_list)} VTK files - {file_list[0]} - {file_list[-1]}")
	# Process each VTK file in the directory
	#for filename in os.listdir(str(vtk_dir)):
	process_pool = Pool(48)
	process_pool.map(partial(process_file, protein_class=protein_class, vtk_dir=vtk_dir, output_dir=output_dir), file_list)

if __name__ == '__main__':
	main()
