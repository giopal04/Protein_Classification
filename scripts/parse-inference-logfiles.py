#!/usr/bin/python3

import sys
import argparse
from pathlib import Path
from collections import defaultdict

def main(debug=False):
	parser = argparse.ArgumentParser(description='Process log file and generate summary.')
	parser.add_argument('input_file',  type=Path, help='Input log file path')
	parser.add_argument('output_file', type=Path, help='Output file path')
	args = parser.parse_args()

	# Data structure to hold sum and count for each label under each object_id
	data = defaultdict(lambda: defaultdict(lambda: {'sum': 0.0, 'count': 0}))

	# Read and process each line of the input file
	with args.input_file.open('r') as f_in:
		for line in f_in:
			line = line.strip()
			if not line:
				print(f'Empty line! {line}')
				continue
			parts = line.split(' - ')
			if len(parts) != 3:
				print(f'Malformed line! {line}')
				continue  # Skip malformed lines

			obj_elev_azim, label_str, confidence_str = parts

			# Extract obj_id from the first part (obj_id-elevation-azimuth)
			obj_id_parts = obj_elev_azim.split('-')
			if len(obj_id_parts) < 3:
				print(f'Incorrect format! {line}')
				continue  # Skip if format is incorrect
			obj_id = obj_id_parts[0]
			label = label_str.strip()
			try:
				confidence = float(confidence_str.strip())
			except ValueError:
				print(f'Invalid confidence! {line}')
				continue  # Skip lines with invalid confidence

			# Update sum and count for the current obj_id and label
			data[obj_id][label]['sum'] += confidence
			data[obj_id][label]['count'] += 1

	# Sort object IDs numerically
	sorted_obj_ids = sorted(data.keys(), key=lambda x: int(x))

	# Write the output file
	with args.output_file.open('w') as f_out:
		for obj_id in sorted_obj_ids:
			labels_info = data[obj_id]
			if debug:
				print(f'labels - {obj_id}: {labels_info}')
			averages = {}
			total_count = 0
			for label, info in labels_info.items():
				avg = info['sum'] / info['count']
				#averages[label] = (avg, info['count'])
				averages[label] = (avg, info['count'])
				total_count += info['count']

			if debug:
				print(f'avgs   - {obj_id}: {averages} ({total_count})')
				print(averages.items())
			
			# Sort labels by descending count, then descending average, then ascending label to break ties
			sorted_labels = sorted(averages.items(), key=lambda x: ((-x[1][1], x[1][0]), x[0]))
			if debug:
				print(f'sorted - {obj_id}: {sorted_labels}')
			top_three = sorted_labels[:3]  # Take up to top three labels
			
			# Format the line parts
			line_parts = [obj_id]
			for label, (avg, count) in top_three:
				line_parts.append(f"{label} {avg:.3f} {count}")
			
			# Join parts and write to file
			f_out.write(' - '.join(line_parts) + '\n')

if __name__ == "__main__":
	main(debug=False)
