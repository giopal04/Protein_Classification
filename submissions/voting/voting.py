#!/usr/bin/env python3

import sys
from collections import defaultdict

def read_file1_or_2(file_path):
	data = defaultdict(list)
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split(',')
			sample_id = int(parts[0].strip())
			#print(f'sample_id: {sample_id}')
			class_ = int(parts[1].strip().zfill(2))
			#print(f'class_: {class_}')
			confidence = float(parts[2].strip())
			data[sample_id].append((class_, confidence))
	return data

def read_file3(file_path):
	data = defaultdict(list)
	with open(file_path, 'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split(',')
			#print(f'parts: {parts}')
			sample_id = int(parts[0].strip().zfill(4))
			# Process class
			class_str = parts[1].strip()[1:-1]  # Remove brackets
			#print(f'class_str: {class_str}')
			class_list = class_str.split(':')
			#print(f'class_list: {class_list}')
			class_ = class_list[0].strip().zfill(2)
			#print(f'class_: {class_}')
			# Process confidence
			conf_str = parts[2].strip()[1:-1]
			#print(f'{conf_str}')
			conf_list = conf_str.split(':')
			confidence = float(conf_list[0].strip())
			data[sample_id].append((int(class_), confidence))
	return data

def read_frequency(file_path):
    freq = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_ = parts[0].strip()
            count = int(parts[1].strip())
            freq[class_] = count
    return freq

def main(file1_path, file2_path, file3_path, freq_path):
	data_dict = defaultdict(list)
	# Read and merge data from all three files
	for sample_id, entries in read_file1_or_2(file1_path).items():
		data_dict[int(sample_id)].extend(entries)
	for sample_id, entries in read_file1_or_2(file2_path).items():
		data_dict[int(sample_id)].extend(entries)
	for sample_id, entries in read_file3(file3_path).items():
		data_dict[int(sample_id)].extend(entries)

	#for sample_id in sorted(data_dict.keys()):
	#	print(f'--- {sample_id}')
	
	freq_dict = read_frequency(freq_path)
	
	for sample_id in sorted(data_dict.keys()):
		entries = data_dict[sample_id]
		classes = [e[0] for e in entries]
		#confidences = [f'{e[1]:.2f}' for e in entries]
		confidences = [float(e[1]) for e in entries]
		
		class_counts = defaultdict(int)
		for cls in classes:
			class_counts[cls] += 1
		max_count = max(class_counts.values())
		
		if max_count >= 2:
			# Majority vote
			selected_class = max(class_counts, key=lambda k: (class_counts[k], k))
			data_str = f'{sample_id} - {classes} - {[f"{c:.2f}" for c in confidences]}'
			reason   = f'there is a majority vote'
			print(f'{data_str:50s} > {reason:30s} -> {sample_id},{selected_class}')
		else:
			# All classes are different, apply complex logic
			min_conf    = min(confidences)
			max_conf    = max(confidences)
			#print(f'{sample_id} - {min_conf} - {max_conf} - {type(min_conf)} - {type(max_conf)}')
			if float(max_conf) >= 0.9 and min_conf >= 0.8:
				remaining = []
			else:
				min_indices = [i for i, c in enumerate(confidences) if c == min_conf]
				remaining   = [e for i, e in enumerate(entries) if i not in min_indices]
			
			if not remaining:
				# All entries had minimum confidence, use frequency of all
				max_freq = -1
				selected_class = classes[0]
				for cls in classes:
					freq = freq_dict.get(cls, 0)
					if freq > max_freq or (freq == max_freq and cls < selected_class):
						max_freq = freq
						selected_class = cls
				data_str = f'{sample_id} - {classes} - {[f"{c:.2f}" for c in confidences]}'
				reason   = f'all equal confidences'
				print(f'{data_str:50s} > {reason:30s} -> {sample_id},{selected_class}')
			else:
				# Check remaining entries' frequencies
				remaining_classes = [e[0] for e in remaining]
				if len(remaining_classes) == 1:
					selected_class = remaining_classes[0]
					data_str = f'{sample_id} - {classes} - {[f"{c:.2f}" for c in confidences]}'
					reason   = f'only one high confidence'
					print(f'{data_str:50s} > {reason:30s} -> {sample_id},{selected_class}') # TODO: add a threshold for this condition
				else:
					# Compare frequencies of the two
					cls0, cls1 = remaining_classes[0], remaining_classes[1]
					freq0 = freq_dict.get(cls0, 0)
					freq1 = freq_dict.get(cls1, 0)
					if freq0 > freq1 or (freq0 == freq1 and cls0 < cls1):
						selected_class = cls0
						data_str = f'{sample_id} - {classes} - {[f"{c:.2f}" for c in confidences]}'
						reason   = f'picked up more frequent class'
						print(f'{data_str:50s} > {reason:30s} -> {sample_id},{selected_class}')
					else:
						selected_class = cls1
						data_str = f'{sample_id} - {classes} - {[f"{c:.2f}" for c in confidences]}'
						reason   = f'picked up more frequent class'
						print(f'{data_str:50s} > {reason:30s} -> {sample_id},{selected_class}')
		
		#print(f"{sample_id},{selected_class}")

if __name__ == "__main__":
	if len(sys.argv) != 5:
		print("Usage: python script.py file1.csv file2.csv file3.csv freq.txt")
		sys.exit(1)
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
