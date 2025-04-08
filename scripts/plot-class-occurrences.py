#!/usr/bin/env python3
import matplotlib.pyplot as plt

'''
# Data provided as a list of tuples (count, bin)
data = [
    (15, 0), (2, 2), (28, 7), (326, 8),
    (15, 9), (1, 10), (176, 14), (5, 16),
    (36, 18), (88, 32), (3, 38), (35, 43),
    (16, 52), (14, 53), (31, 54), (2, 70),
    (9, 72), (9, 74), (2, 75), (101, 81),
    (41, 82), (22, 83), (1, 84), (116, 88),
    (26, 89), (1, 94)
]
'''

# Read the filename via argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Input filename')
args = parser.parse_args()
filename = args.filename

# Read data line by line, in tuples, space-separated
data = []
with open(filename, 'r') as f:
	for line in f:
		count, bin_pos = line.split()
		data.append((int(count), int(bin_pos)))

# Extracting bin positions and counts
x = [bin_pos for count, bin_pos in data]
y = [count for count, bin_pos in data]

# Creating the bar plot
plt.bar(x, y, color='blue')

# Adding labels and title
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class occurrences')

# Display the plot
plt.show()
