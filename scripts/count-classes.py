#!/usr/bin/env python3

# Usage: cat files.txt | ./count-classes.py

import sys
from collections import defaultdict

def main():
    counts = defaultdict(int)
    
    for line in sys.stdin:
        filename = line.strip()
        parts = filename.split('-')
        if len(parts) >= 2:
            class_id = parts[1]
            counts[class_id] += 1
    
    # Sort classes numerically by converting to integer
    sorted_classes = sorted(counts.items(), key=lambda x: int(x[0]))
    
    with open('class_counts.txt', 'w') as f:
        for class_id, count in sorted_classes:
            f.write(f"{class_id} {count}\n")

if __name__ == "__main__":
    main()
