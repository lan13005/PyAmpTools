#!/usr/bin/env python

import re
import sys
from collections import Counter

'''
Counts the occurence of hexadecimal addresses in a text file.
This is useful if you have a giant print log (i.e. from Amptools debugging)
to try and catch unsuspecting points pointing to same address
'''

def count_hex_addresses(file_path):
    hex_pattern = re.compile(r'\b0x[0-9a-fA-F]+\b')

    with open(file_path, 'r') as file:
        content = file.read()
        hex_addresses = hex_pattern.findall(content)
        count = Counter(hex_addresses)

    return count

# Example usage
argc = len(sys.argv)
assert( argc == 2 )
file_path = sys.argv[1]  # Replace with your file path
hex_count = count_hex_addresses(file_path)

# Display the results
for address, count in hex_count.items():
    if count > 1:
        print(f"{address}: {count} occurrences")
