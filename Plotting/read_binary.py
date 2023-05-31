import os
import struct

import numpy as np

filename = 'test_arr.bin'
file_size = os.path.getsize(filename)

element_size = struct.calcsize('B')
num_elements = file_size//element_size

print('num elements',num_elements)

with open(filename,'rb') as f:
    binary_data = f.read()
    char_array = struct.unpack(f'{num_elements}B',binary_data)

print(char_array)