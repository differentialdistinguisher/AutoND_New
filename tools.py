import numpy as np


import numpy as np
import os
import copy
import pandas as pd


def dir_create(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
            
import os
import sys
class Logger(object):

    def __init__(self, logpath,logname,stream=sys.stdout):
        output_dir = logpath
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # log_name = logname + "_"+str(param.Round)+'r_'+str(param.group_size)+'.log'
        log_name = logname +'.log'
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

'''
import sys
sys.stdout = tools.Logger(logpath="./test/", logname="test")
'''


def int_to_bin_array(num, n):
    # 将整数转换为二进制比特字符串
    binary_str = bin(num)[2:]
    # 在字符串前面补零，达到指定的长度
    binary_str = '0' * (n - len(binary_str)) + binary_str
    # 将二进制字符串转换为numpy数组
    binary_array = np.array([int(bit) for bit in binary_str], dtype=np.uint8)
    return binary_array

'''
num = 2**32-1
n = 32
print(int_to_bin_array(num, n))  
num = 2**32-2
print(int_to_bin_array(num, n))  
'''



def integersToBitArray(arr, num_bits):
    bit_array = np.array([ int_to_bin_array(num,num_bits) for num in arr])
    return bit_array

def bitArrayToIntegers(arr):
    packed = np.packbits(arr,  axis = 1)
    return [int.from_bytes(x.tobytes(), 'big') for x in packed]

'''
num_bits = 32
num = 5

diffs = np.random.randint(2, size = (num, num_bits), dtype=np.uint8)
print("*"*50)
print("diffs",type(diffs))
print(diffs)

diffs_int = bitArrayToIntegers(diffs)
print("diffs_int",type(diffs_int))
print(diffs_int)
print(bin(diffs_int[0]))
diffs_new = integersToBitArray(diffs_int,num_bits)
# 实现函数bitArrayToIntegers的逆操作integersToBitArray使得res1等于diffs
print("diffs_new",type(diffs_new))
print(diffs_new)

diffs_new_int = bitArrayToIntegers(diffs_new)
print("diffs_int",type(diffs_new_int))
print(diffs_new_int)

print(np.all(diffs==diffs_new))
print(np.all(diffs_int==diffs_new_int))
'''

def hex_to_bin_weight(hex_str):
    # 将16进制字符串转换为二进制字符串
    bin_str = bin(int(hex_str, 16))[2:]

    # 计算二进制中置1的位数
    weight = bin_str.count('1')

    return weight
