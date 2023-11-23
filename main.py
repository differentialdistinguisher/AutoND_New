import os
import gc
from os import urandom
from glob import glob
import importlib
import numpy as np
import os
import optimizer
import train_nets
import argparse

import csv


import datetime
current_time  = datetime.datetime.now()
time_string  = current_time.strftime("%Y-%m-%d_%H-%M-%S")



def make_train_samples(encryption_function, plain_bits, key_bits, n, nr, diff_type=0, delta_state=0, delta_key=0):
    keys0 = (np.frombuffer(urandom(n*key_bits),dtype=np.uint8)&1).reshape(n, key_bits);
    pt0 = (np.frombuffer(urandom(n*plain_bits),dtype=np.uint8)&1).reshape(n, plain_bits);
    C0 = encryption_function(pt0, keys0, nr)
    if diff_type == 1:
        pt1 = pt0^delta_state
        keys1 = keys0^delta_key
        C1 = encryption_function(pt1, keys1, nr)
        del keys1
        del pt1
        gc.collect()
        
    else:
        C1 = (np.frombuffer(urandom(n*C0.shape[1]),dtype=np.uint8)&1).reshape(n, -1)        
    C = np.hstack([C0, C1])
    del keys0
    del pt0
    del C0
    del C1
    gc.collect()
    
    return C

def make_train_data(encryption_function, plain_bits, key_bits, n, nr, delta_states=None, delta_keys=None):
    num = n // 2
    assert n % 2 == 0
    X0 = make_train_samples(encryption_function, plain_bits, key_bits, num, nr, diff_type=1, delta_state=delta_states[0], delta_key=delta_keys[0])
    if len(delta_states) == 1:
        X1 = make_train_samples(encryption_function, plain_bits, key_bits, num, nr, diff_type=0)
    else:
        X1 = make_train_samples(encryption_function, plain_bits, key_bits, num, nr, diff_type=1, delta_state=delta_states[1], delta_key=delta_keys[1])
    X = np.concatenate((X0, X1), axis=0).reshape(n, -1)
    del X0
    del X1
    gc.collect()
    Y0 = np.zeros((num,),dtype=np.uint8)
    Y1 = np.ones((num,),dtype=np.uint8)
    Y = np.concatenate((Y0, Y1))
    del Y0
    del Y1
    gc.collect()
    # print(X.shape,Y.shape)
    return X, Y


# From an integer, returns a 1 X num_bits numpy uint8 array
def integer_to_binary_array(int_val, num_bits):
    # return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8).reshape(1, num_bits)
    return np.array([int(i) for i in bin(int_val)[2:].zfill(num_bits)], dtype = np.uint8)


# From a 1 X num_bits numpy uint8 array, returns an integer
def binary_array_to_integer(binary_array):
    # print(binary_array)
    return int(''.join(map(str, binary_array)), 2)

def findGoodInputDifferences(cipher_name, scenario, output_dir, epsilon = 0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cipher = importlib.import_module('ciphers.' + cipher_name, package='ciphers')
    s = cipher.__name__[8:] + "_" + scenario
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    word_size = cipher.word_size
    encryption_function = cipher.encrypt
    best_differences, highest_round = optimizer.optimize(plain_bits, key_bits, encryption_function, scenario = scenario, log_file=f'{output_dir}/{s}', epsilon = epsilon)
    return best_differences, highest_round


def trainNeuralDistinguishers(types, cipher_name, scenario, output_dir, input_differences, starting_round, epochs = None, nets =['gohr', 'dbitnet'], num_samples=None):
    cipher = importlib.import_module('ciphers.' + cipher_name, package='ciphers')
    s = cipher.__name__[8:] + "_" + scenario
    plain_bits = cipher.plain_bits
    key_bits = cipher.key_bits
    word_size = cipher.word_size
    encryption_function = cipher.encrypt
    if not os.path.exists(f'{output_dir}/{types}/'):
        os.makedirs(f'{output_dir}/{types}/')

    delta_plains = []
    delta_keys = []
    if scenario == "related-key":
        for input_difference in  input_differences:
            delta = integer_to_binary_array(input_difference, plain_bits+key_bits)
            delta_plains.append(delta[:plain_bits])
            delta_keys.append(delta[plain_bits:])                 
    else:
        for input_difference in  input_differences:
            delta_plains.append(integer_to_binary_array(input_difference, plain_bits))
            delta_keys.append(0)            
    
    diff_str = "_".join([str(hex(binary_array_to_integer(delta_plain))) for delta_plain in delta_plains])
    if scenario == "related-key":
        diff_str += "_"+"_".join([str(hex(binary_array_to_integer(delta_key))) for delta_key in delta_keys])
    results = {}

    input_size = plain_bits
    results['Difference'] = diff_str
    for net in nets:
        print(f'Training {net} for input difference {diff_str}, starting from round {starting_round}...')
        results[net] = {}
        best_round, best_val_acc, intermediate_results = train_nets.train_neural_distinguisher(
            starting_round = starting_round,
            data_generator = lambda num_samples, nr : make_train_data(encryption_function, plain_bits, key_bits, num_samples, nr, delta_plains, delta_keys),
            model_name = net,
            input_size = input_size,
            word_size = word_size,
            
            log_prefix = f'{output_dir}/{types}/{s}_{diff_str}',
            _epochs = epochs,
            _num_samples = num_samples)
        results[net]['Best round'] = best_round
        results[net]['Validation accuracy'] = best_val_acc
        
        file_path = f"./{output_dir}/{cipher_name}{time_string}.csv"
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['types', 'cipher_name', 'scenario', 'diff_str', 'round', 'model_name', 'val_acc'])
            for row in intermediate_results:
                writer.writerow([types, cipher_name, scenario, diff_str, row["round"], net, row["val_acc"]])

    return results


def parse_and_validate():
    ciphers_list = glob('ciphers/*.py')
    ciphers_list = [cipher[8:-3] for cipher in ciphers_list]
    scenarios_list = ['single-key', 'related-key']
    parser = argparse.ArgumentParser(description='Obtain good input differences for neural cryptanalysis.')
    parser.add_argument('cipher', type=str, nargs='?', default = 'speck3264',
            help=f'the name of the cipher to be analyzed, from the following list: {ciphers_list}')
    parser.add_argument('scenario', type=str, nargs='?',
            help=f'the scenario, either single-key or related-key', default = 'single-key')
    parser.add_argument('strategy', type=str, nargs='?',
            help=f'The strategy to perform differential pair selection, either 1 or 2', default = '2')
    parser.add_argument('-o', '--output', type=str, nargs='?', default ='results',
            help=f'the folder where to store the experiments results')
    
    arguments = parser.parse_args()
    cipher_name = arguments.cipher
    scenario = arguments.scenario
    strategy = arguments.strategy
    output_dir = arguments.output
    
    if cipher_name not in ciphers_list:
        raise Exception(f'Cipher name error: it has to be one of {ciphers_list}.')
    if scenario not in scenarios_list:
        raise Exception(f'Scenario name error: it has to be one of {scenarios_list}.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return cipher_name, scenario, strategy, output_dir


    
def main_strategy1(cipher_name, scenario, output_dir, best_differences,highest_round):
    print("=" * 70)
    print(f"PART 2: Training DBitNet with two differences using the simple training pipeline...")
    types = "twodiff"
    for best_diff in best_differences:
        diff1 = best_diff
        diff2s = optimizer.getTopDiffs(f'./{output_dir}/differences/{cipher_name + "_" + scenario}', max(len(best_differences),10))
        results = []
        best_round = -1
        best_val_acc = -1
        for diff2 in diff2s:
            input_differences = []
            if diff2 != diff1:
                diff2 = int(diff2)
                input_differences.append(diff1)
                input_differences.append(diff2)
            else:
                continue
            print(diff2s)
            print((diff1),(diff2))
            print(hex(diff1),hex(diff2))
            result = trainNeuralDistinguishers(types, cipher_name, scenario, output_dir, input_differences, max(1, highest_round-2), nets =['dbitnet'])
            print("=" * 70)
            print(result)
            results.append(result)
            if result["dbitnet"]['Best round'] is None:
                    continue
            if result["dbitnet"]['Best round'] > best_round or (result["dbitnet"]['Best round'] == best_round and result["dbitnet"]['Validation accuracy'] > best_val_acc ):
                # best_diff = input_differences
                best_round = result["dbitnet"]['Best round']
                best_val_acc = result["dbitnet"]['Validation accuracy']
                print("better result: ", result)
            # break
        print(results)
        
    
    
def main_strategy2(cipher_name, scenario, output_dir, best_differences,highest_round):
    
    print(f"PART 2: Training DBitNet using the simple training pipeline...")
    types = "normal"
    results = []
    best_diff = -1
    best_round = -1
    best_val_acc = -1
    for input_difference in best_differences:
        print("input_difference", hex(input_difference))
        result = trainNeuralDistinguishers(types, cipher_name, scenario, output_dir, [input_difference], max(1, highest_round-2), nets =['dbitnet'])
        results.append(result)
        if result["dbitnet"]['Best round'] > best_round or (result["dbitnet"]['Best round'] == best_round and result["dbitnet"]['Validation accuracy'] > best_val_acc ):
            best_diff = input_difference
            best_round = result["dbitnet"]['Best round']
            best_val_acc = result["dbitnet"]['Validation accuracy']
            print("best difference: ", result)
        
    print("=" * 70)
    print(f"PART 3: Training DBitNet with two differences using the simple training pipeline...")
    types = "twodiff"
    diff1 = best_diff
    diff2s = optimizer.getTopDiffs(f'./{output_dir}/differences/{cipher_name + "_" + scenario}', max(len(best_differences),10))
    results = []
    best_round = -1
    best_val_acc = -1
    for diff2 in diff2s:
        input_differences = []
        if diff2 != diff1:
            diff2 = int(diff2)
            input_differences.append(diff1)
            input_differences.append(diff2)
        else:
            continue
        
        print(hex(diff1),hex(diff2))
        result = trainNeuralDistinguishers(types, cipher_name, scenario, output_dir, input_differences, max(1, highest_round-2), nets =['dbitnet'])
        print("=" * 70)
        
        results.append(result)
        if result["dbitnet"]['Best round'] is None:
                continue
        if result["dbitnet"]['Best round'] > best_round or (result["dbitnet"]['Best round'] == best_round and result["dbitnet"]['Validation accuracy'] > best_val_acc ):
            # best_diff = input_differences
            best_round = result["dbitnet"]['Best round']
            best_val_acc = result["dbitnet"]['Validation accuracy']
            print("better result: ",result)
        # break
    print(results)
    

    
if __name__ == "__main__":
    cipher_name, scenario, strategy, output_dir = parse_and_validate()
    s = cipher_name + "_" + scenario
    
    # FindGoodInputDifferences
    epsilon = 0.1
    print("\n")
    print("=" * 70)
    print(f"PART 1: Finding the {epsilon}-close input differences and the `highest round` using the evolutionary optimizer for ", s, "...")
    best_differences, highest_round = findGoodInputDifferences(cipher_name, scenario, output_dir=output_dir+"/differences", epsilon=epsilon)
    print(best_differences)
    print(f'Found {len(best_differences)} {epsilon}-close differences: {[hex(x) for x in best_differences]}. \nThe highest round with a bias score above the threshold was {highest_round}. \nThe best differences and their scores for each round are stored under {output_dir}/{s}, and the full list of differences along with their weighted scores are stored under {output_dir}/{s}_best_weighted_differences.csv.')
    print("=" * 70)
    if strategy == "1":
        main_strategy1(cipher_name, scenario, output_dir, best_differences,highest_round)
    else:
        main_strategy2(cipher_name, scenario, output_dir, best_differences,highest_round)
        
    
