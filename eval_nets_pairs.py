import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import tensorflow as tf
import numpy as np

import logging
import glob
import argparse
import importlib

import main as autond

# ------------------------------------------------
# Configuration and constants
# ------------------------------------------------

# logging.basicConfig(level=logging.INFO)
# logging.getLogger().setLevel(logging.INFO)

# 创建logger对象
logger = logging.getLogger()

# 创建FileHandler对象
file_handler = logging.FileHandler('eval_results.txt', mode='a')
file_handler.setLevel(logging.INFO)

# 创建ConsoleHandler对象
console_handler =  logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建Formatter对象
formatter = logging.Formatter('%(asctime)s - %(message)s')

# 将Formatter对象添加到FileHandler对象和ConsoleHandler对象
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将FileHandler对象和ConsoleHandler对象添加到logger对象
logger.addHandler(file_handler)
# logger.addHandler(console_handler)
# 设置日志级别
logger.setLevel(logging.INFO)

nEval = 5 # number of evaluation datasets which are freshly generated
num_val_samples = 10**6 # number of samples in each of the evaluation datasets
batchsize = 5000

def evaluate_X_Y(model, X, Y, pairs):
    """Returns the accuracy, TPR, and TNR of the model(X) with ground-truth Y.
    (Taken from Gohr's repository)
    """
    # ------------ calculate TPR and TNR
    z_atomic = model.predict(X, batch_size=batchsize,verbose = 0)
    z_atomic = z_atomic.reshape(-1, pairs)
    # combine scores under independence assumption
    prod_enc = z_atomic.prod(axis=1)
    prod_rnd = (1 - z_atomic).prod(axis=1)
    # avoid division by zero
    z = 1 / (1 + np.divide(prod_rnd, prod_enc, out=np.zeros_like(prod_enc), where=(prod_enc != 0)))
    # decide score based on the number of zeros in numerator and denominator
    z[prod_enc == 0] = np.sum(z_atomic[prod_enc == 0] == 0, axis=1) < np.sum(z_atomic[prod_enc == 0] == 1, axis=1)
    
    Y = Y.reshape(-1, pairs)
    Y = Y.sum(axis=1)/pairs


    # evaluate accuracy, tpr, tnr and mse
    z_bin = (z > 0.5)
    # diff = Y - z
    # mse = np.mean(diff * diff)
    n = len(z)
    n0 = np.sum(Y == 0)
    n1 = np.sum(Y == 1)
    acc = np.sum(z_bin == Y) / n
    tpr = np.sum(z_bin[Y == 1]) / n1
    tnr = np.sum(z_bin[Y == 0] == 0) / n0

    return acc, tpr, tnr

def evaluate_Xlist_Ylist(model, Xlist, Ylist, pairs):

    allAccs, allTPRs, allTNRs = [], [], []

    for X, Y in zip(Xlist, Ylist):
        acc, tpr, tnr = evaluate_X_Y(model, X, Y, pairs)

        allAccs.append(acc)
        allTPRs.append(tpr)
        allTNRs.append(tnr)

        logging.info(f"\t acc={acc:.4f} \t tpr={tpr:.4f} \t tnr={tnr:.4f}")

    return allAccs, allTPRs, allTNRs

def get_deltas_from_scenario(scenario, input_differences, plain_bits, key_bits):
    """Returns delta_plain, delta_key for the scenario."""
    
    delta_plains = []
    delta_keys = []
    for input_difference in input_differences:
        
        if scenario == "related-key":
            delta = autond.integer_to_binary_array(input_difference, plain_bits + key_bits)
            # delta_key = delta[:, plain_bits:]
            delta_key = delta[plain_bits:]
        elif scenario == "single-key":
            delta = autond.integer_to_binary_array(input_difference, plain_bits)
            delta_key = 0
        else:
            raise ValueError(f"An unknown scenario '{scenario}' was encountered.")
        delta_plain = delta[:plain_bits]
        delta_plains.append(delta_plain)
        delta_keys.append(delta_key)
        
    return delta_plains,delta_keys

def parseTheCommandLine(parser):
    # ---- add arguments to parser
    # model arguments
    parser.add_argument('--model_path',
                        type=str,
                        required=False,
                        help=f'The path to the h5 file with the model weights.')
    parser.add_argument('--model_type',
                        type=str,
                        default='dbitnet',
                        required=False,
                        choices=['gohr-depth1', 'dbitnet'],
                        help=f'The model type (gohr-depth1 or dbitnet) of the model h5 file.')

    # cipher arguments
    parser.add_argument('--cipher',
                        type=str,
                        default='speck3264',
                        required=True,
                        help=f'The name of the cipher to be analyzed, from the following list: {ciphers_list}.')
    parser.add_argument('--scenario',
                        type=str,
                        required=False,
                        choices=['single-key', 'related-key'],
                        help=f'The scenario, either single-key or related-key',
                        default = 'single-key')

    # dataset arguments
    parser.add_argument('--dataset_path_X',
                        #type=str,
                        required=False,
                        nargs='+',
                        help=f'Optional path to a pre-existing *.npy dataset file TODO format-hint')
    parser.add_argument('--dataset_path_Y',
                        #type=str,
                        required=False,
                        nargs='+',
                        help=f'Optional path to a pre-existing *.npy dataset file TODO format-hint')
    parser.add_argument('--input_difference',
                        type=str,
                        required=False,
                        help=f"The input difference for the data generation, e.g. '0x400000_0x102000'.")
    parser.add_argument('--round_number',
                        type=str,
                        required=False,
                        help=f"The round number for the data generation, e.g. 5.")
    parser.add_argument('--pairs',
                        type=str,
                        default='1',
                        required=False,
                        help=f"The round number for the data generation, e.g. 5.")

    # results arguments
    parser.add_argument('--add_str',
                        type=str,
                        required=False,
                        default='',
                        help=f'Add an additional string to the evaluation filename.')
    # ---------------------------------------------------
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # create a list of available ciphers in the ciphers folder:
    ciphers_list = glob.glob('ciphers/*.py')
    ciphers_list = [cipher[8:-3] for cipher in ciphers_list]

    # ---------------------------------------------------
    # Parse arguments from command line
    # ---------------------------------------------------
    parser = argparse.ArgumentParser(
        description='Evaluate an existing neural distinguisher.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    args = parseTheCommandLine(parser)

    # ---------------------------------------------------
    # infer from the command line arguments:
    cipher = importlib.import_module('ciphers.' + args.cipher, package='ciphers')
    input_size = cipher.plain_bits

    # create the filename for the results (the same path as the model, just with ending '_eval.npz')
    filename_results = args.model_path.replace('.h5', f'_eval{args.add_str}.npz')

    # ---------------------------------------------------
    logger.info(f"Creating model '{args.model_type}' with weights from path \n\t '{args.model_path}'...")
    if args.model_type == 'gohr-depth1':
        import gohrnet
        model = gohrnet.make_model(2*input_size, word_size=cipher.word_size)
    elif args.model_type == 'dbitnet':
        import dbitnet
        model = dbitnet.make_model(2*input_size)
    else:
        logger.fatal(f"Model creation failed for model_type={args.model_type}.")
        exit()

    # the optimizer and loss don't really matter for the evaluation we do here:
    optimizer = tf.keras.optimizers.Adam(amsgrad=True)
    model.compile(optimizer=optimizer, loss='mse', metrics=['acc'])
    model.load_weights(args.model_path)
    
    if (args.input_difference is not None) & (args.round_number is not None):
        logger.info(f"""Creating new validation dataset for 
                        cipher: {args.cipher}, 
                        scenario: {args.scenario}, 
                        input difference: {args.input_difference}, 
                        round number: {args.round_number}, 
                        pairs: {args.pairs}
                        cipher.plain_bits: {cipher.plain_bits}, 
                        cipher.key_bits: {cipher.key_bits}
                        ...""")

        # input_difference: convert str to hexadecimal int
        input_difference_strs = args.input_difference.split("_")
        input_differences = []
        for input_difference_str in input_difference_strs:
            input_differences.append(int(input_difference_str, base=16))

        args.round_number = int(args.round_number)
        args.pairs = int(args.pairs)
        

        delta_plains, delta_keys = get_deltas_from_scenario(args.scenario,
                                                          input_differences,
                                                          cipher.plain_bits,
                                                          cipher.key_bits)
        # print(delta_plains, delta_keys)
        data_generator = lambda num_samples, nr: autond.make_train_data(cipher.encrypt,
                                                                         cipher.plain_bits,
                                                                         cipher.key_bits,
                                                                         num_samples,
                                                                         nr,
                                                                         delta_plains,
                                                                         delta_keys)
        
        if args.pairs <= 8:
            Xlist = []
            Ylist = []
            for i in range(nEval):
                X, Y = data_generator(num_val_samples * args.pairs, args.round_number)
                Xlist.append(X)
                Ylist.append(Y)
            accs, tprs, tnrs = evaluate_Xlist_Ylist(model, Xlist, Ylist, args.pairs)
        else:
            for i in range(nEval):
                accs, tprs, tnrs = [], [], []
                groups = 10
                for j in range(groups):
                    X, Y = data_generator(num_val_samples // groups * args.pairs, args.round_number)
                    acc, tpr, tnr = evaluate_X_Y(model, X, Y, args.pairs)
                    accs.append(acc)
                    tprs.append(tpr)
                    tnrs.append(tnr)
                
                logging.info(f"\t acc={sum(accs)/len(accs):.4f} \t tpr={sum(tprs)/len(tprs):.4f} \t tnr={sum(tnrs)/len(tnrs):.4f}")
            
    else:
        logger.fatal("Please, pass one of the arguments: 'data_path', ('input_difference' and 'round_number').")
        exit()

'''

python eval_nets_pairs.py --cipher speck3264 --scenario single-key  --input_difference 0x400000_0x502000 --round_number 8 --model_path ./models/speck3264_single-key_polish_dbitnet_round8_best_polish2.h5  --pairs 1

python eval_nets_pairs.py --cipher simon3264 --scenario single-key  --input_difference 0x400_0x100 --round_number 11 --model_path ./models/simon3264_single-key_polish_dbitnet_round11_best_polish2.h5  --pairs 1
python eval_nets_pairs.py --cipher simon3264 --scenario single-key  --input_difference 0x40_0x100 --round_number 12 --model_path ./models/simon3264_single-key_polish_dbitnet_round12_best_polish2.h5  --pairs 1

python eval_nets_pairs.py --cipher simeck3264 --scenario single-key  --input_difference 0x4000_0x8000 --round_number 11 --model_path ./models/simeck3264_single-key_polish_dbitnet_round11_best_polish2.h5  --pairs 1
python eval_nets_pairs.py --cipher simeck3264 --scenario single-key  --input_difference 0x4000_0x2000 --round_number 12 --model_path ./models/simeck3264_single-key_polish_dbitnet_round12_best_polish2.h5  --pairs 1

'''