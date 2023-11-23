# A new (related-key) neural distinguisher considering two differentials
## 1. The folder **ciphers** contains the cipher to be analyzed.

- Simon
- Speck
- Simeck
- Hight


## 2. If you want to run our code, please follow the pattern below
```bash
python main.py cipher scenario strategy -o out_dir
```
- cipher: the name of the cipher to be analyzed in the folder **ciphers**.

- scenario: the scenario, either single-key or related-key', default = 'single-key'. 
- strategy: the strategy to perform differential pair selection, either 1 or 2', default = '2'. 
- out_dir: the folder to store the experiments results, default ='results'.
  
- For example
    ```bash
    python main speck3264 single-key 2 -o results
    ```

## 3. The folder **models** gives our enhanced neural distinguishers.

-   If you want to evaluate our neural distinguishers on fresh test datasets, please execute the following code
    ```bash

    python eval_nets_pairs.py --cipher speck3264 --scenario single-key  --input_difference 0x400000_0x502000 --round_number 8 --model_path ./models/speck3264_single-key_polish_dbitnet_round8_best_polish2.h5  --pairs 1

    python eval_nets_pairs.py --cipher simon3264 --scenario single-key  --input_difference 0x400_0x100 --round_number 11 --model_path ./models/simon3264_single-key_polish_dbitnet_round11_best_polish2.h5  --pairs 1

    python eval_nets_pairs.py --cipher simon3264 --scenario single-key  --input_difference 0x40_0x100 --round_number 12 --model_path ./models/simon3264_single-key_polish_dbitnet_round12_best_polish2.h5  --pairs 1

    python eval_nets_pairs.py --cipher simeck3264 --scenario single-key  --input_difference 0x4000_0x8000 --round_number 11 --model_path ./models/simeck3264_single-key_polish_dbitnet_round11_best_polish2.h5  --pairs 1

    python eval_nets_pairs.py --cipher simeck3264 --scenario single-key  --input_difference 0x4000_0x2000 --round_number 12 --model_path ./models/simeck3264_single-key_polish_dbitnet_round12_best_polish2.h5  --pairs 1

    ```

- For example, running 
```bash
python eval_nets_pairs.py --cipher speck3264 --scenario single-key  --input_difference 0x400000_0x502000 --round_number 8 --model_path ./models/speck3264_single-key_polish_dbitnet_round8_best_polish2.h5  --pairs 1
```
results in
```bash
INFO:root:Creating model 'dbitnet' with weights from path 
         './models/speck3264_single-key_polish_dbitnet_round8_best_polish2.h5'...
INFO:root:Creating new validation dataset for 
                        cipher: speck3264,
                        scenario: single-key,
                        input difference: 0x400000_0x502000,
                        round number: 8,
                        pairs: 1
                        cipher.plain_bits: 32,
                        cipher.key_bits: 64
                        ...
INFO:root:       acc=0.5193      tpr=0.5210      tnr=0.5175
INFO:root:       acc=0.5189      tpr=0.5195      tnr=0.5183
INFO:root:       acc=0.5188      tpr=0.5206      tnr=0.5170
INFO:root:       acc=0.5188      tpr=0.5206      tnr=0.5170
INFO:root:       acc=0.5188      tpr=0.5197      tnr=0.5178

```
## 4. References
```bash
[1] Bellini E, Gerault D, Hambitzer A, et al. A Cipher-Agnostic Neural Training Pipeline with Automated Finding of Good Input Differences[J]. Cryptology ePrint Archive, 2022. https://github.com/Crypto-TII/AutoND

```
