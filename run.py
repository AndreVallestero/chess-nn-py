import sys
from numpy import zeros, uint8, empty, tanh, sum, exp
'''
Param1: fen_str: String in FEN notation
Param2(optional): weights: Weight list string

If Param2 doesn't exist, search for a weights file in root/working directory(weights.json)
'''


if len(sys.argv) < 1:
    print('Missing parameters\n\t- fen (str): Board state as FEN\n\t- [weights] (str): Weights as a matrix')

def run(fen_str: str, weights: list) -> float:
    PIECES = 'PRBNKQ'
    CSTLS = 'KQkq'
    ENPAS_RANKS = [3, 6]

    inputs = zeros(855, dtype=uint8)

    segs = fen_str.split(' ')
    ranks = segs[0].split('/')
    i = 0
    for j in range(len(ranks)):
        for char in ranks[j]:
            if char.isdigit():
                for k in range(12, int(char)*13, 13):
                    inputs[i+k] = 1
                i += int(char) * 13
            else:
                offset = 6 if char.islower() else 0
                inputs[i + PIECES.index(char.upper()) + offset] = 1
                i += 13

    if 'w' == segs[1]:
        inputs[i] = 1
    i += 1

    for char in CSTLS:
        if char in segs[2]:
            inputs[i] = 1
    i += 4

    if '-' != segs[3]:
        rank = 8 if segs[3][1] == '6' else 0
        cfile = ord(segs[3][0]) - 97
        inputs[i + rank + cfile] = 1
    i += 16

    inputs[i] = int(segs[4])
    i += 1
    inputs[i] = int(segs[5])

    # For each neuron calculate all the weights and put it in the storage array
    input_layer = weights[0]
    results = empty(len(input_layer))
    alt_results = empty(len(input_layer))
    for j, genome in enumerate(input_layer):
        results[j] = tanh(sum((inputs + genome[0]) * genome[1]))

    for j, layer in enumerate(weights[1]):
        for genome in layer:
            if j % 2:
                results[j] = tanh(sum((alt_results + genome[0]) * genome[1]))
            else:
                alt_results[j] = tanh(sum((results + genome[0]) * genome[1]))
    
    output_layer = weights[2]
    if len(weights[1]) % 2:
        for j, genome in enumerate(output_layer):
            return sig(sum((alt_results + genome[0]) * genome[1]))
    else:
        for j, genome in enumerate(output_layer):
            return sig(sum((results + genome[0]) * genome[1]))


def sig(x):
    return 1 / (1 + exp(-x))

def print_board(board):
    PIECES = 'PRBNKQ'
    for i in range(8):
        print('')
        for j in range(8):
            for k in range(13):
                if board[i*8*13 + j*13 + k] != 0:
                    if k == 12:
                        print('-', end='')
                    else:
                        char = PIECES[k%6]
                        if k > 5:
                            char = char.lower()
                        print(char, end='')
    print('\n')
