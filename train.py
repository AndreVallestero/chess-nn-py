'''
python -m pip install -U
python -m pip install -U python-chess
'''

#https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
#https://towardsdatascience.com/my-experiments-in-replacing-deep-learning-backpropagation-sgd-with-a-genetic-algorithm-c6e308382926
#https://towardsdatascience.com/gas-and-nns-6a41f1e8146d
#https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164

#https://github.com/winstvn/mltonin-machine-learning/blob/master/training.ipynb
#CSV where every weight is saved as csv, each layer is a new line

import json
from os import listdir
from numpy import ascontiguousarray, random, float32, fromfile, ndarray, mean
from chess import pgn
from matplotlib import pyplot as plt

from run import run

INPUT_SIZE = 855
BOARDS_PER_EPOCH = 16
HBOARDS_PER_EPOCH = round(BOARDS_PER_EPOCH/2)
MAX_FLOAT = float(2) #float(1 << 8)
MIN_FLOAT = float(-MAX_FLOAT)

BLACK_CACHE = 'black-wins'
WHITE_CACHE = 'white-wins'

DEF_EXT = '.json'
DEF_TRAIN_DATA_DIR = 'training-data'
DEF_POP_SIZE = 64
DEF_POP_NAME = 'unnamed'
DEF_LAYER_NEURS = 2048
DEF_GENOME_LAYS = 4
DEF_EPOCHS = 100 #64

# Evolution consts
POP_EVO_WEIGHT = [1, # Preserve
                  #1, # Breed with equal or better and crossover
                  3, # Mutate
                  5] # Kill and generate from scratch

def main():
    pop = []

    # -------- Get USER INPUTS --------
    train_data_dir = input(f'Training data dir [{DEF_TRAIN_DATA_DIR}]: ').strip().lower()
    train_data_dir = DEF_TRAIN_DATA_DIR if train_data_dir in [''] else train_data_dir

    prev_pop_dir = input('Previous population dir [none]: ').strip().lower()

    if prev_pop_dir in ['', 'none']:
        pop_size = input(f'Population size [{DEF_POP_SIZE}]: ').strip().lower()
        pop_size = DEF_POP_SIZE if pop_size in [''] else int(pop_size)

        pop_name = input(f'Population name [{DEF_POP_NAME}]: ').strip().lower()
        pop_name = DEF_POP_NAME if pop_name in [''] else pop_name

        layer_neurs = input(f'Number of neurons per layer [{DEF_LAYER_NEURS}]: ').strip().lower()
        layer_neurs = DEF_LAYER_NEURS if layer_neurs in [''] else int(layer_neurs)

        genome_lays = input(f'Number of hidden layers per genome [{DEF_GENOME_LAYS}]: ').strip().lower()
        genome_lays = DEF_GENOME_LAYS if genome_lays in [''] else int(genome_lays)
    else:
        prev_pop_files = [name for name in listdir(prev_pop_dir) if name.lower().endswith(DEF_EXT)]
        prev_pop_size = len(prev_pop_files)
        prev_pop_name = prev_pop_dir.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]

        for genome_file in prev_pop_files:
            with open(genome_file) as f:
                genome_data = json.load(f)
                for i in enumerate(genome_data):
                    genome_data[i] = ascontiguousarray(genome_data[i], dtype='float32')
                pop.append(Genome(len(genome_data[1]), genome_data[1].shape[2], genome_data))

        pop_size = input(f'Population size [{prev_pop_size}]: ').strip().lower()
        pop_size = prev_pop_size if pop_size in ['', 'same'] else int(pop_size)

        pop_name = input(f'Population name [{prev_pop_name}]: ').strip().lower()
        pop_name = prev_pop_name if pop_name in ['', 'same'] else pop_name

        layer_neurs = len(pop[0][1])
        print(f'Number of neurons per layer: {layer_neurs}')

        genome_lays = len(pop[0]-2)
        print(f'Number of hidden layers per genome: {genome_lays}')

    epochs = input(f'Epochs [{DEF_EPOCHS}]: ').strip().lower()
    epochs = DEF_EPOCHS if epochs in [''] else int(epochs)

    # -------- GET TRAINING DATA --------
    train_data_files = listdir(train_data_dir)
    boards = []
    black_cache_matches = [filename for filename in train_data_files if BLACK_CACHE in filename]
    white_cache_matches = [filename for filename in train_data_files if WHITE_CACHE in filename]
    if any(black_cache_matches) and any(white_cache_matches):
        print('Found cached FEN array binaries, will reuse for training')
        black_cache_match = black_cache_matches[0]
        black_dtype = '<' + black_cache_match.rsplit('.', 1)[-1]
        white_cache_match = white_cache_matches[0]
        white_dtype = '<' + black_cache_match.rsplit('.', 1)[-1]
        boards.append(fromfile(f'{train_data_dir}/{black_cache_match}', dtype=black_dtype))
        boards.append(fromfile(f'{train_data_dir}/{white_cache_match}', dtype=white_dtype))
        print('Cached FENABs loaded')
    else:
        print('Indexing training PGN files, each "." represents 1 game')
        train_pgns = [name for name in train_data_files if name.lower().endswith('.pgn')]
        len_pgns = len(train_pgns)
        temp_boards = [[], []]
        for i, pgn_name in enumerate(train_pgns):
            print(f'{i+1} / {len_pgns} ', end='')
            with open(f'{train_data_dir}/{pgn_name}') as pgnf:
                pgn_game = pgn.read_game(pgnf)
                while pgn_game:
                    if pgn_game.headers['Result'] not in ['1/2-1/2', '*']:
                        print('.', end='', flush=True)
                        target_label = temp_boards[int(pgn_game.headers['Result'][0])]
                        try:
                            board = pgn_game.board()
                            for move in pgn_game.mainline_moves():
                                board.push(move)
                                target_label.append(board.fen())
                        except:
                            print('Error encountered while traversing board. Skipping')
                    try:
                        pgn_game = pgn.read_game(pgnf)
                    except:
                        print('Error encountered while reading game. Skipping')
                        continue
                print()
        boards.append(ascontiguousarray(temp_boards[0]))
        boards.append(ascontiguousarray(temp_boards[1]))
        print(f'Caching FEN array binaries in {train_data_dir}')
        boards[0].tofile(f'{train_data_dir}/{BLACK_CACHE}.{str(boards[0].dtype)[1:]}')
        boards[1].tofile(f'{train_data_dir}/{WHITE_CACHE}.{str(boards[1].dtype)[1:]}')
    len_boards0 = len(boards[0])
    len_boards1 = len(boards[1])

    # -------- PREPARE POPULATION --------
    print(f'Generating {pop_size} genomes: ', end='')
    while len(pop) < pop_size:
        print(len(pop)+1, end=' ', flush=True)
        pop.append(Genome(genome_lays, layer_neurs))
    pop = pop[0:pop_size]
    print()

    # -------- CONFIGURE EVOLUTION --------
    pop_evo_count = [min(pop_size * weight // sum(POP_EVO_WEIGHT), 1) for weight in POP_EVO_WEIGHT]
    pop_evo_count[-1] += pop_size - sum(pop_evo_count)

    # -------- CONFIGURE PLOT --------
    pop_hist = [(0.5,0.5,0.5,0.5)]
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.ion()
    plt.show()

    # Go through epochs
    for i in range(1, epochs+1):
        print(f'Starting epoch {i}')
        print('\tChoosing training data for epoch')
        epoch_boards = [[], []]
        epoch_boards[0] = boards[0][random.choice(
            len_boards0, size=HBOARDS_PER_EPOCH, replace=False)]
        epoch_boards[1] = boards[1][random.choice(
            len_boards1, size=HBOARDS_PER_EPOCH, replace=False)]

        print('\tGetting population\'s fitnesses: ', end='')
        for j, genome in enumerate(pop):
            print(j+1, end=' ', flush=True)
            genome.fit = 0
            for label in range(len(epoch_boards)):
                for fen in epoch_boards[label]:
                    genome.fit += calc_fitness(run(fen, genome.nn), label)

        print('\n\tSorting population based on fitness')
        pop.sort(key=lambda g: g.fit, reverse=True)

        pop_hist.append((pop[-1].fit / BOARDS_PER_EPOCH,
            pop[pop_size//2].fit / BOARDS_PER_EPOCH, 
            mean([genome.fit for genome in pop]) / BOARDS_PER_EPOCH,
            pop[0].fit / BOARDS_PER_EPOCH))
        print(f'\tMin: {pop_hist[-1][0]} | Median: {pop_hist[-1][1]} | Mean: {pop_hist[-1][2]} | Max: {pop_hist[-1][3]}')
        ax1.clear()
        ax1.plot(pop_hist)
        plt.draw()
        plt.pause(0.001)

        print('\tEvolving population')
        evo_start = pop_evo_count[0]
        evo_stop = pop_evo_count[0] + pop_evo_count[1]
        for j in range(evo_start, evo_stop):
            pop[j].mut()
        evo_start = evo_stop
        evo_stop += pop_evo_count[2]
        for j in range(evo_start, evo_stop):
            pop[j].rand()

def calc_fitness(result: ndarray, label: int) -> float:
    return 1 - abs(label - result)

class Genome:
    def __init__(self, layers: int, neurons: int, nn: list = None, fit = 0):
        self.lays = layers
        self.neurs = neurons
        if nn is None:
            self.rand()
        else:
            self.nn = nn
        self.fit = fit

    def rand(self):
        self.nn = [None]*3
        self.nn[0] = float32(random.uniform(low=MIN_FLOAT, high=MAX_FLOAT, size=(
            self.neurs * 2 * INPUT_SIZE))).reshape(self.neurs, 2, INPUT_SIZE)
        self.nn[1] = float32(random.uniform(low=MIN_FLOAT, high=MAX_FLOAT, size=(
            self.lays * self.neurs * 2 * self.neurs))).reshape(
                self.lays, self.neurs, 2, self.neurs)
        self.nn[2] = float32(random.uniform(low=MIN_FLOAT, high=MAX_FLOAT, size=(
            2 * self.neurs))).reshape(1, 2, self.neurs)

    def mut(self):
        layer_i = random.randint(2)
        part_i = random.randint(1)
        if layer_i == 0:
            neuron_i = random.randint(self.neurs)
            synapse_i = random.randint(INPUT_SIZE)
            self.nn[0][neuron_i][part_i][synapse_i] = float32(
                random.uniform(low=MIN_FLOAT, high=MAX_FLOAT))
        elif layer_i == 1:
            hlayer_i = random.randint(self.lays)
            neuron_i = random.randint(self.neurs)
            synapse_i = random.randint(self.neurs)
            self.nn[1][hlayer_i][neuron_i][part_i][synapse_i] = float32(
                random.uniform(low=MIN_FLOAT, high=MAX_FLOAT))
        else:
            synapse_i = random.randint(self.neurs)
            self.nn[2][0][part_i][synapse_i] = float32(
                random.uniform(low=MIN_FLOAT, high=MAX_FLOAT))
        
if __name__ == '__main__':
    main()
    input('Training complete, press enter to exit ')
    plt.show()
