# chess-nn-py
A simple chess genetic neural network engine written in python

## Notes
- Uses games in PGN notation for training
- If game ends in a tie, it is not used for training
- For each move, encode the board into FEN and sends it to main
- Engine works by testing each move
- Use the return value for training Float(0,1) where closer to 0 is higher chance of black winning, 1 is higher chance of white winning
- Neural network flattens FEN to [64][14] + FEN extras binary array
	- 64 = 8x8 board, 14 = piece types + is enpassant target + blank square, FEN extras (castle, next move, full move, half move)
- After each training generation (before mutation) save population and model, named based on their rank
- Mutate at the beginning of eatch epoch, before training


- Engine will use python-chess and board.legal_moves to test the boardstate after each move
	- get_best_move(fen, [weights])
	
## Inputs calculation

- Each square has 13* states, 1 for each piece and color totaling 12, + 1 blank square, and 1 enpassant square
- There are 64 squares on the board. In combination of square states, this results in 896‬ square states which can be 0 or 1

- Normal square states = 832 = 8 * 8 * 13
- Plus 1 for who's turn is next (0 or 1)
- Plus 4 for castling states (0 or 1)
- Plus 16 for en-passantable square states (8 * 2)
- Plus 1 for halfmove clock  (0 to 50)
- Plus 1 for fullmove counter (0 to inf)
- Total: 855 inputs

* ** CHANGE ONLY 3 and 6 rank require 14 states, everything else only needs 13 ** *
En-passant can only happen on rank 3 and 6


evolve():
	1/3 (best third) Choose random partner, breed and crossover
	1/3 (middle third) Random mutation
	1/3 (bottom third) Killed and replaced with new genomes



Kill chance, based on performance
	Survivors go though reproduce chance, based on performance
		Children (product of crossover) go through mutation chance, complete random

## TODO

- Implement crossover as a random start and end point where one genome is spliced and joined to another

## References
- https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6
- https://towardsdatascience.com/my-experiments-in-replacing-deep-learning-backpropagation-sgd-with-a-genetic-algorithm-c6e308382926
- https://towardsdatascience.com/gas-and-nns-6a41f1e8146d
- https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164
