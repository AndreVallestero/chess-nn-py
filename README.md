# chess-nn-py
A simple chess genetic neural network engine written in python

## Notes
- Uses games in PGN notation for training
- If game ends in a tie, it is not used for training
- For each move, encode the board into XFEN and sends it to main
- Engine works by testing each move
- Use the return value for training Float(0,1) where closer to 0 is higher chance of black winning, 1 is higher chance of white winning
- Neural network flattens XFEN to [64][14] + XFEN extras binary array
	- 64 = 8x8 board, 14 = piece types + is enpassant target + blank square, XFEN extras (castle, next move, full move, half move)
	