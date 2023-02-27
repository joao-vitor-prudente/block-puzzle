# block-puzzle

A terminal based version of the block puzzle game highly optimized for training reinforcement learning algorithms.

Made sure to vectorize the every single bit of the codebase to make it as fast as possible and to make it easy to paralelize in the gpu (just change the numpy arrays to cupy arrays) to make the model training as computationally cheep as possible.

The game consists of a 9x9 grid and 3 random shapes that appear every round. The player has to place on the grid all the shapes before three new appear. When a row, column or a 3x3 square on the corner or exactly on the middle is completely full, it is cleared. The game ends when the player can't place any of the shapes on the grid.