import numpy as np


dot = np.ones((1, 1), dtype=np.int_)

small_i_1 = np.ones((1, 2), dtype=np.int_)
small_i_2 = np.rot90(small_i_1)

medium_i_1 = np.ones((1, 3), dtype=np.int_)
medium_i_2 = np.rot90(medium_i_1)

big_i_1 = np.ones((1, 4), dtype=np.int_)
big_i_2 = np.rot90(big_i_1)

huge_i_1 = np.ones((1, 5), dtype=np.int_)
huge_i_2 = np.rot90(huge_i_1)

big_l_1 = np.array([
    [1, 0, 0],
    [1, 0, 0],
    [1, 1, 1],
], dtype=np.int_)
big_l_2 = np.rot90(big_l_1)
big_l_3 = np.rot90(big_l_2)
big_l_4 = np.rot90(big_l_3)

big_l_rev_1 = np.flip(big_l_1, axis=1)
big_l_rev_2 = np.rot90(big_l_rev_1)
big_l_rev_3 = np.rot90(big_l_rev_2)
big_l_rev_4 = np.rot90(big_l_rev_3)

medium_l_1 = np.array([
    [1, 0],
    [1, 0],
    [1, 1],
], dtype=np.int_)
medium_l_2 = np.rot90(medium_l_1)
medium_l_3 = np.rot90(medium_l_2)
medium_l_4 = np.rot90(medium_l_3)

medium_l_rev_1 = np.flip(medium_l_1, axis=1)
medium_l_rev_2 = np.rot90(medium_l_rev_1)
medium_l_rev_3 = np.rot90(medium_l_rev_2)
medium_l_rev_4 = np.rot90(medium_l_rev_3)

small_l_1 = np.array([
    [1, 0],
    [1, 1],
], dtype=np.int_)
small_l_2 = np.rot90(small_l_1)
small_l_3 = np.rot90(small_l_2)
small_l_4 = np.rot90(small_l_3)

small_l_rev_1 = np.flip(small_l_1, axis=1)
small_l_rev_2 = np.rot90(small_l_rev_1)
small_l_rev_3 = np.rot90(small_l_rev_2)
small_l_rev_4 = np.rot90(small_l_rev_3)

plus = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0],   
], dtype=np.int_)

small_t_1 = np.array([
    [1, 1, 1],
    [0, 1, 0],
], dtype=np.int_)
small_t_2 = np.rot90(small_t_1)
small_t_3 = np.rot90(small_t_2)
small_t_4 = np.rot90(small_t_3)

big_t_1 = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
], dtype=np.int_)
big_t_2 = np.rot90(big_t_1)
big_t_3 = np.rot90(big_t_2)
big_t_4 = np.rot90(big_t_3)

small_diagonal_1 = np.array([
    [1, 0],
    [0, 1],
], dtype=np.int_)
small_diagonal_2 = np.rot90(small_diagonal_1)

big_diagonal_1 = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
], dtype=np.int_)
big_diagonal_2 = np.rot90(big_diagonal_1)

u_1 = np.array([
    [1, 0, 1],
    [1, 1, 1],
], dtype=np.int_)
u_2 = np.rot90(u_1)
u_3 = np.rot90(u_2)
u_4 = np.rot90(u_3)

s_1 = np.array([
    [0, 1, 1],
    [1, 1, 0],
], dtype=np.int_)
s_2 = np.rot90(s_1)

s_rev_1 = np.flip(s_1, axis=1)
s_rev_2 = np.rot90(s_rev_1)
