import numpy as np
import numpy.typing as npt
from grid_settings import GRID_SIZE, GRID_DIV_SIZE, PIECES
import functools
from dataclasses import dataclass

int_arr = npt.NDArray[np.int_]
bool_arr = npt.NDArray[np.bool_]
obj_arr = npt.NDArray[np.object_]


def get_random_pieces(seed: float | None = None) -> list[int_arr]:
    
    np.random.seed = seed
    random_integers = np.random.randint(0, len(PIECES) - 1, size=3)  # type: ignore
    return [PIECES[i] for i in random_integers]


def get_available_spaces(grid: int_arr, piece: int_arr) -> bool_arr:
    
    def check_starting_point(x: int, y: int, _new_grid: int_arr) -> np.bool_:
        
        # create new empty grid and put the piece in it at a starting position
        _new_grid = np.zeros_like(grid)
        _new_grid[x:x+piece.shape[0], y:y+piece.shape[1]] = piece
        
        # add the new grid to the current grid and if there is any overlap it means we summed up 1 + 1 = 2
        _new_grid = np.add(grid, _new_grid)
        
        # check for any 2 in the summed up grid
        return np.logical_not(np.any(_new_grid == 2))
    
    new_grid = np.zeros_like(grid)
    
    # vectorize the function
    check_starting_point = functools.partial(check_starting_point, _new_grid=new_grid)
    check_starting_point = np.vectorize(check_starting_point, otypes=[np.bool_])
    
    # define the limits to check the starting point
    limit_x = grid.shape[0] - piece.shape[0] + 1
    limit_y = grid.shape[1] - piece.shape[1] + 1
    indices = np.indices((limit_x, limit_y))
    
    # check if point is available for every point in the grid
    new_grid[0:limit_x, 0:limit_y] = check_starting_point(indices[0], indices[1])
    return new_grid.astype(np.bool_)


@dataclass
class Piece:
    id_: np.int_
    shape: int_arr
    is_available: np.bool_
    available_spaces: bool_arr


def store_pieces(grid: int_arr, pieces: list[int_arr]) -> obj_arr:
    
    # create a list of index, piece and array of available spaces
    pieces_ = [
        (i, piece, get_available_spaces(grid, piece))
        for i, piece in enumerate(pieces)
    ]
    pieces_ = np.array([
        Piece(np.int_(i), shape, np.any(spaces), spaces)
        for i, shape, spaces in pieces_
    ], dtype=np.object_)
    return pieces_


def fetch_movement():
    
    return input("Enter piece index, x and y coordinates separated by spaces: ")


def validate_response(movement: str) -> bool:
    
    # check if the number of information given is correct
    if not movement.count(" ") == 2:
        print("invalid formatting")
        return False
    
    # check if the information given are numbers
    if not all(map(str.isdigit, movement.split())):
        print("invalid formating") 
        return False
    return True


def parse_response(move: str) -> tuple[np.int_, ...]:
    
    return tuple(map(np.int_, move.split()))


def validate_movement(movement: tuple[np.int_, ...], pieces: obj_arr) -> bool:
    
    piece_index, x, y = movement

    # check if the piece index is valid and available  

    if not 0 <= piece_index < len(pieces):
        print("invalid piece index")
        return False
    
    # check if coordinates are not off grid limits    
    if not 0 <= x < GRID_SIZE and not 0 <= y < GRID_SIZE:
        print("coordinates are off grid limits")
        return False
    
    # check if there is no spaces available for the piece
    if not pieces[piece_index].is_available:
        print("piece has nowhere to go in the grid")
        return False
    
    # check if the coordinates are valid
    if not pieces[piece_index].available_spaces[x, y]:
        print("coordinates are occupied")
        return False
    
    return True


def get_movement(pieces: obj_arr) -> tuple[np.int_, ...]:
    while True:
        res = fetch_movement()
        if not validate_response(res): continue
        movement = parse_response(res)
        if not validate_movement(movement, pieces): continue
        break
    return movement    


def pop_used_pieces(pieces: obj_arr, movement: tuple[np.int_, ...]) -> obj_arr:
    pieces_before = pieces[:movement[0]]
    pieces_after = pieces[movement[0]+1:]
    return np.concatenate((pieces_before, pieces_after)) 


def execute_movement(grid: int_arr, movement: tuple[np.int_, ...], pieces: obj_arr) -> int_arr:
    
    piece_index, x, y = movement
    piece = pieces[piece_index].shape
    grid[x:x+piece.shape[0], y:y+piece.shape[1]] += piece
    return grid


def highlight_squares_to_pop(grid: int_arr) -> int_arr:
    blank = np.full_like(grid, -1)

    grid = np.where(np.all(grid == 1, axis=0), blank, grid)
    grid = np.where(np.all(np.logical_or(grid == 1, grid == -1), axis=1), blank, grid.T).T

    grid = grid.reshape(GRID_DIV_SIZE, GRID_DIV_SIZE, GRID_DIV_SIZE, GRID_DIV_SIZE).swapaxes(1, 2)
    grid[np.all(np.logical_or(grid == 1, grid == -1), axis=(2, 3))] =  np.full((GRID_DIV_SIZE, GRID_DIV_SIZE), -1)
    grid = grid.swapaxes(1, 2).reshape(GRID_SIZE, GRID_SIZE)
    return grid


def calculate_points(grid: int_arr) -> np.int_:
    return 2 * (grid == -1).sum()
    

def pop_spaces(grid: int_arr) -> int_arr:
    grid[grid == -1] = 0
    return grid


def format_ui(grid: int_arr, pieces: obj_arr, points: np.int_) -> str:
    formatted_points = f"Points: {points}"
    
    format_grid = np.vectorize(lambda x: u"\u25A1" if x == 0 else u"\u25A0")
    
    g = format_grid(grid)

    formatted_grid = " ".join([
        "" + cell if i == 0 else
        "\n\n" + cell if i % 27 == 0 else 
        "\n" + cell if i % 9 == 0 else 
        "  " + cell if i % 3 == 0 else cell 
        for i, cell in enumerate(g.flatten())
    ])
    
    # formatted grid without separations
    
    # formatted_grid = " ".join([
    #     "" + cell if i == 0 else
    #     "\n" + cell if i % 9 == 0 else cell
    #     for i, cell in enumerate(g.flatten())
    # ])

    formatted_pieces = "\n\n".join([
        "\n".join([" ".join(piece[i, :]) for i in range(piece.shape[0])])
        for piece in [format_grid(piece.shape) for piece in pieces]
    ])
    
    return f"{formatted_points}\n\n{formatted_grid}\n\n{formatted_pieces}\n"
    

def game_loop():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int_)
    points = np.int_(0)
    while True:
        pieces_ = get_random_pieces()
        pieces = store_pieces(grid, pieces_)
        while True:
            if pieces.shape[0] == 0: break
            if not np.any([piece.is_available for piece in pieces]): raise Exception("Game Over")
            ui = format_ui(grid, pieces, points)
            print(ui)
            movement = get_movement(pieces)          
            grid = execute_movement(grid, movement, pieces)
            grid = highlight_squares_to_pop(grid)
            points += calculate_points(grid)
            grid = pop_spaces(grid)
            pieces = pop_used_pieces(pieces, movement) 
            pieces = store_pieces(grid, [piece.shape for piece in pieces])





if __name__ == "__main__":
    game_loop()