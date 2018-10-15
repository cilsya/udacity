
from utils import *


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
diagonal_units = []
unitlist = row_units + column_units + square_units

# TODO: Update the unit list to add the new diagonal units
rows_reversed = list(reversed(rows))
cols_reversed = list(reversed(cols))
#
zipped_A1_to_I9 = list(zip(rows, cols))
zipped_A9_to_I1 = list(zip(rows, cols_reversed))
#
diagonal_A1_to_I9 = [a[0]+a[1] for a in zipped_A1_to_I9]
diagonal_A9_to_I1 = [a[0]+a[1] for a in zipped_A9_to_I1]

# Append to list
diagonal_units.append(diagonal_A1_to_I9)
diagonal_units.append(diagonal_A9_to_I1)

# Update the unitlist
unitlist = unitlist + diagonal_units


# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).
    """
    # Get a list of dictionary keys that have only 2 values
    two_value_dict_keys_list = [box for box in values.keys() if len(values[box]) == 2]
    
    # Now determine which units have exactly 2 possible values
    for two_value_key_a in two_value_dict_keys_list:
      
      current_possible_values_a = values[two_value_key_a]
      
      # Get the units for this box
      units_for_box = units[two_value_key_a]
      
      # Cycle through each unit. If the value appears twice
      # in the boxes of the unit, remove those values in the 
      # other boxes of the unit
      for unit in units_for_box:
          
          box_of_units_keys = []
          for box_for_unit in unit:
              box_for_unit_possible_values_a = values[box_for_unit]
              
              if box_for_unit_possible_values_a == current_possible_values_a:
                  box_of_units_keys.append(box_for_unit)
              
              # If we found two boxes in the unit with the same value.
              # break out of the loop and return the box_of_units_keys.
              #if len(box_of_units_keys) == 2:
              #  break
                
          # Make all the entries unique
          box_of_units_keys = list(set(box_of_units_keys))
          
          # If we found at least two keys in the unit with the same
          # value, remove those values from the other box
          if len(box_of_units_keys) > 1:
              
              # cycle through boxes for the unit again
              for box_for_unit in unit:
                  
                  # Skip keys we know we want to keep
                  if box_for_unit in box_of_units_keys:
                    continue
                  
                  # Cycle through the digits and remove them
                  for digit in current_possible_values_a:
                    
                      # Update the values for that box in the unit
                      values[box_for_unit] = values[box_for_unit].replace(digit, '')

    # Return the updated dictionary
    return values
  
def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    # TODO: Copy your code from the classroom to complete this function
    #raise NotImplementedError
    
    # Get a list of dictionary keys that have only 1 value
    one_value_dict_keys_list = [box for box in values.keys() if len(values[box]) == 1]
    
    # Now cycle through the solved keys and remove that value from its peers.
    for box in one_value_dict_keys_list:
        solved_digit_as_string = values[box]
        
        # Cycle through the peers and replace the solved digit (as type string)
        # with a blank string
        for peer in peers[box]:
            
            # Update the values for that peer
            values[peer] = values[peer].replace(solved_digit_as_string, '')
            
    # Return the updated dictionary
    return values


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    # TODO: Copy your code from the classroom to complete this function
    #raise NotImplementedError
    
    digits = cols # 123456789
    for unit in unitlist:
        for digit in digits:
            
            # Only look at all the boxes (the grid address) in the unit.
            # If the box in value list key we are currently looking at
            # has the digit we are looking for as a possible solution
            # for the box, put it in the list.
            possible_boxes_for_digit_list = [box for box in unit if digit in values[box]]
            
            # If the the box list is of size 1, that means that is the only solution
            if len(possible_boxes_for_digit_list) == 1:
                
                # The element is 0 because there is only 1 element
                box_solution = possible_boxes_for_digit_list[0]
                
                values[box_solution] = digit
    
    # Return the updated dictionary
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    # TODO: Copy your code from the classroom and modify it to complete this function
    #raise NotImplementedError
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        value = eliminate(values)
        
        # Your code here: Use the Only Choice Strategy
        values = only_choice(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values



def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    # TODO: Copy your code from the classroom to complete this function
    #raise NotImplementedError
    
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes):
        return values ## Solved!
    # Choose one of the unfilled squares with the fewest possibilities
    n,s = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)
    # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
    for value in values[s]:
        new_sudoku = values.copy()
        new_sudoku[s] = value
        
        # This is recursion, calling itself.
        attempt = search(new_sudoku)
        if attempt:
            return attempt


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
