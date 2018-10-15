
from sample_players import DataPlayer

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only *required* method. You can modify
    the interface for get_action by adding named parameters with default
    values, but the function MUST remain compatible with the default
    interface.

    **********************************************************************
    NOTES:
    - You should **ONLY** call methods defined on your agent class during
      search; do **NOT** add or call functions outside the player class.
      The isolation library wraps each method of this class to interrupt
      search when the time limit expires, but the wrapper only affects
      methods defined on this class.

    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller is responsible for
        cutting off the function after the search time limit has expired. 

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #import random
        #self.queue.put(random.choice(state.actions()))
        
        #---------------------------
        # Hyper-parameters
        # (Start)
        #---------------------------
        
        # This is a hyper-parameter that needs to be tuned.
        depth_limit = 100
        
        # Score mode
        # self.score_mode = 0: score baseline heuristic
        # self.score_mode = 1: score custom heuristic A
        self.score_mode = 1
        
        # This is for score_custom_heuristic.
        # It is the penalty multiplier for having the opponent having moves.
        self.penalty_coefficient = 2
        
        #---------------------------
        # Hyper-parameters
        # (End)
        #---------------------------
        
        # Turns out "iterative deepening" is just a for loop...
        # From lecture: 
        #   Lesson 2 - Optimizing Minimax Search, 17. Coding: Iterative Deepening
        for depth in range(1, depth_limit+1):
            #best_move = minimax_decision(gameState, depth)
            #best_move = o_MinimaxPlayer.minimax(state, depth)
            
            
            # Using Alpha-Beta Prunning from lecture:
            #   Lesson 2 - Optimizing Minimax Search, 25. Coding: Alpha-Beta Pruning
            #
            # NOTE: In that lecture, it said 
            #       "We have removed the depth-limiting search and are not using iterative deepening in 
            #       "this quiz. You can trivially modify alpha-beta search to use both, just as you did 
            #        before with minimax.
            #
            # Adding back depth-limiting search and iterative deepening.
            action = self.alpha_beta_search(state, depth)
            
            # Queue up the action if found.
            if action is not None:
            
                #return best_move
                #
                # NOTE: Notice there isn't a return. The reason being
                #       the method keeps providing the best action 
                #       available as it is going through the tree.
                #       If it runs out of time, go with the best found
                #       so far.
                self.queue.put(action)
        
    def alpha_beta_search(self, state, depth):
        """ Return the move along a branch of the game tree that
        has the best possible value.  A move is a pair of coordinates
        in (column, row) order corresponding to a legal move for
        the searching player.

        You can ignore the special case of calling this function
        from a terminal state.
        """
        #alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None
        for a in state.actions():
            #v = self.min_value(state.result(a), alpha, beta)
            #
            # Using best score instead of alpha.
            v = self.min_value(state.result(a), best_score, beta, depth - 1)
            #alpha = self.max(alpha, v)
            if v > best_score:
                best_score = v
                best_move = a
        return best_move
    
    def min_value(self, state, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
#         if state.terminal_test():
#             return state.utility(0)
        
        if depth <= 0:
            #return my_moves(gameState)
            return self.score(state, mode = self.score_mode)

        if state.terminal_test():
            
            # NOTE: is defined in class BasePlayer
            #       in sample_players.py file
            return state.utility(self.player_id)
        
        v = float("inf")
        for a in state.actions():

            # NOTE: Notice Max_value for min_value method.
            v = min(v, self.max_value(state.result(a), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v
    
    def max_value(self, state, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
#         if state.terminal_test():
#             return state.utility(0)
        
        if depth <= 0:
            #return my_moves(gameState)
            return self.score(state, mode = self.score_mode)

        if state.terminal_test():
            
            # NOTE: is defined in class BasePlayer
            #       in sample_players.py file
            return state.utility(self.player_id)
        
        # NOTE: Notice the negative sign in front of infiniti.
        v = float("-inf")
        for a in state.actions():

            # NOTE: Notice Min_value for max_value method.
            v = max(v, self.min_value(state.result(a), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
        
    def score(self, state, mode = 0):
        """
        Description: Heuristic score.
                     mode = 0: score baseline
                     mode = 1: custom heuristic a
        """
        
        if mode == 1:
            return self.score_custom_heuristic(state)
        
        # Return the score baseline heuristic in any other case
        return self.score_baseline(state)
        
        
    def score_baseline(self, state):
        """
        Description: Using Advanced Heuristic for score
                     Baseline: #my_moves - #opponent_moves heuristic from lecture 
                               (should use fair_matches flag in run_match.py)
        """
        
        # Look in file samples_players.py, look at GreedyPlayer class method score,
        # it is a similar implementation
        own_loc = state.locs[self.player_id]
        own_liberties = state.liberties(own_loc)
        
        # Get the opposite player ID.
        foe_loc = state.locs[ 1 - self.player_id]
        foe_liberties = state.liberties(foe_loc)
        
        #return len(own_liberties)
        
        # Implement the heuristic and return it.
        #
        # Return
        return len(own_liberties) - len(foe_liberties)
    
    
    def score_custom_heuristic(self, state):
        """
        Description: Custom heuristic.
                     delta_liberty_difference / distance_between_current_and_opponent
                     
                     where:
                     delta_liberty_difference = (current_moves_remaining - penalty_multiplier*opponent_moves_remaining)
        """
        
#         if state.terminal_test():
            
#             # NOTE: is defined in class BasePlayer
#             #       in sample_players.py file
#             return state.utility(self.player_id)
        
        # Calculate delta of moves
        # Look in file samples_players.py, look at GreedyPlayer class method score,
        # it is a similar implementation
        own_loc = state.locs[self.player_id]
        own_liberties = state.liberties(own_loc)
        
        # Get the opposite player ID.
        #
        # NOTE: penalty_coefficient is a hyper-parameter that we set.
        foe_loc = state.locs[ 1 - self.player_id]
        foe_liberties = state.liberties(foe_loc)
        delta_liberty_difference = len(own_liberties) - self.penalty_coefficient*len(foe_liberties)
        
        # Using Manhattan distance between the current player and opponent.
        #
        # NOTE: The location is represented in Bitboard Encoding
        #       https://github.com/udacity/artificial-intelligence/blob/master/Projects/3_Adversarial%20Search/isolation/README.md
        #       Have to decode it back.
        #       column = x%13
        #       row = math.floor(x/13)
        import math
        current_x = math.floor(own_loc/13)
        current_y = own_loc%13
        #
        # To handle NoneType, make foe_loc the same as own_loc.
        # Might not be the best solution in the world.
        if foe_loc is None:
            foe_loc = own_loc
        
        foe_x =  math.floor(foe_loc/13)
        foe_y =  foe_loc%13
        delta_manhattan_distance = abs(current_y - foe_y) + abs(current_x - foe_x) +0.001
        
    
#         # DEBUG
#         #
#         print("DEBUG - own_loc: {}".format(own_loc))
#         print("DEBUG - foe_loc: {}".format(foe_loc))
        
        # Final result
        result = float( delta_liberty_difference / delta_manhattan_distance )
        #result = delta_liberty_difference
        
        # Return the result
        return result
        
        
        
