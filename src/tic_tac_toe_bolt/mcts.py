import numpy as np
import torch
import copy
try:
    from tic_tac_toe_bolt import _mcts_cpp
except ImportError:
    _mcts_cpp = None

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        terminated = False
        winner = 0
        
        # 1. Selection
        while not node.is_leaf():
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            obs, reward, terminated, truncated, info = state.step(action)
            if terminated:
                leaf_value = -1.0 
                node.update_recursive(-leaf_value)
                return

        # 2. Expansion and Evaluation
        action_probs, leaf_value = self._policy(state)
        
        node.expand(action_probs)
        
        # 3. Backup
        node.update_recursive(-leaf_value) 

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # Loop for n_playout times
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTS_CPP:
    def __init__(self, model_path, c_puct=5, n_playout=10000, device="cpu"):
        if _mcts_cpp is None:
            raise ImportError("C++ extension not found. Please build it first.")
        self._mcts = _mcts_cpp.MCTS(model_path, c_puct, n_playout, str(device))
    
    def get_move_probs(self, env, temp=1e-3):
        # Convert env state to C++ state
        cpp_state = self._convert_env_to_cpp_state(env)
        acts, probs = self._mcts.get_move_probs(cpp_state, temp)
        return acts, probs

    def update_with_move(self, last_move):
        self._mcts.update_with_move(last_move)

    def _convert_env_to_cpp_state(self, env):
        # Assuming env is TicTacToeBoltEnv
        cpp_state = _mcts_cpp.TicTacToeState()
        
        unwrapped = env.unwrapped
        
        # Convert board to list of lists
        board_list = unwrapped.board.tolist()
        
        # Convert player_moves to required format
        # Python dict: {1: [(r, c), ...], -1: [...]}
        # C++ map: int -> vector<pair<int, int>>
        # Pybind11 handles dict -> map and list of tuples -> vector of pairs automatically if types match
        player_moves = unwrapped.player_moves
        
        cpp_state.set_state(board_list, unwrapped.current_player, player_moves)
        
        return cpp_state
