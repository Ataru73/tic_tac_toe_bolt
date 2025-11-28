import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        terminated = False
        
        # 1. Selection
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            obs, reward, terminated, truncated, info = state.step(action)
            
            if terminated:
                # If the game ended, the value is from the perspective of the player who just moved.
                # If reward is 1 (win), then the previous player won.
                # The next player (current state) is the loser.
                # So leaf_value for the current node (next player) should be -1.
                leaf_value = -1.0 
                if reward == 0: # Draw
                     leaf_value = 0.0
                
                node.update_recursive(-leaf_value)
                return

        # 2. Expansion and Evaluation
        action_probs, leaf_value = self._policy(state)
        
        node.expand(action_probs)
        
        # 3. Backup
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

try:
    from cublino_contra import _mcts_cpp
except ImportError:
    _mcts_cpp = None

class MCTS_CPP:
    def __init__(self, model_path, c_puct=5, n_playout=2000, device="cpu"):
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
        cpp_state = _mcts_cpp.CublinoState()
        
        # Convert board (7, 7, 3) to numpy array then to C++
        # pybind11 handles numpy -> py::array_t automatically
        cpp_state.set_state_from_python(env.unwrapped.board, env.unwrapped.current_player)
        
        return cpp_state
