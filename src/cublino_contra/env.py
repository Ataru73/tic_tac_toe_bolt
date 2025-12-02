import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque

class CublinoContraEnv(gym.Env):
    metadata = {"render_modes": ["ascii", "ansi"]}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.board_size = 7
        self.history_length = 4  # Stack last 4 board states
        
        # Observation Space: 7x7 grid with 12 channels (4 stacked states * 3 channels each)
        # Each state has 3 channels:
        # Channel 0: Player (0: Empty, 1: P1, -1: P2)
        # Channel 1: Top Value (0: Empty, 1-6)
        # Channel 2: South Value (0: Empty, 1-6)
        self.observation_space = spaces.Box(
            low=-1, high=6, shape=(self.board_size, self.board_size, 3 * self.history_length), dtype=np.int8
        )

        # Action Space: Select a square (row, col) and a direction (0:N, 1:E, 2:S, 3:W)
        # Total actions: 7 * 7 * 4 = 196
        self.action_space = spaces.Discrete(self.board_size * self.board_size * 4)

        # Vector mapping for dice orientation logic
        self._val_to_vec = {
            1: np.array([0, 0, 1]),
            6: np.array([0, 0, -1]),
            2: np.array([0, -1, 0]), # South
            5: np.array([0, 1, 0]),  # North
            3: np.array([1, 0, 0]),  # East
            4: np.array([-1, 0, 0])  # West
        }
        self._vec_to_val = {tuple(v): k for k, v in self._val_to_vec.items()}
        
        self.max_steps = 300
        
        # Initialize state history buffer
        self.state_history = deque(maxlen=self.history_length)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size, 3), dtype=np.int8)
        self.current_player = 1 # P1 starts
        self.steps = 0

        # Setup P1 (Row 0)
        # Top=6, South=3
        for col in range(self.board_size):
            self.board[0, col] = [1, 6, 3]

        # Setup P2 (Row 6)
        # Top=6, South=4 (Since North is 3 relative to P2, so South is 7-3=4)
        for col in range(self.board_size):
            self.board[6, col] = [-1, 6, 4]
            
        # Initialize state history with copies of initial board
        self.state_history.clear()
        for _ in range(self.history_length):
            self.state_history.append(self.board.copy())
        
        # Draw history for 3-fold repetition
        self.history = {}
        # Add initial state
        state_key = self._get_state_key()
        self.history[state_key] = 1

        return self._get_obs(), {}

    def _get_state_key(self):
        # Include current player in the state key to distinguish turns
        return (bytes(self.board), self.current_player)

    def _get_obs(self):
        # Stack the last history_length board states along the channel dimension
        # Returns shape: (7, 7, 12) where 12 = 4 states * 3 channels
        return np.concatenate(list(self.state_history), axis=2)

    def step(self, action):
        # Decode action
        direction = action % 4
        square_idx = action // 4
        row = square_idx // self.board_size
        col = square_idx % self.board_size

        # Directions: 0:N, 1:E, 2:S, 3:W
        dr, dc = 0, 0
        if direction == 0: dr = 1  # North (Row + 1)
        elif direction == 1: dc = 1 # East (Col + 1)
        elif direction == 2: dr = -1 # South (Row - 1)
        elif direction == 3: dc = -1 # West (Col - 1)

        # Validate Move
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return self._get_obs(), -1, False, True, {"error": "Invalid source coordinates", "reason": "Invalid Move"}
        
        die = self.board[row, col]
        player, top, south = die

        if player != self.current_player:
             # Invalid: Not your die or empty
             return self._get_obs(), -1, False, True, {"error": "Not your die", "reason": "Invalid Move"}

        # Check Backward Move
        if self.current_player == 1 and direction == 2: # P1 cannot move South
             return self._get_obs(), -1, False, True, {"error": "Cannot move backwards", "reason": "Invalid Move"}
        if self.current_player == -1 and direction == 0: # P2 cannot move North
             return self._get_obs(), -1, False, True, {"error": "Cannot move backwards", "reason": "Invalid Move"}

        target_row, target_col = row + dr, col + dc

        if not (0 <= target_row < self.board_size and 0 <= target_col < self.board_size):
             # Invalid: Out of bounds
             return self._get_obs(), -1, False, True, {"error": "Out of bounds", "reason": "Invalid Move"}

        if self.board[target_row, target_col, 0] != 0:
             # Invalid: Target occupied
             return self._get_obs(), -1, False, True, {"error": "Target occupied", "reason": "Invalid Move"}

        # Perform Move
        new_top, new_south = self._rotate_die(top, south, direction)
        
        # Update Board
        self.board[row, col] = [0, 0, 0]
        self.board[target_row, target_col] = [self.current_player, new_top, new_south]
        
        # Check Win Condition (Reached opponent's back row)
        # P1 target: Row 6
        # P2 target: Row 0
        if self.current_player == 1 and target_row == 6:
            return self._get_obs(), 1, True, False, {"winner": 1}
        if self.current_player == -1 and target_row == 0:
            return self._get_obs(), 1, True, False, {"winner": -1}

        # Resolve Battles
        self._resolve_battles(target_row, target_col)
        
        # Update state history
        self.state_history.append(self.board.copy())

        # Switch Turn
        self.current_player *= -1
        
        self.steps += 1
        if self.steps >= self.max_steps:
             return self._get_obs(), 0, False, True, {"winner": 0, "reason": "Max steps reached"}
        
        # Check 3-fold repetition
        state_key = self._get_state_key()
        self.history[state_key] = self.history.get(state_key, 0) + 1
        
        if self.history[state_key] >= 3:
            return self._get_obs(), 0, True, False, {"winner": 0, "reason": "3-fold repetition"}

        return self._get_obs(), 0, False, False, {}

    def _rotate_die(self, top, south, direction):
        # 0:N, 1:E, 2:S, 3:W
        # Map to vectors
        v_top = self._val_to_vec[top]
        v_south = self._val_to_vec[south]
        v_east = np.cross(v_top, v_south)
        
        # Values
        val_top = top
        val_south = south
        val_north = 7 - south
        val_bottom = 7 - top
        val_east = self._vec_to_val[tuple(v_east)]
        val_west = 7 - val_east

        if direction == 0: # North
            # New Top = Old South
            # New South = Old Bottom
            return val_south, val_bottom
        elif direction == 2: # South
            # New Top = Old North
            # New South = Old Top
            return val_north, val_top
        elif direction == 1: # East
            # New Top = Old West
            # New South = Old South (unchanged)
            return val_west, val_south
        elif direction == 3: # West
            # New Top = Old East
            # New South = Old South (unchanged)
            return val_east, val_south
        return top, south

    def _resolve_battles(self, moved_r, moved_c):
        # Check neighbors of the moved die for opponent dice
        # If an opponent die is found, check if it is now "contested" (surrounded by >= 2 current_player dice)
        
        opponent = -self.current_player
        neighbors = self._get_neighbors(moved_r, moved_c)
        
        # We need to handle simultaneous battles.
        # Identify all contested opponent dice first.
        contested_dice = []
        
        for r, c in neighbors:
            if self.board[r, c, 0] == opponent:
                # Check if this opponent die is surrounded by >= 2 of current_player
                opp_neighbors = self._get_neighbors(r, c)
                friendly_count = sum(1 for nr, nc in opp_neighbors if self.board[nr, nc, 0] == self.current_player)
                
                if friendly_count >= 2:
                    contested_dice.append((r, c, opp_neighbors))
        
        # Resolve each battle
        # Note: Resolutions are simultaneous based on board state BEFORE any removals.
        dice_to_remove = []
        for r, c, defender_neighbors in contested_dice:
            defender_die_val = self.board[r, c, 1]
            
            # FIX: Calculate sums based on current board state, do not modify yet
            defender_friendly_sum = sum(self.board[nr, nc, 1] 
                                      for nr, nc in defender_neighbors 
                                      if self.board[nr, nc, 0] == opponent)
            
            attacker_sum = sum(self.board[nr, nc, 1] 
                             for nr, nc in defender_neighbors 
                             if self.board[nr, nc, 0] == self.current_player)
            
            defender_total = defender_die_val + defender_friendly_sum
            
            if defender_total < attacker_sum:
                dice_to_remove.append((r, c))
                
        # FIX: Apply removals after all calculations are done
        for r, c in dice_to_remove:
            self.board[r, c] = [0, 0, 0]

    def _get_neighbors(self, r, c):
        # Orthogonal neighbors
        nbs = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                nbs.append((nr, nc))
        return nbs

    def get_legal_actions(self):
        """
        Returns a list of legal action indices.
        Action = (row * 7 + col) * 4 + direction
        """
        legal_actions = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c, 0] == self.current_player:
                    # Check all 4 directions
                    # 0:N, 1:E, 2:S, 3:W
                    for direction in range(4):
                        # P1 cannot move South (2), P2 cannot move North (0)
                        if self.current_player == 1 and direction == 2: continue
                        if self.current_player == -1 and direction == 0: continue

                        dr, dc = 0, 0
                        if direction == 0: dr = 1
                        elif direction == 1: dc = 1
                        elif direction == 2: dr = -1
                        elif direction == 3: dc = -1
                        
                        target_r, target_c = r + dr, c + dc
                        
                        # Check bounds
                        if 0 <= target_r < self.board_size and 0 <= target_c < self.board_size:
                            # Check occupancy
                            if self.board[target_r, target_c, 0] == 0:
                                action_idx = (r * self.board_size + c) * 4 + direction
                                legal_actions.append(action_idx)
        return legal_actions

    def render(self):
        if self.render_mode == "ascii":
            print(self._render_ascii())
    
    def _render_ascii(self):
        # Render board
        # P1: Positive numbers, P2: Negative numbers? Or just Colors.
        # Let's use format: P V (Player, Value)
        lines = []
        lines.append("  " + " ".join([f" {c}  " for c in range(self.board_size)]))
        for r in range(self.board_size - 1, -1, -1):
            row_str = f"{r} "
            for c in range(self.board_size):
                p, v, s = self.board[r, c]
                if p == 0:
                    row_str += " ..  "
                elif p == 1:
                    row_str += f" +{v}  "
                else:
                    row_str += f" -{v}  "
            lines.append(row_str)
        return "\n".join(lines)

