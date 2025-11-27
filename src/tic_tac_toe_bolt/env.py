import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy

class TicTacToeBoltEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_moves=100):
        self.window_size = 512  # The size of the PyGame window
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self.render_mode = render_mode
        self.max_moves = max_moves
        self.window = None
        self.clock = None
        
        # Game state
        self.board = None
        self.current_player = 1 # 1 or -1
        self.player_moves = {1: [], -1: []} # Track moves for "Infinite" mechanic
        self.move_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1
        self.player_moves = {1: [], -1: []}
        self.move_count = 0
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map action (0-8) to (row, col)
        row = action // 3
        col = action % 3

        # Check if move is valid
        if self.board[row, col] != 0:
            return self._get_obs(), -10, False, False, self._get_info()

        # "Infinite" mechanic: Check if player has 3 marks
        if len(self.player_moves[self.current_player]) == 3:
            # Remove oldest mark
            old_r, old_c = self.player_moves[self.current_player].pop(0)
            self.board[old_r, old_c] = 0

        # Place new mark
        self.board[row, col] = self.current_player
        self.player_moves[self.current_player].append((row, col))
        self.move_count += 1

        # Check for win
        if self._check_win(self.current_player):
            reward = 1
            terminated = True
            winner = self.current_player
        else:
            reward = 0
            terminated = False
            winner = 0
        
        # Check for truncation (max moves)
        truncated = False
        if not terminated and self.move_count >= self.max_moves:
            truncated = True

        # Switch player
        self.current_player *= -1

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        return self.board.copy()

    def _get_info(self):
        return {"current_player": self.current_player}

    def _check_win(self, player):
        # Check rows, cols, diagonals
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        pix_square_size = (self.window_size / 3)

        # Draw gridlines
        for x in range(self.window_size + 1):
            if x % int(pix_square_size) == 0:
                pygame.draw.line(
                    canvas,
                    0,
                    (0, x),
                    (self.window_size, x),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (x, 0),
                    (x, self.window_size),
                    width=3,
                )

        # Draw marks
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 1: # X
                    color = (255, 0, 0)
                    # Check if this mark is disappearing (oldest mark of current player, if they have 3)
                    if self.current_player == 1 and len(self.player_moves[1]) == 3 and self.player_moves[1][0] == (r, c):
                        color = (150, 0, 0) # Darker red
                        
                    pygame.draw.line(
                        canvas,
                        color,
                        (c * pix_square_size + 20, r * pix_square_size + 20),
                        ((c + 1) * pix_square_size - 20, (r + 1) * pix_square_size - 20),
                        width=8,
                    )
                    pygame.draw.line(
                        canvas,
                        color,
                        ((c + 1) * pix_square_size - 20, r * pix_square_size + 20),
                        (c * pix_square_size + 20, (r + 1) * pix_square_size - 20),
                        width=8,
                    )
                elif self.board[r, c] == -1: # O
                    color = (0, 0, 255)
                    # Check if this mark is disappearing (oldest mark of current player, if they have 3)
                    if self.current_player == -1 and len(self.player_moves[-1]) == 3 and self.player_moves[-1][0] == (r, c):
                        color = (0, 0, 150) # Darker blue
                        
                    pygame.draw.circle(
                        canvas,
                        color,
                        (int((c + 0.5) * pix_square_size), int((r + 0.5) * pix_square_size)),
                        int(pix_square_size / 3),
                        width=8,
                    )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def __deepcopy__(self, memo):
        """Custom deepcopy to handle pygame objects that cannot be pickled."""
        # Create a new instance without calling __init__ to avoid pygame initialization
        cls = self.__class__
        new_env = cls.__new__(cls)
        memo[id(self)] = new_env
        
        # Copy all attributes except pygame objects
        new_env.window_size = self.window_size
        new_env.observation_space = self.observation_space
        new_env.action_space = self.action_space
        new_env.render_mode = None  # Don't copy render mode to avoid pygame issues
        new_env.max_moves = self.max_moves
        new_env.window = None
        new_env.clock = None
        
        # Deep copy game state
        new_env.board = self.board.copy() if self.board is not None else None
        new_env.current_player = self.current_player
        new_env.player_moves = copy.deepcopy(self.player_moves, memo)
        new_env.move_count = self.move_count
        
        return new_env

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
