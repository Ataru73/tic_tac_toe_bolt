import gymnasium as gym
import os
import torch
import numpy as np
import pygame
import argparse
import sys
import time

try:
    from src.cublino_contra.model import PolicyValueNet
    from src.cublino_contra.mcts import MCTS, MCTS_CPP
    import src.cublino_contra # Register env
except ImportError:
    # If running as script from src/cublino_contra/
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.cublino_contra.model import PolicyValueNet
    from src.cublino_contra.mcts import MCTS, MCTS_CPP
    import src.cublino_contra # Register env

class HumanPlayer:
    def __init__(self):
        self.player = None
        self.selected_square = None # (row, col)
        self.legal_moves = [] # List of (target_row, target_col) for selected square
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        # Wait for mouse click
        # We need to handle selection logic here.
        # 1. Click on own piece -> Select it, show legal moves.
        # 2. Click on legal target -> Execute move.
        # 3. Click elsewhere -> Deselect.
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Map to grid
                    # Window size is 700 (100 per cell)
                    cell_size = 100
                    col = int(x // cell_size)
                    row = int(y // cell_size)
                    # PyGame coords: (0,0) is top-left.
                    # Board coords: (0,0) is bottom-left (Row 0).
                    # So Row 0 is at y=600. Row 6 is at y=0.
                    # row_idx = 6 - row
                    board_row = 6 - row
                    board_col = col
                    
                    if not (0 <= board_row < 7 and 0 <= board_col < 7):
                        continue
                        
                    # Check if clicked on own piece
                    if env.unwrapped.board[board_row, board_col, 0] == self.player:
                        self.selected_square = (board_row, board_col)
                        # Calculate legal moves for this piece
                        self.legal_moves = []
                        # Check 4 directions
                        for d in range(4):
                            dr, dc = 0, 0
                            if d == 0: dr = 1
                            elif d == 1: dc = 1
                            elif d == 2: dr = -1
                            elif d == 3: dc = -1
                            
                            tr, tc = board_row + dr, board_col + dc
                            if 0 <= tr < 7 and 0 <= tc < 7 and env.unwrapped.board[tr, tc, 0] == 0:
                                self.legal_moves.append((tr, tc, d))
                                
                    elif self.selected_square:
                        # Check if clicked on a legal target
                        for tr, tc, d in self.legal_moves:
                            if tr == board_row and tc == board_col:
                                # Execute move
                                sr, sc = self.selected_square
                                action = (sr * 7 + sc) * 4 + d
                                self.selected_square = None
                                self.legal_moves = []
                                return action
                        
                        # If not a legal target, deselect
                        self.selected_square = None
                        self.legal_moves = []
            
            render_game(env, self.selected_square, self.legal_moves)

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, use_cpp=False, model_path=None, device="cpu"):
        self._policy_value_function = policy_value_function
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._model_path = model_path
        self._device = device
        self._use_cpp = use_cpp

        self._init_mcts()

    def _init_mcts(self):
        if self._use_cpp and self._model_path is not None:
            try:
                self.mcts = MCTS_CPP(self._model_path, self._c_puct, self._n_playout, str(self._device))
                print("Using C++ MCTS for AI player.")
            except Exception as e:
                print(f"Failed to load C++ MCTS for AI: {e}. Falling back to Python MCTS.")
                self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout)
                self._use_cpp = False
        else:
            self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout)
            print("Using Python MCTS for AI player.")

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env):
        legal_moves = env.unwrapped.get_legal_actions()
        if len(legal_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp=1e-3)
            move = acts[np.argmax(probs)]
            return move
        else:
            print("WARNING: No sensible moves found!")
            return -1

def render_game(env, selected_square=None, legal_moves=None):
    screen = pygame.display.get_surface()
    if screen is None:
        pygame.init()
        screen = pygame.display.set_mode((700, 700))
        pygame.display.set_caption("Cublino Contra")
        
    screen.fill((255, 255, 255)) # White background
    
    cell_size = 100
    font = pygame.font.SysFont(None, 36)
    
    # Draw Grid
    for r in range(7):
        for c in range(7):
            rect = pygame.Rect(c * cell_size, (6 - r) * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 0, 0), rect, 1)
            
            # Draw Content
            p, top, south = env.unwrapped.board[r, c]
            if p != 0:
                color = (200, 0, 0) if p == 1 else (0, 0, 200) # Red for P1, Blue for P2
                pygame.draw.circle(screen, color, rect.center, 40)
                
                # Draw Values
                text = font.render(f"{top}/{south}", True, (255, 255, 255))
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
                
    # Highlight Selection
    if selected_square:
        r, c = selected_square
        rect = pygame.Rect(c * cell_size, (6 - r) * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (0, 255, 0), rect, 5) # Green border
        
    # Highlight Legal Moves
    if legal_moves:
        for r, c, d in legal_moves:
            rect = pygame.Rect(c * cell_size, (6 - r) * cell_size, cell_size, cell_size)
            s = pygame.Surface((cell_size, cell_size))
            s.set_alpha(100)
            s.fill((0, 255, 0))
            screen.blit(s, rect.topleft)

    pygame.display.flip()

def run_game(model_path=None, human_starts=True, difficulty=20):
    env = gym.make("CublinoContra-v0")
    env.reset()
    
    # Init PyGame
    pygame.init()
    pygame.display.set_mode((700, 700))
    pygame.display.set_caption("Cublino Contra")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    policy_value_net = PolicyValueNet(board_size=7).to(device)
    if model_path and os.path.exists(model_path):
        # Check if it's a state dict or script
        try:
            policy_value_net.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except:
             print(f"Could not load state dict from {model_path}. Assuming it is a script model or invalid.")
             # If it's a script model, we can't load it into PolicyValueNet directly easily for Python inference
             # But we need it for Python fallback.
             # For now, assume state dict.
    elif model_path:
        print(f"Warning: Model path {model_path} does not exist. Using untrained model.")
    else:
        print("No model path provided. Using untrained model.")
        
    # Save model as TorchScript for C++ MCTS
    model_path_cpp = "temp_play_model_cublino.pt"
    script_model = torch.jit.script(policy_value_net.cpu())
    script_model.save(model_path_cpp)
    policy_value_net.to(device)

    def policy_value_fn(env):
        legal_actions = env.get_legal_actions()
        board = env.board
        board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            log_act_probs, value = policy_value_net(board_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        return zip(legal_actions, act_probs[legal_actions]), value.item()

    # Difficulty
    n_playout = difficulty * 20
    print(f"Difficulty Level: {difficulty} (Simulations: {n_playout})")

    # Players
    human = HumanPlayer()
    ai_player = MCTSPlayer(policy_value_fn, c_puct=5, n_playout=n_playout, use_cpp=True, model_path=model_path_cpp, device=device)
    
    players = {1: human, -1: ai_player}
    if not human_starts:
        players = {1: ai_player, -1: human}
            
    human.set_player_ind(1 if human_starts else -1)
    ai_player.set_player_ind(-1 if human_starts else 1)

    obs, info = env.reset()
    done = False
    
    ai_player.reset_player()

    while not done:
        render_game(env, human.selected_square, human.legal_moves)
        
        current_player_idx = env.unwrapped.current_player
        player = players[current_player_idx]
        
        if isinstance(player, HumanPlayer):
            # print("Your turn!")
            action = player.get_action(env)
        else:
            print("AI is thinking...")
            action = player.get_action(env)
        
        ai_player.mcts.update_with_move(action)

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            render_game(env)
            if isinstance(player, HumanPlayer):
                print("You win!")
            else:
                print("AI wins!")
            done = True
            pygame.time.wait(3000)
        elif truncated:
            render_game(env)
            print("Draw!")
            done = True
            pygame.time.wait(3000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model .pth file")
    parser.add_argument("--ai_starts", action="store_true", help="If set, AI starts first")
    parser.add_argument("--difficulty", type=int, default=20, choices=range(1, 21), help="Difficulty level (1-20)")
    args = parser.parse_args()
    
    run_game(model_path=args.model, human_starts=not args.ai_starts, difficulty=args.difficulty)
