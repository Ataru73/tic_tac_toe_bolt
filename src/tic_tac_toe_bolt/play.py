import gymnasium as gym
import os
import torch
import numpy as np
import pygame
import argparse
import sys

from tic_tac_toe_bolt.model import PolicyValueNet
from tic_tac_toe_bolt.mcts import MCTS
import tic_tac_toe_bolt # Register env

try:
    from tic_tac_toe_bolt import _mcts_cpp
except ImportError:
    _mcts_cpp = None

class HumanPlayer:
    def __init__(self):
        self.player = None
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        # Wait for mouse click
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    # Map to grid
                    # Window size is 512
                    cell_size = 512 / 3
                    col = int(x // cell_size)
                    row = int(y // cell_size)
                    action = row * 3 + col
                    
                    # Check if valid (not occupied)
                    if env.unwrapped.board[row, col] == 0:
                        return action
            
            env.render()

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
        if self._use_cpp and _mcts_cpp is not None and self._model_path is not None:
            try:
                from tic_tac_toe_bolt.mcts import MCTS_CPP # Import from the correct mcts.py
                self.mcts = MCTS_CPP(self._model_path, self._c_puct, self._n_playout, str(self._device))
                print("Using C++ MCTS for AI player.")
            except Exception as e:
                print(f"Failed to load C++ MCTS for AI: {e}. Falling back to Python MCTS.")
                from tic_tac_toe_bolt.mcts import MCTS
                self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout)
                self._use_cpp = False # Indicate fallback
        else:
            from tic_tac_toe_bolt.mcts import MCTS
            self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout)
            print("Using Python MCTS for AI player.")

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        # For Python MCTS, _root is directly exposed, update_with_move(-1) resets it.
        # For C++ MCTS, update_with_move(-1) also resets it via the C++ binding.
        self.mcts.update_with_move(-1)

    def get_action(self, env):
        sensible_moves = [i for i in range(9) if env.unwrapped.board[i//3, i%3] == 0]
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp=1e-3)
            # Choose action with highest prob
            move = acts[np.argmax(probs)]
            return move
        else:
            print("WARNING: No sensible moves found!")
            return -1

def run_game(model_path=None, human_starts=True):
    env = gym.make("TicTacToeBolt-v0", render_mode="human")
    env.reset()
    env.render() # Init pygame
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    policy_value_net = PolicyValueNet().to(device)
    if model_path and os.path.exists(model_path):
        policy_value_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    elif model_path:
        print(f"Warning: Model path {model_path} does not exist. Using untrained model.")
    else:
        print("No model path provided. Using untrained model.")
        
    # Save model as TorchScript for C++ MCTS (must be done after loading to device)
    model_path_cpp = "temp_play_model.pt"
    script_model = torch.jit.script(policy_value_net.cpu()) # Script on CPU, C++ MCTS will load to target device
    script_model.save(model_path_cpp)
    policy_value_net.to(device) # Move original model back to device if it was on CUDA

    def policy_value_fn(env):
        board = env.unwrapped.board
        current_player = env.unwrapped.current_player
        canonical_board = board * current_player
        
        input_board = np.zeros((1, 3, 3, 3))
        input_board[0, 0, :, :] = (canonical_board == 1)
        input_board[0, 1, :, :] = (canonical_board == -1)
        input_board[0, 2, :, :] = 1.0
        
        input_tensor = torch.FloatTensor(input_board).to(device)
        
        legal_positions = []
        for i in range(9):
            if board[i // 3, i % 3] == 0:
                legal_positions.append(i)

        with torch.no_grad():
            log_act_probs, value = policy_value_net(input_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        return zip(legal_positions, act_probs[legal_positions]), value.item()

    # Players
    human = HumanPlayer()
    ai_player = MCTSPlayer(policy_value_fn, c_puct=5, n_playout=400, use_cpp=True, model_path=model_path_cpp, device=device)
    
    players = {1: human, -1: ai_player}
    if not human_starts:
        players = {1: ai_player, -1: human}
            
    obs, info = env.reset()
    done = False
    
    # Reset MCTS trees for new game
    ai_player.reset_player()

    while not done:
        current_player_idx = env.unwrapped.current_player
        player = players[current_player_idx]
        
        if isinstance(player, HumanPlayer):
            print("Your turn!")
            action = player.get_action(env)
        else:
            print("AI is thinking...")
            # Get NN evaluation for printing (policy_value_fn already returns policy/value)
            legal_moves_and_probs, nn_value = policy_value_fn(env)
            
            print(f"NN Value (from AI's perspective): {nn_value:.4f}")
            print("NN Policy (legal moves and probabilities):")
            sorted_policy = sorted(list(legal_moves_and_probs), key=lambda x: x[1], reverse=True)
            for move, prob in sorted_policy:
                print(f"  Move {move}: {prob:.4f}")

            action = player.get_action(env)
        
        # Update AI's MCTS with the action taken
        ai_player.mcts.update_with_move(action)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated:
            if isinstance(player, HumanPlayer):
                print("You win!")
            else:
                print("AI wins!")
            done = True
            pygame.time.wait(3000)
        elif truncated:
            print("Draw!")
            done = True
            pygame.time.wait(3000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model .pth file")
    parser.add_argument("--ai_starts", action="store_true", help="If set, AI starts first")
    args = parser.parse_args()
    
    run_game(model_path=args.model, human_starts=not args.ai_starts)
