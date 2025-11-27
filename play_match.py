import gymnasium as gym
import torch
import numpy as np
import argparse
import sys
import os

from tic_tac_toe_bolt.model import PolicyValueNet
from tic_tac_toe_bolt.mcts import MCTS
import tic_tac_toe_bolt # Register env

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
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

def get_policy_value_fn(policy_value_net):
    def policy_value_fn(env):
        board = env.unwrapped.board
        current_player = env.unwrapped.current_player
        canonical_board = board * current_player
        
        input_board = np.zeros((1, 3, 3, 3))
        input_board[0, 0, :, :] = (canonical_board == 1)
        input_board[0, 1, :, :] = (canonical_board == -1)
        input_board[0, 2, :, :] = 1.0
        
        input_tensor = torch.FloatTensor(input_board)
        
        legal_positions = []
        for i in range(9):
            if board[i // 3, i % 3] == 0:
                legal_positions.append(i)

        if policy_value_net is None:
            # Random policy
            act_probs = np.ones(9) / 9
            value = 0.0
        else:
            with torch.no_grad():
                log_act_probs, value = policy_value_net(input_tensor)
                act_probs = np.exp(log_act_probs.numpy().flatten())
            value = value.item()
            
        return zip(legal_positions, act_probs[legal_positions]), value
    return policy_value_fn

def load_model(model_path):
    if model_path and os.path.exists(model_path):
        policy_value_net = PolicyValueNet()
        policy_value_net.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
        return policy_value_net
    elif model_path:
        print(f"Warning: Model path {model_path} does not exist. Using random/untrained model.")
        return None
    else:
        print("No model path provided. Using random/untrained model.")
        return None

def evaluate_models(model1_path, model2_path, n_games, c_puct, n_playout):
    env = gym.make("TicTacToeBolt-v0")
    
    net1 = load_model(model1_path)
    net2 = load_model(model2_path)
    
    player1 = MCTSPlayer(get_policy_value_fn(net1), c_puct=c_puct, n_playout=n_playout)
    player2 = MCTSPlayer(get_policy_value_fn(net2), c_puct=c_puct, n_playout=n_playout)
    
    # Results: [Model 1 wins, Model 2 wins, Draws]
    results = [0, 0, 0]
    
    for i in range(n_games):
        player1.reset_player()
        player2.reset_player()

        # Alternate starting player
        # If i is even, Model 1 is Player 1 (starts).
        # If i is odd, Model 2 is Player 1 (starts).
        
        if i % 2 == 0:
            players = {1: player1, -1: player2}
            p1_is_model1 = True
        else:
            players = {1: player2, -1: player1}
            p1_is_model1 = False
            
        obs, info = env.reset()
        done = False
        
        while not done:
            current_player_idx = env.unwrapped.current_player
            player = players[current_player_idx]
            action = player.get_action(env)

            # Update both players' MCTS with the chosen action
            player1.mcts.update_with_move(action)
            player2.mcts.update_with_move(action)

            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # current_player_idx won
                if (current_player_idx == 1 and p1_is_model1) or (current_player_idx == -1 and not p1_is_model1):
                    results[0] += 1
                else:
                    results[1] += 1
                done = True
            elif truncated:
                results[2] += 1
                done = True
                
        print(f"Game {i+1}/{n_games} finished. Current stats: M1: {results[0]}, M2: {results[1]}, Draw: {results[2]}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a match between two models")
    parser.add_argument("--model1", type=str, help="Path to model 1 .pth file")
    parser.add_argument("--model2", type=str, help="Path to model 2 .pth file")
    parser.add_argument("--n_games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--c_puct", type=float, default=5, help="MCTS c_puct")
    parser.add_argument("--n_playout", type=int, default=400, help="MCTS n_playout")
    
    args = parser.parse_args()
    
    results = evaluate_models(args.model1, args.model2, args.n_games, args.c_puct, args.n_playout)
    
    print("\nFinal Results:")
    print(f"Model 1 Wins: {results[0]}")
    print(f"Model 2 Wins: {results[1]}")
    print(f"Draws: {results[2]}")
