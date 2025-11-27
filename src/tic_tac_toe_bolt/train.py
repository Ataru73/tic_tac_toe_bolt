import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import os

from tic_tac_toe_bolt.model import PolicyValueNet
from tic_tac_toe_bolt.mcts import MCTS
import tic_tac_toe_bolt # Register env

class TrainPipeline:
    def __init__(self, init_model=None):
        # Params
        self.board_width = 3
        self.board_height = 3
        self.n_in_row = 3
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 64 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 10
        self.epochs = 5 # num_train_steps per batch
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        
        # Environment
        self.env = gym.make("TicTacToeBolt-v0")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model
        if init_model:
            self.policy_value_net = torch.load(init_model, map_location=self.device)
        else:
            self.policy_value_net = PolicyValueNet().to(self.device)
            
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-4)

    def policy_value_fn(self, env):
        """
        input: env
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        # Get legal moves
        # In our env, we can check board for 0s.
        board = env.unwrapped.board
        legal_positions = np.argwhere(board.flatten() == 0).flatten() # Wait, infinite mechanic means all moves are valid unless occupied?
        # Yes, but occupied squares are invalid.
        # Actually, if we place on occupied, we get -10.
        # So we should consider occupied as illegal for MCTS to avoid wasting simulations.
        legal_positions = []
        for i in range(9):
            r, c = i // 3, i % 3
            if board[r, c] == 0:
                legal_positions.append(i)
        
        # Prepare input
        # Channel 0: Current player marks (1s)
        # Channel 1: Opponent marks (-1s)
        # Channel 2: 1s (Turn)
        current_player = env.unwrapped.current_player
        canonical_board = board * current_player
        
        input_board = np.zeros((1, 3, 3, 3))
        # Channel 0: My marks (where canonical_board == 1)
        input_board[0, 0, :, :] = (canonical_board == 1)
        # Channel 1: Opponent marks (where canonical_board == -1)
        input_board[0, 1, :, :] = (canonical_board == -1)
        # Channel 2
        input_board[0, 2, :, :] = 1.0
        
        input_tensor = torch.FloatTensor(input_board).to(self.device)
        
        # Predict
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(input_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        # Mask illegal moves
        probs = zip(legal_positions, act_probs[legal_positions])
        return probs, value.item()

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print(f"Batch i:{i+1}, Episode Len:{self.episode_len}")
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}")
                    
                if (i+1) % self.check_freq == 0:
                    print("Saving model...")
                    torch.save(self.policy_value_net.state_dict(), f'current_policy_{i+1}.pth')
        except KeyboardInterrupt:
            print('\n\rquit')

    def collect_selfplay_data(self, n_games=1):
        for _ in range(n_games):
            winner, play_data = self.start_self_play(self.env, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def start_self_play(self, env, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        env.reset()
        mcts = MCTS(self.policy_value_fn, self.c_puct, self.n_playout)
        states, mcts_probs, current_players = [], [], []
        
        while True:
            # Get move probabilities
            acts, probs = mcts.get_move_probs(env, temp=temp)
            
            # Store data
            # Store canonical board: 1 for current player, -1 for opponent
            canonical_board = env.unwrapped.board.copy() * env.unwrapped.current_player
            states.append(canonical_board)
            
            mcts_probs.append(probs) # This is a list of probabilities for all actions?
            # get_move_probs returns acts, act_probs. 
            # We need a full 9-dim vector.
            prob_vec = np.zeros(9)
            for a, p in zip(acts, probs):
                prob_vec[a] = p
            mcts_probs[-1] = prob_vec
            current_players.append(env.unwrapped.current_player)
            
            # Perform a move
            move = np.random.choice(acts, p=probs)
            mcts.update_with_move(move)
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            if terminated:
                # Winner is the current player (who just moved)
                # The values in winner_z should be relative to the player at that step.
                # If player P1 won, then for all steps where P1 moved, return 1.
                # For all steps where P2 moved, return -1.
                winner_z = np.zeros(len(current_players))
                winner_z[np.array(current_players) == env.unwrapped.current_player] = 1.0
                winner_z[np.array(current_players) != env.unwrapped.current_player] = -1.0
                
                # Reset MCTS
                mcts.update_with_move(-1) 
                return env.unwrapped.current_player, zip(states, mcts_probs, winner_z)
                
            if truncated: # Should not happen in infinite tic tac toe usually, unless we set a limit
                # Draw? Or just stop.
                # Let's assume draw.
                winner_z = np.zeros(len(current_players))
                return 0, zip(states, mcts_probs, winner_z)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # state: 3x3 board
            # mcts_prob: 9-dim vector
            for i in [1, 2, 3, 4]:
                # rotate counter-clockwise
                equi_state = np.array([np.rot90(state, i)])
                equi_mcts_prob = np.rot90(mcts_prob.reshape(3, 3), i)
                extend_data.append((equi_state[0], equi_mcts_prob.flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(state)])
                equi_mcts_prob = np.fliplr(mcts_prob.reshape(3, 3))
                extend_data.append((equi_state[0], equi_mcts_prob.flatten(), winner))
        return extend_data

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        # Convert to tensors
        # State batch needs to be converted to (Batch, 3, 3, 3)
        state_batch_tensor = torch.zeros(self.batch_size, 3, 3, 3)
        for i, board in enumerate(state_batch):
            state_batch_tensor[i, 0, :, :] = torch.from_numpy((board == 1).astype(np.float32))
            state_batch_tensor[i, 1, :, :] = torch.from_numpy((board == -1).astype(np.float32))
            state_batch_tensor[i, 2, :, :] = 1.0
            
        mcts_probs_tensor = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_tensor = torch.FloatTensor(np.array(winner_batch)).to(self.device)
        state_batch_tensor = state_batch_tensor.to(self.device)
        
        # Train
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            log_act_probs, value = self.policy_value_net(state_batch_tensor)
            
            # Value loss
            value_loss = F.mse_loss(value.view(-1), winner_tensor)
            
            # Policy loss
            policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_act_probs, 1))
            
            loss = value_loss + policy_loss
            loss.backward()
            self.optimizer.step()
            
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
            
        return loss.item(), entropy.item()

import torch.nn.functional as F

if __name__ == "__main__":
    training = TrainPipeline()
    training.run()
