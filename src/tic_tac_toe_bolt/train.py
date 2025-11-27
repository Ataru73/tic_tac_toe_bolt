import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import os
import concurrent.futures
import time # Import time module

from tic_tac_toe_bolt.model import PolicyValueNet
from tic_tac_toe_bolt.mcts import MCTS, MCTS_CPP
import tic_tac_toe_bolt # Register env

def run_self_play_worker(model_path, c_puct, n_playout, device_str, temp, num_games_to_play_per_worker):
    """ Worker function for parallel self-play """
    env = gym.make("TicTacToeBolt-v0")
    
    use_cpp = True
    try:
        mcts = MCTS_CPP(model_path, c_puct, n_playout, device_str)
    except Exception as e:
        print(f"Worker failed to use C++ MCTS: {e}. Falling back to Python MCTS.")
        use_cpp = False
        device = torch.device(device_str)
        policy_value_net = torch.jit.load(model_path, map_location=device)
        
        def policy_value_fn(env):
            # Same logic as TrainPipeline.policy_value_fn but standalone
            legal_positions = []
            for i in range(9):
                if env.unwrapped.board[i // 3, i % 3] == 0:
                    legal_positions.append(i)
            
            current_player = env.unwrapped.current_player
            player_moves = env.unwrapped.player_moves
            
            input_board = np.zeros((1, 3, 3, 3))
            
            # Helper to fill channel
            def fill_channel(channel, p):
                if p in player_moves:
                    moves = player_moves[p]
                    n = len(moves)
                    for idx, (r, c) in enumerate(moves):
                        val = 0.0
                        if idx == n - 1: val = 1.0
                        elif idx == n - 2: val = 0.66
                        elif idx == n - 3: val = 0.33
                        input_board[0, channel, r, c] = val

            fill_channel(0, current_player) # Self
            fill_channel(1, -current_player) # Opponent
            input_board[0, 2, :, :] = 1.0 # Turn
            
            input_tensor = torch.FloatTensor(input_board).to(device)
            
            with torch.no_grad():
                output = policy_value_net(input_tensor)
                log_act_probs, value = output
                act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
                
            return zip(legal_positions, act_probs[legal_positions]), value.item()
            
        mcts = MCTS(policy_value_fn, c_puct, n_playout)

    all_play_data = []
    for _ in range(num_games_to_play_per_worker):
        env.reset()
        mcts.update_with_move(-1) # Reset MCTS tree for new game

        states, mcts_probs, current_players = [], [], []
        
        while True:
            # We need the state representation for the buffer
            # Re-generate it here to ensure we save exactly what MCTS sees (conceptually)
            # Actually, MCTS_CPP does this internally. We need to replicate it for the python buffer.
            
            current_player = env.unwrapped.current_player
            player_moves = env.unwrapped.player_moves
            state_tensor = np.zeros((3, 3, 3)) # (Channels, H, W)
            
            def fill_channel_np(channel, p):
                if p in player_moves:
                    moves = player_moves[p]
                    n = len(moves)
                    for idx, (r, c) in enumerate(moves):
                        val = 0.0
                        if idx == n - 1: val = 1.0
                        elif idx == n - 2: val = 0.66
                        elif idx == n - 3: val = 0.33
                        state_tensor[channel, r, c] = val
            
            fill_channel_np(0, current_player)
            fill_channel_np(1, -current_player)
            state_tensor[2, :, :] = 1.0
            
            states.append(state_tensor) # Save the (3,3,3) tensor
            
            acts, probs = mcts.get_move_probs(env, temp=temp)
            
            prob_vec = np.zeros(9)
            for a, p in zip(acts, probs):
                prob_vec[a] = p
            mcts_probs.append(prob_vec)
            current_players.append(env.unwrapped.current_player)
            
            move = np.random.choice(acts, p=probs)
            mcts.update_with_move(move)
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            if terminated:
                actual_winner = -env.unwrapped.current_player
                winner_z = np.zeros(len(current_players))
                winner_z[np.array(current_players) == actual_winner] = 1.0
                winner_z[np.array(current_players) != actual_winner] = -1.0
                
                all_play_data.append(list(zip(states, mcts_probs, winner_z)))
                break # Break from inner while loop to start next game
                
            if truncated:
                winner_z = np.zeros(len(current_players))
                all_play_data.append(list(zip(states, mcts_probs, winner_z)))
                break # Break from inner while loop to start next game

    return all_play_data

class MCTSPlayer:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=1e-3, return_prob=0):
        sensible_moves = [i for i in range(9) if env.unwrapped.board[i//3, i%3] == 0]
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(9)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for self-play training)
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: No sensible moves found!")
            return -1

class TrainPipeline:
    def __init__(self, init_model=None):
        # Params
        self.board_width = 3
        self.board_height = 3
        self.n_in_row = 3
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = 50 # Reduced from 400
        self.c_puct = 8
        self.buffer_size = 10000
        self.batch_size = 64 # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 10 # Number of games to collect per batch iteration
        self.num_games_per_worker = 5 # Each worker plays this many games before returning
        self.epochs = 5 # num_train_steps per batch
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.episode_len = 0
        
        # Environment
        self.env = gym.make("TicTacToeBolt-v0")
        self.eval_env = gym.make("TicTacToeBolt-v0")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model
        if init_model:
            # Check if it's a full checkpoint or just model weights
            checkpoint = torch.load(init_model, map_location=self.device)
            self.policy_value_net = PolicyValueNet().to(self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {init_model}")
                self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-4)
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("Loaded optimizer state.")
            else:
                # Assume it's just state_dict (legacy or simple save)
                self.policy_value_net.load_state_dict(checkpoint)
                print(f"Loaded model weights from {init_model}")
                self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-4)
        else:
            self.policy_value_net = PolicyValueNet().to(self.device)
            self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-4)

    def policy_value_fn(self, env):
        """
        input: env
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = []
        for i in range(9):
            if env.unwrapped.board[i // 3, i % 3] == 0:
                legal_positions.append(i)
        
        current_player = env.unwrapped.current_player
        player_moves = env.unwrapped.player_moves
        
        input_board = np.zeros((1, 3, 3, 3))
        
        def fill_channel(channel, p):
            if p in player_moves:
                moves = player_moves[p]
                n = len(moves)
                for idx, (r, c) in enumerate(moves):
                    val = 0.0
                    if idx == n - 1: val = 1.0
                    elif idx == n - 2: val = 0.66
                    elif idx == n - 3: val = 0.33
                    input_board[0, channel, r, c] = val

        fill_channel(0, current_player) # Self
        fill_channel(1, -current_player) # Opponent
        input_board[0, 2, :, :] = 1.0 # Turn
        
        input_tensor = torch.FloatTensor(input_board).to(self.device)
        
        # Predict
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(input_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        # Mask illegal moves
        probs = zip(legal_positions, act_probs[legal_positions])
        return probs, value.item()
    
    def get_policy_value_fn(self, policy_value_net):
        def policy_value_fn(env):
            legal_positions = []
            for i in range(9):
                if env.unwrapped.board[i // 3, i % 3] == 0:
                    legal_positions.append(i)
            
            current_player = env.unwrapped.current_player
            player_moves = env.unwrapped.player_moves
            
            input_board = np.zeros((1, 3, 3, 3))
            
            def fill_channel(channel, p):
                if p in player_moves:
                    moves = player_moves[p]
                    n = len(moves)
                    for idx, (r, c) in enumerate(moves):
                        val = 0.0
                        if idx == n - 1: val = 1.0
                        elif idx == n - 2: val = 0.66
                        elif idx == n - 3: val = 0.33
                        input_board[0, channel, r, c] = val

            fill_channel(0, current_player)
            fill_channel(1, -current_player)
            input_board[0, 2, :, :] = 1.0
            
            input_tensor = torch.FloatTensor(input_board).to(self.device)

            with torch.no_grad():
                log_act_probs, value = policy_value_net(input_tensor)
                act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
                
            return zip(legal_positions, act_probs[legal_positions]), value.item()
        return policy_value_fn

    def evaluate_policy(self, n_games=10):
        """
        Evaluate the trained policy by playing against the best policy
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        best_mcts_player = MCTSPlayer(self.get_policy_value_fn(self.best_policy_net), c_puct=self.c_puct, n_playout=self.n_playout)
        
        win_cnt = {1: 0, -1: 0, 0: 0} # 1: current, -1: best, 0: draw
        
        for i in range(n_games):
            # Alternate start
            if i % 2 == 0:
                # Current starts (Player 1), Best is Player -1
                players = {1: current_mcts_player, -1: best_mcts_player}
                current_player_key = 1
            else:
                # Best starts (Player 1), Current is Player -1
                players = {1: best_mcts_player, -1: current_mcts_player}
                current_player_key = -1
                
            obs, _ = self.eval_env.reset()
            current_mcts_player.reset_player()
            best_mcts_player.reset_player()
            done = False
            
            while not done:
                current_player_idx = self.eval_env.unwrapped.current_player
                player = players[current_player_idx]
                action = player.get_action(self.eval_env)
                
                # Update both players with the move (essential for correct MCTS state)
                current_mcts_player.mcts.update_with_move(action)
                best_mcts_player.mcts.update_with_move(action)
                
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                
                if terminated:
                    # current_player_idx won
                    if current_player_idx == current_player_key:
                        win_cnt[1] += 1
                    else:
                        win_cnt[-1] += 1
                    done = True
                elif truncated:
                    win_cnt[0] += 1
                    done = True
                    
        return win_cnt

    def run(self):
        # Initialize best policy as current policy
        self.best_policy_net = copy.deepcopy(self.policy_value_net)
        
        try:
            for i in range(self.game_batch_num):
                start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                end_time = time.time()
                duration = end_time - start_time
                games_per_second = self.play_batch_size / duration if duration > 0 else 0
                print(f"Batch i:{i+1}, Episode Len:{self.episode_len}, Games/s: {games_per_second:.2f}")
                if len(self.data_buffer) > self.batch_size:
                    # Save current weights before update
                    old_params = copy.deepcopy(self.policy_value_net.state_dict())
                    
                    loss, entropy = self.policy_update()
                    print(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}")
                    
                    # Evaluate
                    if (i+1) % self.check_freq == 0:
                        print("Evaluating new policy against best policy...")
                        win_cnt = self.evaluate_policy(n_games=10)
                        print(f"Eval Results (Current vs Best): Wins: {win_cnt[1]}, Losses: {win_cnt[-1]}, Draws: {win_cnt[0]}")
                        
                        win_ratio = 1.0 * (win_cnt[1] + 0.5*win_cnt[0]) / (sum(win_cnt.values()))
                        print(f"Win Ratio: {win_ratio:.2f}")
                        
                        if win_ratio >= 0.55: # If improvement  
                            print("New best policy found! Saving...")
                            self.best_policy_net.load_state_dict(self.policy_value_net.state_dict())
                            torch.save(self.policy_value_net.state_dict(), f'current_policy_{i+1}.pth')
                        else:
                            print("New policy not better. NOT Reverting.")
                            # COMMENT OUT THESE LINES:
                            # self.policy_value_net.load_state_dict(old_params)
                            # print("Reverted to old params")   
                                                     
        except KeyboardInterrupt:
            print('\n\rquit')
            print("Saving checkpoint...")
            checkpoint = {
                'model_state_dict': self.policy_value_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }
            torch.save(checkpoint, 'checkpoint.pth')

    def collect_selfplay_data(self, n_games=1):
        # Save model for workers
        model_path = "temp_model.pt"
        script_model = torch.jit.script(self.policy_value_net)
        script_model.save(model_path)
        
        device_str = str(self.device)
        
        # Determine number of workers
        # We want to collect n_games in total
        # Each worker plays self.num_games_per_worker
        num_workers = (n_games + self.num_games_per_worker - 1) // self.num_games_per_worker
        max_workers = min(num_workers, os.cpu_count() or 4)
        
        self.total_episode_len = 0 # Track total episode length for averaging
        self.collected_games = 0 # Track collected games

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_self_play_worker, model_path, self.c_puct, self.n_playout, device_str, self.temp, self.num_games_per_worker)
                for _ in range(num_workers)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    list_of_games = future.result() # Each worker returns a list of games (each game is a list of steps)
                    for game_steps in list_of_games:
                        self.total_episode_len += len(game_steps)
                        self.collected_games += 1
                        # augment the data
                        augmented_data = self.get_equi_data(game_steps) 
                        self.data_buffer.extend(augmented_data)
                except Exception as e:
                    print(f"Self-play worker generated an exception: {e}")

        # Update self.episode_len with average for logging
        self.episode_len = self.total_episode_len / self.collected_games if self.collected_games > 0 else 0

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            # state: (3, 3, 3) tensor (C, H, W)
            # mcts_prob: 9-dim vector
            for i in [1, 2, 3, 4]:
                # rotate counter-clockwise
                # np.rot90 rotates axes (0, 1) by default.
                # state is (C, H, W). We want to rotate H, W (axes 1, 2).
                equi_state = np.rot90(state, i, axes=(1, 2))
                equi_mcts_prob = np.rot90(mcts_prob.reshape(3, 3), i)
                extend_data.append((equi_state, equi_mcts_prob.flatten(), winner))
                
                # flip horizontally (axis 2)
                equi_state_flip = np.flip(equi_state, axis=2)
                equi_mcts_prob_flip = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state_flip, equi_mcts_prob_flip.flatten(), winner))
        return extend_data

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        # Convert to tensors
        # state_batch is already a list of (3, 3, 3) numpy arrays
        state_batch_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
            
        mcts_probs_tensor = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_tensor = torch.FloatTensor(np.array(winner_batch)).to(self.device)
        
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
import multiprocessing
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    training = TrainPipeline(init_model=args.resume)
    training.run()
