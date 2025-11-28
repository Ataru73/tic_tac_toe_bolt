import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
import os
import concurrent.futures
import time
import argparse
import multiprocessing

from src.cublino_contra.model import PolicyValueNet
from src.cublino_contra.mcts import MCTS, MCTS_CPP
from src.cublino_contra.env import CublinoContraEnv

def run_self_play_worker(model_path, c_puct, n_playout, device_str, temp, num_games_to_play_per_worker):
    """ Worker function for parallel self-play """
    env = CublinoContraEnv()
    
    use_cpp = True
    try:
        mcts = MCTS_CPP(model_path, c_puct, n_playout, device_str)
    except Exception as e:
        print(f"Worker failed to use C++ MCTS: {e}. Falling back to Python MCTS.")
        use_cpp = False
        device = torch.device(device_str)
        # Load model
        policy_value_net = PolicyValueNet(board_size=7).to(device)
        policy_value_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_value_net.eval()
        
        def policy_value_fn(state):
            # state is the env
            legal_actions = state.get_legal_actions()
            
            board = state.board
            # Convert to tensor: (1, 3, 7, 7)
            board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                log_act_probs, value = policy_value_net(board_tensor)
                act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
                
            return zip(legal_actions, act_probs[legal_actions]), value.item()
            
        mcts = MCTS(policy_value_fn, c_puct, n_playout)

    all_play_data = []
    for _ in range(num_games_to_play_per_worker):
        env.reset()
        mcts.update_with_move(-1) # Reset MCTS tree

        states, mcts_probs, current_players = [], [], []
        
        while True:
            # Store state as (3, 7, 7) for training
            board = env.board
            state_tensor = np.transpose(board, (2, 0, 1)) # (3, 7, 7)
            states.append(state_tensor)
            
            acts, probs = mcts.get_move_probs(env, temp=temp)
            
            prob_vec = np.zeros(196) # 7*7*4
            for a, p in zip(acts, probs):
                prob_vec[a] = p
            mcts_probs.append(prob_vec)
            current_players.append(env.current_player)
            
            move = np.random.choice(acts, p=probs)
            mcts.update_with_move(move)
            
            obs, reward, terminated, truncated, info = env.step(move)
            
            if terminated or truncated:
                # Winner is the one who just moved (current_player of the step that caused termination)
                # Wait, env.step switches turn at the end.
                # So if terminated, the player who made the move is -env.current_player
                # But my MCTS implementation handles value from perspective of current player.
                
                # Let's look at reward.
                # If P1 wins, reward is 1.
                # If P2 wins, reward is 1 (from their perspective? No, env returns 1 if winner found).
                # Let's check env.py again.
                # if self.current_player == 1 and target_row == 6: return ..., 1, True, ...
                # Then switch turn? No, returns immediately.
                # So if terminated, the current_player (who made the move) is the winner.
                
                # In train.py logic:
                # actual_winner = env.current_player (because env.step returns before switching turn on win)
                # Wait, let's check env.py carefully.
                
                # env.py:
                # if win condition: return ..., 1, True, ... {"winner": 1}
                # It does NOT switch turn.
                # So env.current_player is the winner.
                
                winner_val = 0
                if 'winner' in info:
                    winner_val = info['winner']
                
                winner_z = np.zeros(len(current_players))
                if winner_val != 0:
                    winner_z[np.array(current_players) == winner_val] = 1.0
                    winner_z[np.array(current_players) != winner_val] = -1.0
                
                all_play_data.append(list(zip(states, mcts_probs, winner_z)))
                break

    return all_play_data

class MCTSPlayer:
    def __init__(self, policy_value_fn, c_puct=5, n_playout=400, is_selfplay=0):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env, temp=1e-3, return_prob=0):
        legal_moves = env.get_legal_actions()
        move_probs = np.zeros(196)
        if len(legal_moves) > 0:
            acts, probs = self.mcts.get_move_probs(env, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # Dirichlet Noise
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            return -1

class TrainPipeline:
    def __init__(self, init_model=None):
        self.board_size = 7
        self.learn_rate = 2e-3
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 64
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.num_games_per_worker = 1
        self.epochs = 5
        self.check_freq = 50
        self.game_batch_num = 1500
        self.episode_len = 0
        
        self.env = CublinoContraEnv()
        self.eval_env = CublinoContraEnv()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_value_net = PolicyValueNet(board_size=self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-4)

        if init_model:
            checkpoint = torch.load(init_model, map_location=self.device)
            self.policy_value_net.load_state_dict(checkpoint)
            print(f"Loaded model from {init_model}")

    def policy_value_fn(self, env):
        legal_actions = env.get_legal_actions()
        board = env.board
        board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(board_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        return zip(legal_actions, act_probs[legal_actions]), value.item()
    
    def get_policy_value_fn(self, policy_value_net):
        def policy_value_fn(env):
            legal_actions = env.get_legal_actions()
            board = env.board
            board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_act_probs, value = policy_value_net(board_tensor)
                act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            return zip(legal_actions, act_probs[legal_actions]), value.item()
        return policy_value_fn

    def evaluate_policy(self, n_games=10):
        current_mcts_player = MCTSPlayer(self.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        best_mcts_player = MCTSPlayer(self.get_policy_value_fn(self.best_policy_net), c_puct=self.c_puct, n_playout=self.n_playout)
        
        win_cnt = {1: 0, -1: 0, 0: 0}
        
        for i in range(n_games):
            if i % 2 == 0:
                players = {1: current_mcts_player, -1: best_mcts_player}
                current_player_key = 1
            else:
                players = {1: best_mcts_player, -1: current_mcts_player}
                current_player_key = -1
                
            self.eval_env.reset()
            current_mcts_player.reset_player()
            best_mcts_player.reset_player()
            done = False
            
            while not done:
                current_player_idx = self.eval_env.current_player
                player = players[current_player_idx]
                action = player.get_action(self.eval_env)
                
                current_mcts_player.mcts.update_with_move(action)
                best_mcts_player.mcts.update_with_move(action)
                
                _, _, terminated, truncated, info = self.eval_env.step(action)
                
                if terminated:
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
        self.best_policy_net = copy.deepcopy(self.policy_value_net)
        
        try:
            for i in range(self.game_batch_num):
                start_time = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                print(f"Batch i:{i+1}, Episode Len:{self.episode_len:.2f}, Time:{time.time()-start_time:.2f}s")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}")
                    
                    if (i+1) % self.check_freq == 0:
                        print("Evaluating...")
                        win_cnt = self.evaluate_policy(n_games=10)
                        print(f"Win/Loss/Draw: {win_cnt[1]}/{win_cnt[-1]}/{win_cnt[0]}")
                        
                        win_ratio = 1.0 * (win_cnt[1] + 0.5*win_cnt[0]) / (sum(win_cnt.values()))
                        if win_ratio >= 0.55:
                            print("New best policy!")
                            self.best_policy_net.load_state_dict(self.policy_value_net.state_dict())
                            torch.save(self.policy_value_net.state_dict(), f'current_policy_cublino_{i+1}.pth')
                        else:
                            print("Not better.")
                            
        except KeyboardInterrupt:
            print("Saving checkpoint...")
            torch.save(self.policy_value_net.state_dict(), 'checkpoint_cublino.pth')

    def collect_selfplay_data(self, n_games=1):
        model_path = "temp_model_cublino.pt"
        # Save as TorchScript for C++ MCTS
        script_model = torch.jit.script(self.policy_value_net)
        script_model.save(model_path)
        
        device_str = str(self.device)
        num_workers = (n_games + self.num_games_per_worker - 1) // self.num_games_per_worker
        max_workers = min(num_workers, os.cpu_count() or 4)
        
        self.total_episode_len = 0
        self.collected_games = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_self_play_worker, model_path, self.c_puct, self.n_playout, device_str, self.temp, self.num_games_per_worker)
                for _ in range(num_workers)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    list_of_games = future.result()
                    for game_steps in list_of_games:
                        self.total_episode_len += len(game_steps)
                        self.collected_games += 1
                        self.data_buffer.extend(game_steps)
                except Exception as e:
                    print(f"Worker exception: {e}")

        self.episode_len = self.total_episode_len / self.collected_games if self.collected_games > 0 else 0

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        state_batch_tensor = torch.FloatTensor(np.array(state_batch)).to(self.device)
        mcts_probs_tensor = torch.FloatTensor(np.array(mcts_probs_batch)).to(self.device)
        winner_tensor = torch.FloatTensor(np.array(winner_batch)).to(self.device)
        
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            log_act_probs, value = self.policy_value_net(state_batch_tensor)
            
            value_loss = F.mse_loss(value.view(-1), winner_tensor)
            policy_loss = -torch.mean(torch.sum(mcts_probs_tensor * log_act_probs, 1))
            
            loss = value_loss + policy_loss
            loss.backward()
            self.optimizer.step()
            
            entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
            
        return loss.item(), entropy.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    training = TrainPipeline(init_model=args.resume)
    if args.dry_run:
        training.game_batch_num = 1
        training.play_batch_size = 1
        training.n_playout = 10
        training.epochs = 1
        
    training.run()
