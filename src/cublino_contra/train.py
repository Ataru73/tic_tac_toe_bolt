import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import random
from collections import deque
import copy
import os
import concurrent.futures
import time
import argparse
import multiprocessing
import sys

try:
    from src.cublino_contra.model import PolicyValueNet
    from src.cublino_contra.mcts import MCTS, MCTS_CPP
    from src.cublino_contra.env import CublinoContraEnv
except ImportError:
    # If running as script from src/cublino_contra/
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from cublino_contra.model import PolicyValueNet
    from cublino_contra.mcts import MCTS, MCTS_CPP
    from cublino_contra.env import CublinoContraEnv

def run_self_play_worker(model_path, c_puct, n_playout, device_str, temp, num_games_to_play_per_worker, draw_reward, log_game=False):
    """ Worker function for parallel self-play """
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        env = CublinoContraEnv()
        
        # Enable C++ MCTS
        use_cpp = True
        try:
            mcts = MCTS_CPP(model_path, c_puct, n_playout, device_str)
        except Exception as e:
            print(f"Worker failed to use C++ MCTS: {e}. Falling back to Python MCTS.")
            use_cpp = False
        
        if not use_cpp:
            device = torch.device(device_str)
            # Load model - handle TorchScript format
            policy_value_net = PolicyValueNet(board_size=7).to(device)
            # The model_path is a TorchScript file, so load it and extract state_dict
            scripted_model = torch.jit.load(model_path, map_location=device)
            policy_value_net.load_state_dict(scripted_model.state_dict())
            policy_value_net.eval()
            
            def policy_value_fn(state):
                # state is the env
                legal_actions = state.get_legal_actions()
                
                obs = state._get_obs()  # Get stacked observation (7, 7, 12)
                # Convert to tensor: (1, 12, 7, 7)
                obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    log_act_probs, value = policy_value_net(obs_tensor)
                    act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
                    
                return zip(legal_actions, act_probs[legal_actions]), value.item()
                
            mcts = MCTS(policy_value_fn, c_puct, n_playout)

        all_play_data = []
        worker_game_stats = {1: 0, -1: 0, 0: 0} # 1: P1 wins, -1: P2 wins, 0: Draws
        game_log_data = None # To store the log of the first game if requested
    
        for game_idx in range(num_games_to_play_per_worker):
            env.reset()

            # --- ADD THIS BLOCK ---
            # Randomize the opening to break "passive" memorization
            # Play 0 to 8 random moves (half moves, so 0-4 full turns)
            num_random_moves = np.random.randint(0, 9) 
            for _ in range(num_random_moves):
                legal = env.get_legal_actions()
                if not legal: break
                random_action = np.random.choice(legal)
                env.step(random_action)
            # ----------------------

            mcts.update_with_move(-1) # Reset MCTS tree
    
            states, mcts_probs, current_players = [], [], []
            game_moves = [] # Store moves for logging
            divergence_detected = False
            illegal_moves_log = []
            
            while True:
                # Store state as (12, 7, 7) for training - 4 stacked states * 3 channels
                obs = env._get_obs()  # Get stacked observation (7, 7, 12)
                state_tensor = np.transpose(obs, (2, 0, 1))  # (12, 7, 7)
                states.append(state_tensor)
                
                legal_moves = env.get_legal_actions()
                if len(legal_moves) == 0:
                    # No legal moves - Loss for current player
                    winner_val = -env.current_player
                    worker_game_stats[winner_val] += 1
                    
                    winner_z = np.zeros(len(current_players))
                    if winner_val != 0:
                        winner_z[np.array(current_players) == winner_val] = 1.0
                        winner_z[np.array(current_players) != winner_val] = -1.0
                    else: 
                        winner_z[:] = draw_reward
                    
                    all_play_data.append(list(zip(states, mcts_probs, winner_z)))
                    break

                acts, probs = mcts.get_move_probs(env, temp=temp if len(states) < 30 else 1e-3)
                
                # Safety Filter: Ensure MCTS only proposes legal moves
                # This handles rare state divergence issues
                legal_acts_set = set(legal_moves)
                valid_indices = [i for i, a in enumerate(acts) if a in legal_acts_set]
                
                if len(valid_indices) < len(acts):
                    divergence_detected = True
                    illegal_ones = [a for a in acts if a not in legal_acts_set]
                    illegal_moves_log.append({"step": len(game_moves), "illegal_acts": illegal_ones})
                    # print(f"WARNING: MCTS proposed illegal moves: {illegal_ones}")

                if not valid_indices:
                    # print(f"WARNING: MCTS proposed NO legal moves! Acts: {acts}, Legal: {legal_moves}")
                    # Fallback: Uniform random legal move
                    acts = legal_moves
                    probs = np.ones(len(legal_moves)) / len(legal_moves)
                else:
                    acts = [acts[i] for i in valid_indices]
                    probs = [probs[i] for i in valid_indices]
                    # Renormalize
                    probs = np.array(probs)
                    probs /= probs.sum()

                prob_vec = np.zeros(196) # 7*7*4
                for a, p in zip(acts, probs):
                    prob_vec[a] = p
                mcts_probs.append(prob_vec)
                current_players.append(env.current_player)
                
                move = np.random.choice(acts, p=probs)
                mcts.update_with_move(move)
                
                game_moves.append(int(move))

                obs, reward, terminated, truncated, info = env.step(move)
                
                if terminated or truncated:
                    if 'error' in info or divergence_detected:
                        import json
                        timestamp = int(time.time())
                        suffix = random.randint(0, 10000)
                        log_type = "error" if 'error' in info else "divergence"
                        filename = f"{log_type}_log_{timestamp}_{suffix}.json"
                        error_log_data = {
                            "winner": 0,
                            "moves": game_moves,
                            "error": info.get('error', 'MCTS proposed illegal moves'),
                            "divergence_details": illegal_moves_log
                        }
                        try:
                            with open(filename, 'w') as f:
                                json.dump(error_log_data, f)
                            print(f"{log_type.capitalize()} logged to {filename}")
                        except Exception as e:
                            print(f"Failed to write log: {e}")

                    winner_val = info.get('winner', 0)
                    worker_game_stats[winner_val] += 1
                    
                    winner_z = np.zeros(len(current_players))
                    if winner_val != 0:
                        winner_z[np.array(current_players) == winner_val] = 1.0
                        winner_z[np.array(current_players) != winner_val] = -1.0
                    else: # Draw
                        winner_z[:] = draw_reward # Penalty for draws
                    
                    all_play_data.append(list(zip(states, mcts_probs, winner_z)))
                    
                    if log_game and game_idx == 0:
                        game_log_data = {
                            "winner": int(winner_val),
                            "moves": game_moves
                        }
                    break
    
        return all_play_data, worker_game_stats, game_log_data
    except Exception as e:
        print(f"CRITICAL WORKER ERROR: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results to avoid crashing main loop hard, but printed error helps debug
        return [], {1: 0, -1: 0, 0: 0}, None
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

                # Increased Alpha from 0.6 to 1.2 to break passive play
                # This forces the agent to explore 'unsafe' attacks
                noise_alpha = 1.2  
                noise = np.random.dirichlet(noise_alpha * np.ones(len(probs)))
                
                # Standard epsilon is 0.25, but you can bump to 0.3 for more exploration
                epsilon = 0.3 
                
                # Mix valid MCTS probs with Noise
                p_with_noise = (1 - epsilon) * probs + epsilon * noise
                
                move = np.random.choice(acts, p=p_with_noise)
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
    def __init__(self, init_model=None, draw_reward=-0.2):
        self.board_size = 7
        self.learn_rate = 2e-3
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 256
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 40 # Increased from 20 to use 8 workers (40/5=8)
        self.num_games_per_worker = 5 # Increased from 1
        self.epochs = 5
        self.check_freq = 50
        self.game_batch_num = 1500
        self.episode_len = 0
        self.draw_reward = draw_reward
        
        self.env = CublinoContraEnv()
        self.eval_env = CublinoContraEnv()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.policy_value_net = PolicyValueNet(board_size=self.board_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), weight_decay=1e-4)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.game_batch_num, eta_min=1e-5)

        if init_model:
            import numpy
            torch.serialization.add_safe_globals([numpy.ndarray, numpy._core.multiarray._reconstruct])
            checkpoint = torch.load(init_model, map_location=self.device, weights_only=False)
            self.policy_value_net.load_state_dict(checkpoint['policy_value_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'data_buffer' in checkpoint:
                self.data_buffer = deque(checkpoint['data_buffer'], maxlen=self.buffer_size)
            if 'episode_len' in checkpoint:
                self.episode_len = checkpoint['episode_len']
            print(f"Loaded model, optimizer, scheduler, data_buffer, and episode_len from {init_model}")

        # Persistent ProcessPoolExecutor
        self.max_workers = 8 # Explicitly set to 8 workers
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        print(f"Initialized ProcessPoolExecutor with {self.max_workers} workers")

    def policy_value_fn(self, env):
        legal_actions = env.get_legal_actions()
        obs = env._get_obs()  # Get stacked observation (7, 7, 12)
        obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(obs_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        return zip(legal_actions, act_probs[legal_actions]), value.item()
    
    def get_policy_value_fn(self, policy_value_net):
        def policy_value_fn(env):
            legal_actions = env.get_legal_actions()
            obs = env._get_obs()  # Get stacked observation (7, 7, 12)
            obs_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_act_probs, value = policy_value_net(obs_tensor)
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
                batch_time = time.time() - start_time
                workers_used = (self.play_batch_size + self.num_games_per_worker - 1) // self.num_games_per_worker
                active_workers = min(workers_used, self.max_workers)
                avg_game_time = (batch_time * active_workers) / self.collected_games if self.collected_games > 0 else 0.0
                print(f"Batch i:{i+1}, Episode Len:{self.episode_len:.2f}, Time:{batch_time:.2f}s, AvgGameTime:{avg_game_time:.2f}s, P1 Wins:{self.batch_game_stats[1]}, P2 Wins:{self.batch_game_stats[-1]}, Draws:{self.batch_game_stats[0]}")
                
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    self.scheduler.step()
                    print(f"Loss: {loss:.4f}, Entropy: {entropy:.4f}")
                    
                if (i+1) % self.check_freq == 0:
                    print("Evaluating...")
                    win_cnt = self.evaluate_policy(n_games=10)
                    print(f"Current Policy Wins: {win_cnt[1]}")
                    print(f"Best Policy Wins: {win_cnt[-1]}")
                    print(f"Draws: {win_cnt[0]}")
                    
                    # Always update best_policy_net to keep data generation fresh
                    print("Updating self-play model to latest version...")
                    self.best_policy_net.load_state_dict(self.policy_value_net.state_dict())
                    
                    # Save checkpoint regardless of win rate
                    torch.save({
                        'policy_value_net': self.policy_value_net.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'data_buffer': list(self.data_buffer),
                        'episode_len': self.episode_len,
                    }, f'current_policy_cublino_{i+1}.pth')
                            
        except KeyboardInterrupt:
            print("Saving checkpoint...")
            torch.save({
                'policy_value_net': self.policy_value_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'data_buffer': list(self.data_buffer), # Convert deque to list for saving
                'episode_len': self.episode_len,
            }, 'checkpoint_cublino.pth')
        finally:
            print("Shutting down executor...")
            self.executor.shutdown()

    def collect_selfplay_data(self, n_games=1):
        model_path = "temp_model_cublino.pt"
        # Save as TorchScript for C++ MCTS
        script_model = torch.jit.script(self.best_policy_net)
        script_model.save(model_path)
        # time.sleep(0.1) # Small safety buffer - Removed as we are reusing workers, file lock might be an issue but usually ok for reading
        
        device_str = str(self.device)
        num_workers = (n_games + self.num_games_per_worker - 1) // self.num_games_per_worker
        
        self.total_episode_len = 0
        self.collected_games = 0
        self.batch_game_stats = {1: 0, -1: 0, 0: 0} # {P1 wins, P2 wins, Draws}

        # Check for log request
        log_game = False
        if os.path.exists("log_game_request"):
            print("Game log requested!")
            log_game = True
            try:
                os.remove("log_game_request")
            except:
                pass

        # Use persistent executor
        futures = []
        for i in range(num_workers):
            # Only ask the first worker to log a game if requested
            do_log = log_game and (i == 0)
            futures.append(self.executor.submit(run_self_play_worker, model_path, self.c_puct, self.n_playout, device_str, self.temp, self.num_games_per_worker, self.draw_reward, do_log))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                list_of_games, worker_game_stats, game_log_data = future.result()
                for game_steps in list_of_games:
                    self.total_episode_len += len(game_steps)
                    self.collected_games += 1
                    self.data_buffer.extend(game_steps)
                
                # Aggregate worker stats
                for player, count in worker_game_stats.items():
                    self.batch_game_stats[player] += count
                
                # Save game log if returned
                if game_log_data is not None:
                    import json
                    timestamp = int(time.time())
                    filename = f"game_log_{timestamp}.json"
                    with open(filename, 'w') as f:
                        json.dump(game_log_data, f)
                    print(f"Game logged to {filename}")

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
    parser.add_argument("--draw_reward", type=float, default=-0.0, help="Reward for a draw (default: -0.0)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
        
    training = TrainPipeline(init_model=args.resume, draw_reward=args.draw_reward)
    if args.dry_run:
        training.game_batch_num = 2 # Run 2 batches to verify reuse
        training.play_batch_size = 1
        training.num_games_per_worker = 1
        training.n_playout = 10
        training.epochs = 1
        
    training.run()
