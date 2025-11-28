import time
import torch
import gymnasium as gym
import numpy as np
import sys
import os

# Add project root to sys.path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.cublino_contra.model import PolicyValueNet
from src.cublino_contra.mcts import MCTS, MCTS_CPP
import src.cublino_contra # Register env

def benchmark_mcts():
    # Setup
    env = gym.make("CublinoContra-v0")
    env.reset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PolicyValueNet(board_size=7).to(device)
    model.eval()
    
    # Save model for C++ MCTS
    model_path = "temp_benchmark_cublino.pt"
    script_model = torch.jit.script(model)
    script_model.save(model_path)
    
    n_playout = 200 # Lower than TTT because Cublino is heavier
    c_puct = 5
    
    # Python MCTS policy function
    def policy_value_fn(env):
        legal_actions = env.unwrapped.get_legal_actions()
        board = env.unwrapped.board
        # Board is (7, 7, 3), model expects (N, 3, 7, 7)
        board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            log_act_probs, value = model(board_tensor)
            act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
            
        return zip(legal_actions, act_probs[legal_actions]), value.item()

    mcts_py = MCTS(policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    
    try:
        mcts_cpp = MCTS_CPP(model_path, c_puct=c_puct, n_playout=n_playout, device=device)
    except Exception as e:
        print(f"Failed to load C++ MCTS: {e}")
        return

    num_iterations = 5 # Reduced iterations for Cublino as it's slower
    print(f"\nRunning benchmark with {n_playout} playouts per move, over {num_iterations} iterations...")

    # Warmup
    print("Warming up...")
    mcts_py.get_move_probs(env, temp=1.0)
    mcts_cpp.get_move_probs(env, temp=1.0)
    
    # Benchmark Python
    print("Benchmarking Python MCTS...")
    start_time = time.time()
    for i in range(num_iterations):
        print(f"  Iter {i+1}/{num_iterations}...")
        mcts_py.get_move_probs(env, temp=1.0)
        # Reset tree for fair comparison per call? 
        # Usually we want to measure 'per decision' time.
        # MCTS class in mcts.py keeps the tree.
        mcts_py.update_with_move(-1) # Reset root
        
    py_time = time.time() - start_time
    
    # Benchmark C++
    print("Benchmarking C++ MCTS...")
    start_time = time.time()
    for i in range(num_iterations):
        print(f"  Iter {i+1}/{num_iterations}...")
        mcts_cpp.get_move_probs(env, temp=1.0)
        mcts_cpp.update_with_move(-1) # Reset root
        
    cpp_time = time.time() - start_time
    
    print("\nResults:")
    print(f"Python MCTS Total Time: {py_time:.4f}s")
    print(f"Python MCTS Avg Time:   {py_time/num_iterations:.4f}s")
    print(f"C++ MCTS Total Time:    {cpp_time:.4f}s")
    print(f"C++ MCTS Avg Time:      {cpp_time/num_iterations:.4f}s")
    print(f"Speedup:                {py_time/cpp_time:.2f}x")

    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == "__main__":
    benchmark_mcts()
