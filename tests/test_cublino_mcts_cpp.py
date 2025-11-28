import unittest
import torch
import numpy as np
import os
from src.cublino_contra.mcts import MCTS_CPP
from src.cublino_contra.env import CublinoContraEnv
from src.cublino_contra.model import PolicyValueNet

class TestCublinoMCTSCPP(unittest.TestCase):
    def test_cpp_mcts_integration(self):
        """Test if C++ MCTS can be instantiated and run."""
        # Create a dummy model
        model = PolicyValueNet(board_size=7)
        model_path = "temp_test_model.pt"
        script_model = torch.jit.script(model)
        script_model.save(model_path)
        
        env = CublinoContraEnv()
        
        try:
            mcts = MCTS_CPP(model_path, n_playout=10)
            
            # Run get_move_probs
            acts, probs = mcts.get_move_probs(env)
            
            self.assertTrue(len(acts) > 0)
            self.assertTrue(len(acts) <= 196)
            self.assertAlmostEqual(sum(probs), 1.0, places=5)
            
            # Test update_with_move
            mcts.update_with_move(acts[0])
            
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
