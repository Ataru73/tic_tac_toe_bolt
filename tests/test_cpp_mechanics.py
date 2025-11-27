import unittest
import torch
import numpy as np
from tic_tac_toe_bolt import _mcts_cpp

class TestMCTSCppMechanics(unittest.TestCase):
    def test_infinite_mechanic_cpp(self):
        state = _mcts_cpp.TicTacToeState()
        state.reset()
        
        # P1 moves: (0,0), (0,1), (0,2) -> Indices 0, 1, 2
        # P2 moves: (1,0), (1,1), (1,2) -> Indices 3, 4, 5
        
        # Move 1 (P1): 0
        state.step(0)
        # Move 2 (P2): 3
        state.step(3)
        # Move 3 (P1): 1
        state.step(1)
        # Move 4 (P2): 4
        state.step(4)
        # Move 5 (P1): 2
        state.step(2)
        # Move 6 (P2): 5
        state.step(5)
        
        # Board should be full-ish. P1 has 3 marks (0, 1, 2). P2 has 3 marks (3, 4, 5).
        # Check board
        # P1 is 1, P2 is -1.
        # Check Obs (Tensor)
        obs = state.get_obs()
        # Obs is (1, 3, 3, 3)
        # Channel 0 (P1): (0,0)=0.33, (0,1)=0.66, (0,2)=1.0 (relative to P1 turn?)
        # Wait, whose turn is it?
        # Moves: P1, P2, P1, P2, P1, P2.
        # Next is P1's turn. current_player = 1.
        # So Channel 0 is P1.
        
        c0 = obs[0][0].numpy()
        print("Channel 0 (P1):")
        print(c0)
        
        self.assertAlmostEqual(c0[0,0], 0.33, delta=0.01)
        self.assertAlmostEqual(c0[0,1], 0.66, delta=0.01)
        self.assertAlmostEqual(c0[0,2], 1.0, delta=0.01)
        
        # Now P1 makes 4th move. Should lose oldest (0,0).
        # Move 7 (P1): 8 (2,2)
        state.step(8)
        
        obs_new = state.get_obs()
        # Next is P2 turn. So Channel 0 is P2, Channel 1 is P1.
        c1 = obs_new[0][1].numpy() # P1 from P2's perspective
        print("\nChannel 1 (P1 after 4th move, from P2 perspective):")
        print(c1)
        
        # P1 marks: (0,1), (0,2), (2,2). (0,0) should be gone (0.0).
        # Ages: (0,1) is now oldest -> 0.33. (0,2) -> 0.66. (2,2) -> 1.0.
        self.assertEqual(c1[0,0], 0.0)
        self.assertAlmostEqual(c1[0,1], 0.33, delta=0.01)
        self.assertAlmostEqual(c1[0,2], 0.66, delta=0.01)
        self.assertAlmostEqual(c1[2,2], 1.0, delta=0.01)

if __name__ == "__main__":
    unittest.main()
