import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        # Input: 3 channels (Player 1, Player 2, Empty/Valid) x 3x3 board
        # Actually, let's just use 3 channels:
        # Channel 0: 1 where current player has marks, 0 otherwise
        # Channel 1: 1 where opponent has marks, 0 otherwise
        # Channel 2: 1 where valid moves are (or just all 1s/0s indicating whose turn? Usually turn is implicit or handled by history)
        # Let's stick to:
        # Channel 0: Current player's marks (1 if present, 0 else)
        # Channel 1: Opponent's marks (1 if present, 0 else)
        # Channel 2: All 1s if Player 1's turn, All 0s if Player 2's turn (optional, but helps if we want to encode turn)
        # Or simpler: Just 2 channels for the board state relative to current player.
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc_input_dim = 128 * 3 * 3
        
        # Policy Head
        self.policy_fc = nn.Linear(self.fc_input_dim, 9)
        
        # Value Head
        self.value_fc1 = nn.Linear(self.fc_input_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input shape: (batch_size, channels=3, height=3, width=3)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(-1, self.fc_input_dim)
        
        # Policy
        policy_logits = self.policy_fc(x)
        policy = F.log_softmax(policy_logits, dim=1)
        
        # Value
        value = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
