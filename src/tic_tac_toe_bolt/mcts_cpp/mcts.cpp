#include <torch/extension.h>
#include <pybind11/stl.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <memory>
#include <algorithm>
#include <random>

// --- Game Logic ---

class TicTacToeState {
public:
    std::array<std::array<int, 3>, 3> board;
    int current_player;
    std::map<int, std::vector<std::pair<int, int>>> player_moves;

    TicTacToeState() {
        reset();
    }

    void reset() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                board[i][j] = 0;
        current_player = 1;
        player_moves[1] = {};
        player_moves[-1] = {};
    }

    TicTacToeState copy() const {
        TicTacToeState new_state;
        new_state.board = board;
        new_state.current_player = current_player;
        new_state.player_moves = player_moves;
        return new_state;
    }

    std::tuple<std::array<std::array<int, 3>, 3>, int, bool> step(int action) {
        int row = action / 3;
        int col = action % 3;

        if (board[row][col] != 0) {
            // Invalid move, return penalty and don't change state
            // In Python we returned -10 reward. Here we just return current state and indicate invalid?
            // MCTS should filter these. But if it happens:
            return {board, -10, false}; 
        }

        // Infinite mechanic
        if (player_moves[current_player].size() == 3) {
            auto old_pos = player_moves[current_player][0];
            player_moves[current_player].erase(player_moves[current_player].begin());
            board[old_pos.first][old_pos.second] = 0;
        }

        board[row][col] = current_player;
        player_moves[current_player].push_back({row, col});

        int reward = 0;
        bool terminated = false;

        if (check_win(current_player)) {
            reward = 1;
            terminated = true;
        }

        current_player *= -1;
        return {board, reward, terminated};
    }

    bool check_win(int player) const {
        // Rows and Cols
        for (int i = 0; i < 3; ++i) {
            if (board[i][0] == player && board[i][1] == player && board[i][2] == player) return true;
            if (board[0][i] == player && board[1][i] == player && board[2][i] == player) return true;
        }
        // Diagonals
        if (board[0][0] == player && board[1][1] == player && board[2][2] == player) return true;
        if (board[0][2] == player && board[1][1] == player && board[2][0] == player) return true;
        return false;
    }

    torch::Tensor get_obs() const {
        // Convert board to tensor for NN input
        // Input shape: (1, 3, 3, 3) -> (Batch, Channels, H, W)
        // Channels: 
        // 0: Current player's pieces with age encoding (0.33=oldest, 0.66=mid, 1.0=newest)
        // 1: Opponent's pieces with age encoding
        // 2: All 1.0 (indicating it's the current player's turn to move)
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs = torch::zeros({1, 3, 3, 3}, options);
        
        // Helper to fill channel based on moves
        auto fill_channel = [&](int channel, int player) {
            if (player_moves.count(player)) {
                const auto& moves = player_moves.at(player);
                int n = moves.size();
                for (int i = 0; i < n; ++i) {
                    float val = 0.0f;
                    // If 1 move: 1.0
                    // If 2 moves: 0.66, 1.0
                    // If 3 moves: 0.33, 0.66, 1.0
                    // General formula: (i + 1 + (3 - n)) / 3.0 ? No.
                    // Just simple mapping based on index relative to end?
                    // Oldest is at index 0. Newest is at index n-1.
                    // Let's stick to the requested:
                    // 3 moves: [0.33, 0.66, 1.0]
                    // 2 moves: [0.66, 1.0] or [0.33, 0.66]? 
                    // Usually "Newest=1.0" is the anchor.
                    // So: index n-1 -> 1.0. index n-2 -> 0.66. index n-3 -> 0.33.
                    
                    if (i == n - 1) val = 1.0f;
                    else if (i == n - 2) val = 0.66f;
                    else if (i == n - 3) val = 0.33f;
                    
                    obs[0][channel][moves[i].first][moves[i].second] = val;
                }
            }
        };

        fill_channel(0, current_player); // Self
        fill_channel(1, -current_player); // Opponent
        
        // Channel 2: Turn indicator (always 1.0 for the player whose turn it is to move)
        obs[0][2] = 1.0; 
        
        return obs;
    }
    
    std::vector<int> get_legal_actions() const {
        std::vector<int> legal;
        for(int i=0; i<9; ++i) {
            if (board[i/3][i%3] == 0) {
                legal.push_back(i);
            }
        }
        return legal;
    }
    void set_state(const std::vector<std::vector<int>>& new_board, int player, const std::map<int, std::vector<std::pair<int, int>>>& moves) {
        for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
                board[i][j] = new_board[i][j];
            }
        }
        current_player = player;
        player_moves = moves;
    }
};

// --- MCTS Logic ---

class TreeNode {
public:
    TreeNode* parent;
    std::map<int, std::unique_ptr<TreeNode>> children;
    int n_visits;
    double Q;
    double u;
    double P;

    TreeNode(TreeNode* parent, double prior_p) 
        : parent(parent), n_visits(0), Q(0), u(0), P(prior_p) {}

    void expand(const std::vector<std::pair<int, double>>& action_priors) {
        for (const auto& ap : action_priors) {
            if (children.find(ap.first) == children.end()) {
                children[ap.first] = std::make_unique<TreeNode>(this, ap.second);
            }
        }
    }

    std::pair<int, TreeNode*> select(double c_puct) {
        double max_val = -std::numeric_limits<double>::infinity();
        int best_act = -1;
        TreeNode* best_node = nullptr;

        for (const auto& item : children) {
            double u_val = c_puct * item.second->P * std::sqrt(n_visits) / (1 + item.second->n_visits);
            double val = item.second->Q + u_val;
            if (val > max_val) {
                max_val = val;
                best_act = item.first;
                best_node = item.second.get();
            }
        }
        return {best_act, best_node};
    }

    void update(double leaf_value) {
        n_visits++;
        Q += (leaf_value - Q) / n_visits;
    }

    void update_recursive(double leaf_value) {
        if (parent) {
            parent->update_recursive(-leaf_value);
        }
        update(leaf_value);
    }

    bool is_leaf() const {
        return children.empty();
    }
};

class MCTS {
    std::unique_ptr<TreeNode> root;
    torch::jit::script::Module module;
    double c_puct;
    int n_playout;
    torch::Device device;

public:
    MCTS(std::string model_path, double c_puct, int n_playout, std::string device_str) 
        : c_puct(c_puct), n_playout(n_playout), device(device_str) {
        try {
            module = torch::jit::load(model_path, device);
            module.eval(); // Set to evaluation mode
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
            throw e;
        }
        root = std::make_unique<TreeNode>(nullptr, 1.0);
    }

    void playout(TicTacToeState& state) {
        TreeNode* node = root.get();
        TicTacToeState current_state = state.copy(); // Work on a copy

        // 1. Selection
        while (!node->is_leaf()) {
            auto selection = node->select(c_puct);
            int action = selection.first;
            node = selection.second;
            
            auto step_res = current_state.step(action);
            bool terminated = std::get<2>(step_res);
            
            if (terminated) {
                // Game over, current player won.
                // Value for next player is -1.
                // Value for current player (Parent of node) is +1.
                node->update_recursive(1.0);
                return;
            }
        }

        // 2. Evaluation
        torch::Tensor obs = current_state.get_obs();
        obs = obs.to(device); // Move to correct device
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs);
        
        auto output = module.forward(inputs).toTuple();
        torch::Tensor policy_logits = output->elements()[0].toTensor().cpu(); // Move back to CPU for processing
        torch::Tensor value = output->elements()[1].toTensor().cpu(); // Move back to CPU for processing
        
        // Softmax policy
        torch::Tensor policy_probs = torch::softmax(policy_logits, 1).squeeze(0); // Shape (9)
        double leaf_value = value.item<double>();

        // Mask illegal moves
        std::vector<int> legal_moves = current_state.get_legal_actions();
        std::vector<std::pair<int, double>> action_priors;
        
        double sum_probs = 0;
        for (int move : legal_moves) {
            double prob = policy_probs[move].item<double>();
            action_priors.push_back({move, prob});
            sum_probs += prob;
        }
        
        // Renormalize
        if (sum_probs > 0) {
            for (auto& ap : action_priors) {
                ap.second /= sum_probs;
            }
        } else {
            // If all legal moves have 0 prob (shouldn't happen with softmax), uniform prior
            for (auto& ap : action_priors) {
                ap.second = 1.0 / legal_moves.size();
            }
        }

        // 3. Expansion
        node->expand(action_priors);

        // 4. Backup
        node->update_recursive(-leaf_value);
    }

    std::pair<std::vector<int>, std::vector<double>> get_move_probs(const TicTacToeState& state, double temp) {
        for (int i = 0; i < n_playout; ++i) {
            TicTacToeState state_copy = state.copy();
            playout(state_copy);
        }

        std::vector<int> acts;
        std::vector<double> probs;
        std::vector<int> visits;

        for (const auto& child : root->children) {
            acts.push_back(child.first);
            visits.push_back(child.second->n_visits);
        }

        // Softmax based on visits
        // probs = softmax(1/temp * log(visits + 1e-10))
        // For temp -> 0, it's argmax.
        
        // Simplified calculation for now (standard visit count ratio if temp=1)
        // Implementing the softmax logic:
        std::vector<double> logits;
        double max_logit = -1e9;
        
        for (int v : visits) {
            double logit = (1.0 / temp) * std::log(v + 1e-10);
            logits.push_back(logit);
            if (logit > max_logit) max_logit = logit;
        }
        
        double sum_exp = 0;
        for (double l : logits) {
            sum_exp += std::exp(l - max_logit);
        }
        
        for (double l : logits) {
            probs.push_back(std::exp(l - max_logit) / sum_exp);
        }

        return {acts, probs};
    }

    void update_with_move(int last_move) {
        if (root->children.count(last_move)) {
            std::unique_ptr<TreeNode> new_root = std::move(root->children[last_move]);
            new_root->parent = nullptr;
            root = std::move(new_root);
        } else {
            root = std::make_unique<TreeNode>(nullptr, 1.0);
        }
    }
};


// --- MCTS Logic ---
// ... (TreeNode and MCTS classes remain unchanged)

// --- Bindings ---

PYBIND11_MODULE(_mcts_cpp, m) {
    py::class_<TicTacToeState>(m, "TicTacToeState")
        .def(py::init<>())
        .def("reset", &TicTacToeState::reset)
        .def("step", [](TicTacToeState& s, int action) {
            auto res = s.step(action);
            return std::make_tuple(std::get<1>(res), std::get<2>(res));
        })
        .def("set_state", &TicTacToeState::set_state)
        .def("get_obs", &TicTacToeState::get_obs)
        .def_readwrite("current_player", &TicTacToeState::current_player);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<std::string, double, int, std::string>())
        .def("get_move_probs", &MCTS::get_move_probs)
        .def("update_with_move", &MCTS::update_with_move);
}
