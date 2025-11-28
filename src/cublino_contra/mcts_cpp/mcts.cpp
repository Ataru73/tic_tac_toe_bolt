#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <memory>
#include <algorithm>
#include <random>
#include <tuple>

// --- Game Logic ---

class CublinoState {
public:
    // Board: 7x7x3 (Player, Top, South)
    // Flattened to 7x7 array of structs or just 3D array?
    // Let's use std::array<std::array<std::array<int, 3>, 7>, 7>
    std::array<std::array<std::array<int, 3>, 7>, 7> board;
    int current_player;
    int board_size = 7;

    CublinoState() {
        reset();
    }

    void reset() {
        // Clear board
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int k = 0; k < 3; ++k)
                    board[i][j][k] = 0;

        current_player = 1;

        // Setup P1 (Row 0)
        for (int col = 0; col < 7; ++col) {
            board[0][col][0] = 1;
            board[0][col][1] = 6;
            board[0][col][2] = 3;
        }

        // Setup P2 (Row 6)
        for (int col = 0; col < 7; ++col) {
            board[6][col][0] = -1;
            board[6][col][1] = 6;
            board[6][col][2] = 4;
        }
    }

    CublinoState copy() const {
        CublinoState new_state;
        new_state.board = board;
        new_state.current_player = current_player;
        return new_state;
    }

    // Returns: {board, reward, terminated}
    std::tuple<std::array<std::array<std::array<int, 3>, 7>, 7>, int, bool> step(int action) {
        int direction = action % 4;
        int square_idx = action / 4;
        int row = square_idx / 7;
        int col = square_idx % 7;

        // Validate source (basic check, though MCTS should filter)
        if (row < 0 || row >= 7 || col < 0 || col >= 7) return {board, -10, true};
        if (board[row][col][0] != current_player) return {board, -10, true};

        int dr = 0, dc = 0;
        if (direction == 0) dr = 1;      // North
        else if (direction == 1) dc = 1; // East
        else if (direction == 2) dr = -1;// South
        else if (direction == 3) dc = -1;// West

        int target_row = row + dr;
        int target_col = col + dc;

        // Validate target
        if (target_row < 0 || target_row >= 7 || target_col < 0 || target_col >= 7) return {board, -10, true};
        if (board[target_row][target_col][0] != 0) return {board, -10, true};

        // Perform Move
        int top = board[row][col][1];
        int south = board[row][col][2];
        
        auto [new_top, new_south] = rotate_die(top, south, direction);

        // Update Board
        board[row][col][0] = 0;
        board[row][col][1] = 0;
        board[row][col][2] = 0;

        board[target_row][target_col][0] = current_player;
        board[target_row][target_col][1] = new_top;
        board[target_row][target_col][2] = new_south;

        // Check Win
        if (current_player == 1 && target_row == 6) return {board, 1, true};
        if (current_player == -1 && target_row == 0) return {board, 1, true};

        // Resolve Battles
        resolve_battles(target_row, target_col);

        // Switch Turn
        current_player *= -1;

        return {board, 0, false};
    }

    std::pair<int, int> rotate_die(int top, int south, int direction) {
        // 0:N, 1:E, 2:S, 3:W
        // Hardcoded logic from env.py
        int north = 7 - south;
        int bottom = 7 - top;
        
        // Calculate East
        // Vector mapping
        // 1: [0,0,1], 6: [0,0,-1]
        // 2: [0,-1,0] (S), 5: [0,1,0] (N)
        // 3: [1,0,0] (E), 4: [-1,0,0] (W)
        
        // Cross product logic is complex to hardcode without vector lib.
        // Let's implement the vector map logic simply.
        int v_top[3] = {0, 0, 0};
        if (top == 1) v_top[2] = 1;
        else if (top == 6) v_top[2] = -1;
        else if (top == 2) v_top[1] = -1;
        else if (top == 5) v_top[1] = 1;
        else if (top == 3) v_top[0] = 1;
        else if (top == 4) v_top[0] = -1;

        int v_south[3] = {0, 0, 0};
        if (south == 1) v_south[2] = 1;
        else if (south == 6) v_south[2] = -1;
        else if (south == 2) v_south[1] = -1;
        else if (south == 5) v_south[1] = 1;
        else if (south == 3) v_south[0] = 1;
        else if (south == 4) v_south[0] = -1;

        // Cross product: top x south = east
        int v_east[3];
        v_east[0] = v_top[1]*v_south[2] - v_top[2]*v_south[1];
        v_east[1] = v_top[2]*v_south[0] - v_top[0]*v_south[2];
        v_east[2] = v_top[0]*v_south[1] - v_top[1]*v_south[0];

        int east = 0;
        if (v_east[0]==1) east=3;
        else if (v_east[0]==-1) east=4;
        else if (v_east[1]==1) east=5;
        else if (v_east[1]==-1) east=2;
        else if (v_east[2]==1) east=1;
        else if (v_east[2]==-1) east=6;
        
        int west = 7 - east;

        if (direction == 0) return {south, bottom}; // North
        if (direction == 2) return {north, top};    // South
        if (direction == 1) return {west, south};   // East
        if (direction == 3) return {east, south};   // West
        return {top, south};
    }

    void resolve_battles(int r, int c) {
        int opponent = -current_player;
        std::vector<std::pair<int, int>> neighbors = get_neighbors(r, c);
        std::vector<std::pair<int, int>> contested_dice;

        for (auto& nb : neighbors) {
            int nr = nb.first;
            int nc = nb.second;
            if (board[nr][nc][0] == opponent) {
                // Check if surrounded by >= 2 friendly
                std::vector<std::pair<int, int>> opp_neighbors = get_neighbors(nr, nc);
                int friendly_count = 0;
                for (auto& onb : opp_neighbors) {
                    if (board[onb.first][onb.second][0] == current_player) {
                        friendly_count++;
                    }
                }
                if (friendly_count >= 2) {
                    contested_dice.push_back({nr, nc});
                }
            }
        }

        std::vector<std::pair<int, int>> dice_to_remove;
        for (auto& defender : contested_dice) {
            int dr = defender.first;
            int dc = defender.second;
            
            int defender_die_val = board[dr][dc][1];
            std::vector<std::pair<int, int>> defender_neighbors = get_neighbors(dr, dc);
            
            int defender_friendly_sum = 0;
            int attacker_sum = 0;

            for (auto& nb : defender_neighbors) {
                int p = board[nb.first][nb.second][0];
                int val = board[nb.first][nb.second][1];
                if (p == opponent) defender_friendly_sum += val;
                else if (p == current_player) attacker_sum += val;
            }

            int defender_total = defender_die_val + defender_friendly_sum;
            if (defender_total < attacker_sum) {
                dice_to_remove.push_back({dr, dc});
            }
        }

        for (auto& d : dice_to_remove) {
            board[d.first][d.second][0] = 0;
            board[d.first][d.second][1] = 0;
            board[d.first][d.second][2] = 0;
        }
    }

    std::vector<std::pair<int, int>> get_neighbors(int r, int c) {
        std::vector<std::pair<int, int>> nbs;
        int dr[] = {0, 0, 1, -1};
        int dc[] = {1, -1, 0, 0};
        for (int i = 0; i < 4; ++i) {
            int nr = r + dr[i];
            int nc = c + dc[i];
            if (nr >= 0 && nr < 7 && nc >= 0 && nc < 7) {
                nbs.push_back({nr, nc});
            }
        }
        return nbs;
    }

    torch::Tensor get_obs() const {
        // (1, 3, 7, 7)
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs = torch::zeros({1, 3, 7, 7}, options);
        
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                obs[0][0][r][c] = (float)board[r][c][0];
                obs[0][1][r][c] = (float)board[r][c][1];
                obs[0][2][r][c] = (float)board[r][c][2];
            }
        }
        return obs;
    }

    std::vector<int> get_legal_actions() const {
        std::vector<int> legal;
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                if (board[r][c][0] == current_player) {
                    for (int d = 0; d < 4; ++d) {
                        int dr = 0, dc = 0;
                        if (d == 0) dr = 1;
                        else if (d == 1) dc = 1;
                        else if (d == 2) dr = -1;
                        else if (d == 3) dc = -1;

                        int tr = r + dr;
                        int tc = c + dc;

                        if (tr >= 0 && tr < 7 && tc >= 0 && tc < 7) {
                            if (board[tr][tc][0] == 0) {
                                legal.push_back((r * 7 + c) * 4 + d);
                            }
                        }
                    }
                }
            }
        }
        return legal;
    }
    
    // For Python interop
    void set_state_from_python(py::array_t<int> board_np, int player) {
        auto r = board_np.unchecked<3>(); // (7, 7, 3)
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int k = 0; k < 3; ++k)
                    board[i][j][k] = r(i, j, k);
        current_player = player;
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
            module.eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
            throw e;
        }
        root = std::make_unique<TreeNode>(nullptr, 1.0);
    }

    void playout(CublinoState& state) {
        TreeNode* node = root.get();
        CublinoState current_state = state.copy();

        // 1. Selection
        while (!node->is_leaf()) {
            auto selection = node->select(c_puct);
            int action = selection.first;
            node = selection.second;
            
            auto step_res = current_state.step(action);
            bool terminated = std::get<2>(step_res);
            
            if (terminated) {
                node->update_recursive(1.0); // Current player won, so previous player (node parent) gets +1? No.
                // If current_state.step returns terminated=true, it means the player who JUST moved won.
                // The node corresponds to the state AFTER the move.
                // Wait.
                // Root = State S0.
                // Select Action A0 -> Node N1 (State S1).
                // If S1 is terminal (P1 won), then N1 value is +1 for P1?
                // MCTS value is usually "value for the player whose turn it is at that node".
                // If S1 is terminal and P1 won, it's P2's turn (but game over).
                // So value for P2 is -1.
                // update_recursive(-leaf_value).
                // If we pass leaf_value = -1 (P2 loses), then parent (P1) gets +1. Correct.
                
                // My step function returns reward=1 if winner.
                // So leaf_value = -1.0 (from perspective of next player).
                node->update_recursive(-(-1.0)); // i.e. +1.0
                return;
            }
        }

        // 2. Evaluation
        torch::Tensor obs = current_state.get_obs();
        obs = obs.to(device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs);
        
        auto output = module.forward(inputs).toTuple();
        torch::Tensor policy_logits = output->elements()[0].toTensor().cpu();
        torch::Tensor value = output->elements()[1].toTensor().cpu();
        
        torch::Tensor policy_probs = torch::softmax(policy_logits, 1).squeeze(0); // (196)
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
        
        if (sum_probs > 0) {
            for (auto& ap : action_priors) ap.second /= sum_probs;
        } else {
            for (auto& ap : action_priors) ap.second = 1.0 / legal_moves.size();
        }

        // 3. Expansion
        node->expand(action_priors);

        // 4. Backup
        node->update_recursive(-leaf_value);
    }

    std::pair<std::vector<int>, std::vector<double>> get_move_probs(const CublinoState& state, double temp) {
        for (int i = 0; i < n_playout; ++i) {
            CublinoState state_copy = state.copy();
            playout(state_copy);
        }

        std::vector<int> acts;
        std::vector<double> probs;
        std::vector<int> visits;

        for (const auto& child : root->children) {
            acts.push_back(child.first);
            visits.push_back(child.second->n_visits);
        }

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

// --- Bindings ---

PYBIND11_MODULE(_mcts_cpp, m) {
    py::class_<CublinoState>(m, "CublinoState")
        .def(py::init<>())
        .def("reset", &CublinoState::reset)
        .def("step", [](CublinoState& s, int action) {
            auto res = s.step(action);
            return std::make_tuple(std::get<1>(res), std::get<2>(res));
        })
        .def("set_state_from_python", &CublinoState::set_state_from_python)
        .def("get_obs", &CublinoState::get_obs)
        .def_readwrite("current_player", &CublinoState::current_player);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<std::string, double, int, std::string>())
        .def("get_move_probs", &MCTS::get_move_probs)
        .def("update_with_move", &MCTS::update_with_move);
}
