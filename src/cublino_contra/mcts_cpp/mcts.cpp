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
#include <array>
#include <functional>

// --- Game Logic ---

// Helper struct to avoid heap allocation for neighbors
struct Neighbors {
    std::pair<int, int> pts[4];
    int count = 0;
    
    void add(int r, int c) {
        pts[count].first = r;
        pts[count].second = c;
        count++;
    }
};

class CublinoState {
public:
    std::array<std::array<std::array<int, 3>, 7>, 7> board;
    int current_player;
    int board_size = 7;
    std::map<size_t, int> history; // Optimized history using hash

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
        history.clear();

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
        
        record_history();
    }
    
    size_t compute_hash() const {
        size_t seed = 0;
        // Boost-like hash combine
        for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < 7; ++j) {
                // Combine 3 values. Using manual shifts/xors is faster than std::hash calls for small ints
                // Pack 3 ints into one for fewer combines? 
                // Values are small: P(-1,0,1)+1 -> 0..2 (2 bits), Top(0-6)->3 bits, South(0-6)->3 bits. 
                // Can pack into 8 bits.
                int p = board[i][j][0] + 1; // 0, 1, 2
                int t = board[i][j][1];
                int s = board[i][j][2];
                int val = (p << 6) | (t << 3) | s;
                
                seed ^= val + 0x9e3779b9 + (seed<<6) + (seed>>2);
            }
        }
        seed ^= (current_player + 1) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
    
    void record_history() {
        size_t h = compute_hash();
        history[h]++;
    }

    CublinoState copy() const {
        CublinoState new_state;
        new_state.board = board;
        new_state.current_player = current_player;
        new_state.history = history;
        return new_state;
    }

    // Optimized step: returns {reward, terminated} directly
    // Modifies state in-place
    std::pair<int, bool> step(int action) {
        int direction = action % 4;
        int square_idx = action / 4;
        int row = square_idx / 7;
        int col = square_idx % 7;

        // Basic validation (assumed checked by get_legal_actions for speed in MCTS)
        // Leaving checks for safety but they could be removed for pure speed if guaranteed legal
        if (row < 0 || row >= 7 || col < 0 || col >= 7) return {-10, true};
        if (board[row][col][0] != current_player) return {-10, true};

        int dr = 0, dc = 0;
        if (direction == 0) dr = 1;      // North
        else if (direction == 1) dc = 1; // East
        else if (direction == 2) dr = -1;// South
        else if (direction == 3) dc = -1;// West

        int target_row = row + dr;
        int target_col = col + dc;

        if (target_row < 0 || target_row >= 7 || target_col < 0 || target_col >= 7) return {-10, true};
        if (board[target_row][target_col][0] != 0) return {-10, true};

        // Perform Move
        int top = board[row][col][1];
        int south = board[row][col][2];
        
        // Inline rotate_die logic or call helper
        // Using helper is cleaner, let's trust compiler to inline
        auto [new_top, new_south] = rotate_die_fast(top, south, direction);

        // Update Board
        board[row][col][0] = 0;
        board[row][col][1] = 0;
        board[row][col][2] = 0;

        board[target_row][target_col][0] = current_player;
        board[target_row][target_col][1] = new_top;
        board[target_row][target_col][2] = new_south;

        // Check Win
        if (current_player == 1 && target_row == 6) return {1, true};
        if (current_player == -1 && target_row == 0) return {1, true};

        // Resolve Battles
        resolve_battles(target_row, target_col);

        // Switch Turn
        current_player *= -1;
        
        // 3-fold Repetition Check
        size_t h = compute_hash();
        history[h]++;
        
        if (history[h] >= 3) {
            return {0, true}; // Draw
        }

        return {0, false};
    }
    
    // Optimized die rotation using lookup table approach or simplified logic
    std::pair<int, int> rotate_die_fast(int top, int south, int direction) {
        int north = 7 - south;
        int bottom = 7 - top;
        int east = 0;
        
        // Fast east calculation table or logic
        // Top 1 -> East depends on South {2,3,4,5} -> {3,5,4,2} ? 
        // This is complex to inline perfectly without table. 
        // Let's use the logic from before but slightly cleaner.
        // Actually, we calculate 'east' only for W/E moves.
        
        if (direction == 0) return {south, bottom}; // North
        if (direction == 2) return {north, top};    // South
        
        // Need East
        // Vector map: 1(Z+), 6(Z-), 2(Y-), 5(Y+), 3(X+), 4(X-)
        // East = Top x South
        // Just implement the table for 24 orientations? No, too big.
        // Logic from before is O(1).
        
        // ... (reuse logic from before, it's fast enough) ...
        // Cross product: top x south = east
        // Map 1->(0,0,1) etc
        int vt0=0, vt1=0, vt2=0;
        if(top==1) vt2=1; else if(top==6) vt2=-1; else if(top==2) vt1=-1; 
        else if(top==5) vt1=1; else if(top==3) vt0=1; else if(top==4) vt0=-1;

        int vs0=0, vs1=0, vs2=0;
        if(south==1) vs2=1; else if(south==6) vs2=-1; else if(south==2) vs1=-1; 
        else if(south==5) vs1=1; else if(south==3) vs0=1; else if(south==4) vs0=-1;

        int ve0 = vt1*vs2 - vt2*vs1;
        int ve1 = vt2*vs0 - vt0*vs2;
        int ve2 = vt0*vs1 - vt1*vs0;

        if (ve0==1) east=3; else if (ve0==-1) east=4;
        else if (ve1==1) east=5; else if (ve1==-1) east=2;
        else if (ve2==1) east=1; else if (ve2==-1) east=6;
        
        int west = 7 - east;
        
        if (direction == 1) return {west, south};   // East
        if (direction == 3) return {east, south};   // West
        return {top, south};
    }

    void resolve_battles(int r, int c) {
        int opponent = -current_player;
        Neighbors neighbors = get_neighbors_fast(r, c);
        
        // Use stack array for contested dice to avoid vector allocation
        // Max 4 neighbors
        std::pair<int, int> contested[4];
        int contested_count = 0;

        for (int i = 0; i < neighbors.count; ++i) {
            int nr = neighbors.pts[i].first;
            int nc = neighbors.pts[i].second;
            
            if (board[nr][nc][0] == opponent) {
                Neighbors opp_neighbors = get_neighbors_fast(nr, nc);
                int friendly_count = 0;
                for (int j = 0; j < opp_neighbors.count; ++j) {
                    int onr = opp_neighbors.pts[j].first;
                    int onc = opp_neighbors.pts[j].second;
                    if (board[onr][onc][0] == current_player) {
                        friendly_count++;
                    }
                }
                if (friendly_count >= 2) {
                    contested[contested_count++] = {nr, nc};
                }
            }
        }

        std::pair<int, int> dice_to_remove[4];
        int remove_count = 0;
        
        for (int i = 0; i < contested_count; ++i) {
            int dr = contested[i].first;
            int dc = contested[i].second;
            
            int defender_die_val = board[dr][dc][1];
            Neighbors defender_neighbors = get_neighbors_fast(dr, dc);
            
            int defender_friendly_sum = 0;
            int attacker_sum = 0;

            for (int j = 0; j < defender_neighbors.count; ++j) {
                int nr = defender_neighbors.pts[j].first;
                int nc = defender_neighbors.pts[j].second;
                
                int p = board[nr][nc][0];
                int val = board[nr][nc][1];
                if (p == opponent) defender_friendly_sum += val;
                else if (p == current_player) attacker_sum += val;
            }

            int defender_total = defender_die_val + defender_friendly_sum;
            if (defender_total < attacker_sum) {
                dice_to_remove[remove_count++] = {dr, dc};
            }
        }

        for (int i = 0; i < remove_count; ++i) {
            board[dice_to_remove[i].first][dice_to_remove[i].second][0] = 0;
            board[dice_to_remove[i].first][dice_to_remove[i].second][1] = 0;
            board[dice_to_remove[i].first][dice_to_remove[i].second][2] = 0;
        }
    }

    Neighbors get_neighbors_fast(int r, int c) {
        Neighbors nbs;
        if (c+1 < 7) nbs.add(r, c+1);
        if (c-1 >= 0) nbs.add(r, c-1);
        if (r+1 < 7) nbs.add(r+1, c);
        if (r-1 >= 0) nbs.add(r-1, c);
        return nbs;
    }

    torch::Tensor get_obs() const {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs = torch::zeros({1, 3, 7, 7}, options);
        // Can be optimized by direct pointer access if bottleneck
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
        legal.reserve(100); // Reserve plausible amount
        for (int r = 0; r < 7; ++r) {
            for (int c = 0; c < 7; ++c) {
                if (board[r][c][0] == current_player) {
                    // Unroll?
                    if (r+1 < 7 && board[r+1][c][0] == 0) legal.push_back((r * 7 + c) * 4 + 0);
                    if (c+1 < 7 && board[r][c+1][0] == 0) legal.push_back((r * 7 + c) * 4 + 1);
                    if (r-1 >= 0 && board[r-1][c][0] == 0) legal.push_back((r * 7 + c) * 4 + 2);
                    if (c-1 >= 0 && board[r][c-1][0] == 0) legal.push_back((r * 7 + c) * 4 + 3);
                }
            }
        }
        return legal;
    }
    
    void set_state_from_python(py::array_t<int> board_np, int player) {
        auto r = board_np.unchecked<3>();
        for (int i = 0; i < 7; ++i)
            for (int j = 0; j < 7; ++j)
                for (int k = 0; k < 3; ++k)
                    board[i][j][k] = r(i, j, k);
        current_player = player;
        history.clear();
        record_history();
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
        // Use loop to advance state to avoid recursive copy if possible, 
        // but here we copy once and mutate.
        CublinoState current_state = state.copy();

        while (!node->is_leaf()) {
            auto selection = node->select(c_puct);
            int action = selection.first;
            node = selection.second;
            
            // Optimized step call: returns {reward, terminated}
            std::pair<int, bool> step_res = current_state.step(action);
            bool terminated = step_res.second;
            
            if (terminated) {
                int reward = step_res.first;
                if (reward == 0) {
                     node->update_recursive(0.0);
                } else {
                     node->update_recursive(-1.0);
                }
                return;
            }
        }

        // Evaluation
        torch::Tensor obs = current_state.get_obs();
        obs = obs.to(device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(obs);
        
        auto output = module.forward(inputs).toTuple();
        
        // Tensor access optimization? CPU transfer is the bottleneck usually.
        torch::Tensor policy_logits = output->elements()[0].toTensor().cpu();
        torch::Tensor value = output->elements()[1].toTensor().cpu();
        
        torch::Tensor policy_probs = torch::softmax(policy_logits, 1).squeeze(0);
        double leaf_value = value.item<double>();

        std::vector<int> legal_moves = current_state.get_legal_actions();
        std::vector<std::pair<int, double>> action_priors;
        action_priors.reserve(legal_moves.size());
        
        double sum_probs = 0;
        // Direct data access to avoid item<double>() overhead loop? 
        // float* probs_ptr = policy_probs.data_ptr<float>(); 
        // Assuming float model.
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

        node->expand(action_priors);
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
        acts.reserve(root->children.size());
        visits.reserve(root->children.size());

        for (const auto& child : root->children) {
            acts.push_back(child.first);
            visits.push_back(child.second->n_visits);
        }

        std::vector<double> logits;
        logits.reserve(visits.size());
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

PYBIND11_MODULE(_mcts_cpp, m) {
    py::class_<CublinoState>(m, "CublinoState")
        .def(py::init<>())
        .def("reset", &CublinoState::reset)
        .def("step", [](CublinoState& s, int action) {
             // Return just reward and terminated for python? 
             // Existing python code expects {board, reward, terminated} tuple.
             // We must maintain interface compatibility if Python env uses it.
             // But here this is C++ MCTS internal state wrapper.
             // Python calls `_convert_env_to_cpp_state` which uses `set_state_from_python`.
             // Python calls `step` on `CublinoState`? No, Python calls `mcts.get_move_probs`.
             // Does any python test call `CublinoState.step` directly?
             // `tests/test_mcts_cpp.py` might.
             // To be safe, let's keep the binding returning the tuple, but construct it here.
             auto res = s.step(action);
             // Return a copy of board to satisfy legacy interface if needed, 
             // but `step` no longer returns board. We can access s.board.
             return std::make_tuple(s.board, res.first, res.second);
        })
        .def("set_state_from_python", &CublinoState::set_state_from_python)
        .def("get_obs", &CublinoState::get_obs)
        .def_readwrite("current_player", &CublinoState::current_player);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<std::string, double, int, std::string>())
        .def("get_move_probs", &MCTS::get_move_probs)
        .def("update_with_move", &MCTS::update_with_move);
}