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
    int step_count = 0;
    int max_steps = 100;
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
        step_count = 0;
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
        new_state.step_count = step_count;
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
        
        // Max Steps Check
        step_count++;
        if (step_count >= max_steps) {
            return {0, true}; // Draw
        }
        
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
        step_count = 0;
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
    double total_value; // Replaces Q, Q = total_value / n_visits
    double P;
    
    // Virtual loss constant
    static constexpr double VIRTUAL_LOSS = 1.0;

    TreeNode(TreeNode* parent, double prior_p) 
        : parent(parent), n_visits(0), total_value(0), P(prior_p) {}

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

        double sqrt_visits = std::sqrt(n_visits + 1e-8); // Avoid sqrt(0)

        for (const auto& item : children) {
            TreeNode* child = item.second.get();
            double Q = 0;
            if (child->n_visits > 0) {
                Q = child->total_value / child->n_visits;
            }
            
            double u_val = c_puct * child->P * sqrt_visits / (1 + child->n_visits);
            double val = Q + u_val;
            
            if (val > max_val) {
                max_val = val;
                best_act = item.first;
                best_node = child;
            }
        }
        return {best_act, best_node};
    }
    
    // Apply virtual loss during selection
    void apply_virtual_loss() {
        n_visits++;
        total_value -= VIRTUAL_LOSS;
    }
    
    // Revert virtual loss and apply real update
    void update(double leaf_value) {
        // n_visits was already incremented in select
        // total_value had VIRTUAL_LOSS subtracted
        // We want total_value_new = total_value_old + leaf_value
        // Current total_value = total_value_old - VIRTUAL_LOSS
        // So:
        total_value += VIRTUAL_LOSS + leaf_value;
    }

    void update_recursive(double leaf_value) {
        update(leaf_value);
        if (parent) {
            parent->update_recursive(-leaf_value);
        }
    }
    
    void update_recursive_no_virtual(double leaf_value) {
        total_value += leaf_value;
        if (parent) {
            parent->update_recursive_no_virtual(-leaf_value);
        }
    }

    // Recursive virtual loss application (downwards? no, select does one step)
    // Actually we need to apply virtual loss to the node we picked
    // And logically this virtual loss propagates? 
    // Usually virtual loss is local to the node to affect selection. 
    // But since Q is backed up, does parent Q change?
    // Standard AlphaZero: Virtual loss is applied to all nodes traversed?
    // "We apply a virtual loss to all nodes on the simulation path"
    // Yes.
    void apply_virtual_loss_recursive() {
        apply_virtual_loss();
        // Don't recurse up, we handle path iteratively in MCTS
    }
    
    // Revert virtual loss recursively
    // This is handled by update_recursive calling update which adds back VIRTUAL_LOSS
    // Wait, update_recursive calls update. Update adds VIRTUAL_LOSS.
    // parent->update_recursive(-leaf).
    // The leaf_value sign flips.
    // Virtual loss is always "bad", so -1.
    // If we simply apply -1 everywhere on the way down.
    // Then on way up, we add +1 + real_value.
    // But value flips sign at each level.
    // If I am P1, child is P2.
    // I pick child. I want to discourage picking child again.
    // Child Q should decrease.
    // Child perspective: P2. Lower Q is bad for P2?
    // Q is value for the player at that node.
    // If P2 node Q decreases, it means P2 is losing.
    // Parent (P1) chooses child with max value.
    // If P2 Q decreases (becomes more negative), does P1 pick it less?
    // No, P1 picks child maximizing Q_child?
    // No, usually in AlphaZero:
    // Q(s,a) is value of taking action a from s.
    // This Q is stored in the edge (or child node).
    // This value is from perspective of player at s.
    // So child node stores value for player at s (parent).
    // In my implementation: Q is from perspective of player at `node`.
    // My select flips perspective implicitly?
    // `parent->update_recursive(-leaf_value)`.
    // Yes. Leaf value is for leaf player. Parent gets -leaf.
    // So `node->total_value` is from `node` player perspective.
    // `parent` selects child maximizing... what?
    // `select`: `Q + u`. `Q = child->total_value / N`.
    // If `child` stores value for `child` player (who is opponent of `parent`),
    // then `parent` should minimize `child->Q`?
    // Or `child` stores value for `parent`?
    // Standard MCTS: Edge stores Q for current player.
    // My implementation: `node->update_recursive(-leaf_value)`.
    // If leaf is win for leaf-player (1.0). Parent gets -1.0.
    // So parent (loser) stores -1.0.
    // Parent's parent (winner) gets +1.0.
    // So `node->total_value` is indeed from perspective of player at `node`.
    // `select` maximizes `child->total_value`.
    // If `parent` (P1) picks child (P2), `child` stores value for P2.
    // P1 wants to move to a state where P2 loses (value -1).
    // So P1 should MINIMIZE child Q?
    // Existing code: `if (val > max_val)`. Maximizes.
    // This implies `child` stores value for `parent`?
    // Let's check `update_recursive`.
    // `update(leaf_value)`.
    // `if (parent) parent->update_recursive(-leaf_value)`.
    // Leaf (P_L). Value v.
    // Parent (P_L-1). Value -v.
    // If `child` stores value for P_L, and `parent` maximizes `child->Q`.
    // Parent P_L-1 wants to WIN. So -v should be high.
    // So v (child value) should be low.
    // So parent should MINIMIZE child value.
    // BUT `select` MAXIMIZES.
    // This implies my MCTS implementation is relying on `child` storing value for `parent`.
    // Let's trace `update_recursive` from leaf.
    // Leaf (state S_L). `update(v)`. Stores v.
    // Leaf is child of Parent node (state S_L-1).
    // Parent calls `select`. Iterates children.
    // Maximizes `child->Q`.
    // If `child` stores `v` (value for S_L player), and `parent` maximizes it...
    // That means `parent` assumes `v` is good for `parent`.
    // But `v` is good for S_L (opponent).
    // This is a contradiction unless `child` stores value for `parent`.
    // In `update_recursive`: `update(leaf_value)`. This updates `node` (child) with `v`.
    // `parent->update_recursive(-v)`. This updates `parent` with `-v`.
    // So `child` definitely stores `v` (for S_L).
    // So `parent` maximizing `child->Q` is maximizing value for OPPONENT.
    // This means `select` logic is WRONG or `update` logic is WRONG in original code.
    // UNLESS...
    // Maybe `child` in `children` map represents "Action to take".
    // And `child` node stores statistics for that edge.
    // The value Q(s,a) is usually stored in the edge.
    // In my code, `child` node IS the edge container.
    // Q(s,a) should be value for player at `s`.
    // My `child` stores `v` (value for player at `child`).
    // So `child->Q` is value for player at `child` (opponent).
    // Parent maximizes it? That implies parent wants opponent to win.
    // ERROR in original logic??
    
    // Let's assume standard AlphaZero logic:
    // We select edge (s,a) maximizing Q(s,a) + U(s,a).
    // Q(s,a) is value for player `s`.
    // When we backup from leaf `s_L` with value `v` (for player `s_L`):
    // The parent of `s_L` is `s_L-1`.
    // `s_L-1` made move to get to `s_L`.
    // So edge (`s_L-1`, `a`) leads to `s_L`.
    // We want to update Q(`s_L-1`, `a`) with value for `s_L-1`.
    // Value for `s_L-1` is `-v`.
    // So we should update `child` (representing edge/node `s_L`) with `-v`.
    // My code: `update(leaf_value)` -> `child` gets `v`.
    // This confirms `child` stores value for `child` player.
    // So `select` logic maximizing `child->Q` is WRONG.
    // It should minimize? Or `update` should flip sign?
    
    // CORRECTION:
    // To support standard maximization in select, `child` must store value for `parent`.
    // So when backing up:
    // `leaf_value` (for leaf player).
    // `node` (leaf). `update(leaf_value)`. Stores value for leaf player.
    // `parent` needs value for parent player: `-leaf_value`.
    // But `parent` selects `node`.
    // It looks at `node->Q`.
    // `node->Q` must be value for `parent`.
    // So `node->update` should receive `-leaf_value`??
    // Let's look at `playout` in original code:
    // `node->update_recursive(-leaf_value);`
    // It calls `node->update(-leaf_value)`.
    // So `node` (leaf) stores `-leaf_value`.
    // `-leaf_value` is value for PARENT of leaf.
    // So `node` stores value for its PARENT.
    // `select` maximizes `node->Q`. Parent maximizes value for Parent.
    // THIS IS CORRECT.
    // My analysis of `update_recursive` was slightly off.
    // `node->update_recursive(v)`:
    // `update(v)`. Node gets v.
    // `parent->update_recursive(-v)`. Parent gets -v.
    
    // So in `playout`:
    // Leaf state S_L. Value `v` (for S_L player).
    // We call `node->update_recursive(-v)`.
    // `node` gets `-v` (Value for S_L-1, i.e., Parent).
    // `parent` gets `-(-v) = v` (Value for S_L-2, i.e. S_L player).
    // This "flip every level" works if `node` stores value for `node->parent`.
    // Yes, Q(s,a) is stored in the node corresponding to `a`.
    // Q(s,a) is value for `s`.
    // `node` represents `(s,a)`. `node->parent` is `s`.
    // So `node` must store value for `parent`.
    // OK, logic holds.
    
    // Virtual Loss Logic with this understanding:
    // `select` picks `child`.
    // `child` stores value for `parent`.
    // We want to discourage `parent` from picking `child` again.
    // So `child->Q` should DECREASE.
    // So `child->total_value -= VIRTUAL_LOSS` is correct.
    // Backup: `child->total_value += VIRTUAL_LOSS + real_value`.
    // `real_value` passed to `child` is `-leaf_value` (value for parent).
    // So `child->total_value` eventually converges to N * (-leaf_value).
    // Correct.
    
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
    int batch_size;

public:
    MCTS(std::string model_path, double c_puct, int n_playout, std::string device_str) 
        : c_puct(c_puct), n_playout(n_playout), device(device_str), batch_size(16) { // Default batch 16
        try {
            module = torch::jit::load(model_path, device);
            module.eval();
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
            throw e;
        }
        root = std::make_unique<TreeNode>(nullptr, 1.0);
    }

    void process_batch(int current_batch_size, CublinoState& root_state) {
        // 1. Selection Phase
        std::vector<TreeNode*> leaves;
        std::vector<CublinoState> leaf_states;
        std::vector<std::vector<TreeNode*>> paths;
        std::vector<bool> is_terminal;
        std::vector<int> terminal_rewards;
        
        leaves.reserve(current_batch_size);
        leaf_states.reserve(current_batch_size);
        paths.reserve(current_batch_size);
        is_terminal.reserve(current_batch_size);
        terminal_rewards.reserve(current_batch_size);
        
        for(int b=0; b<current_batch_size; ++b) {
            TreeNode* node = root.get();
            CublinoState state = root_state.copy();
            std::vector<TreeNode*> path;

            // FIX: Apply VL to root immediately so it can be reverted later
            node->apply_virtual_loss(); 
            path.push_back(node);
            
            bool term = false;
            int reward = 0;

            while (!node->is_leaf()) {
                auto selection = node->select(c_puct);
                int action = selection.first;
                node = selection.second;
                path.push_back(node);
                
                // Apply virtual loss immediately to affect next selection in this batch
                node->apply_virtual_loss();

                auto step_res = state.step(action);
                term = step_res.second;
                reward = step_res.first;
                
                if (term) break;
            }
            
            leaves.push_back(node);
            leaf_states.push_back(std::move(state));
            paths.push_back(std::move(path));
            is_terminal.push_back(term);
            terminal_rewards.push_back(reward);
        }

        // 2. Inference Phase
        std::vector<torch::Tensor> obs_list;
        std::vector<int> valid_indices;
        obs_list.reserve(current_batch_size);
        valid_indices.reserve(current_batch_size);

        for(int i=0; i<current_batch_size; ++i) {
            if (!is_terminal[i]) {
                obs_list.push_back(leaf_states[i].get_obs());
                valid_indices.push_back(i);
            }
        }

        torch::Tensor policy_probs_batch;
        torch::Tensor values_batch;

        if (!obs_list.empty()) {
            torch::Tensor batch_obs = torch::stack(obs_list).to(device).squeeze(1); // (B, 3, 7, 7)
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(batch_obs);

            auto output = module.forward(inputs).toTuple();
            // Move to CPU once
            policy_probs_batch = torch::softmax(output->elements()[0].toTensor(), 1).cpu(); // (B, 196)
            values_batch = output->elements()[1].toTensor().cpu(); // (B, 1)
        }

        // 3. Backup Phase
        for(int i=0; i<current_batch_size; ++i) {
            TreeNode* leaf = leaves[i];
            double value = 0;

            if (is_terminal[i]) {
                if (terminal_rewards[i] == 0) {
                     value = 0.0;
                } else {
                     value = 1.0;
                }
            } else {
                // Find index in batch
                int batch_idx = -1;
                for(int k=0; k<valid_indices.size(); ++k) if(valid_indices[k]==i) batch_idx=k;
                
                torch::Tensor probs = policy_probs_batch[batch_idx];
                value = values_batch[batch_idx].item<double>(); // Value for current player at leaf

                // Expand
                std::vector<int> legal_moves = leaf_states[i].get_legal_actions();
                std::vector<std::pair<int, double>> action_priors;
                action_priors.reserve(legal_moves.size());
                
                double sum_probs = 0;
                for (int move : legal_moves) {
                    double prob = probs[move].item<double>();
                    action_priors.push_back({move, prob});
                    sum_probs += prob;
                }
                
                if (sum_probs > 0) {
                    for (auto& ap : action_priors) ap.second /= sum_probs;
                } else {
                    for (auto& ap : action_priors) ap.second = 1.0 / legal_moves.size();
                }
                leaf->expand(action_priors);
                
                // Value from Net is for player at Leaf.
                // We want to pass value to update logic.
                // `playout` did `node->update_recursive(-leaf_value)`.
                // So if net says 0.8 (good for Leaf), we pass -0.8.
                // Node (Leaf) gets -0.8 (bad for Parent).
                // Parent gets 0.8. Correct.
                value = -value;
            }

            // Backup along path
            // We need to revert virtual loss on the path nodes.
            // AND update with real value.
            // My TreeNode::update adds VIRTUAL_LOSS + leaf_value.
            // So we just call update_recursive(value) on leaf?
            // Yes, but we need to call it on the leaf node.
            // `update_recursive` traverses up to root.
            // But wait, we applied virtual loss to ALL nodes on path?
            // In selection loop: `node->apply_virtual_loss()` for each selected node.
            // Path includes root? 
            // Loop: `node` starts at root. `select` returns child.
            // `node` becomes child. `path.push_back(node)`.
            // `node->apply_virtual_loss()`.
            // So we applied to child, grandchild, ... leaf.
            // Root virtual loss? 
            // Standard: Root visit count increases.
            // My loop applied to all SELECTED nodes.
            // So root didn't get virtual loss (it wasn't selected, it was start).
            // But root N should increase.
            // In `get_move_probs`, root N is sum of children N.
            // `update_recursive` goes up to parent.
            // Does it stop at root? `if (parent)`. Root has no parent.
            // `update(leaf_value)`.
            // So calling `leaf->update_recursive(value)` will traverse up and fix everyone.
            // Including root.
            // BUT, `root` did NOT have `apply_virtual_loss` called on it in my loop!
            // `path` contains `node` after select.
            // Correct.
            // So `leaf->update_recursive` will add `VIRTUAL_LOSS` to root... but root never subtracted it!
            // ERROR.
            // `update` assumes `VIRTUAL_LOSS` was subtracted.
            // So we must ensure `apply_virtual_loss` logic matches `update` logic.
            // My `update_recursive` goes all the way to root.
            // So `root` will get `+ VIRTUAL_LOSS`.
            // So `root` must have had `apply_virtual_loss`.
            // Solution: Apply virtual loss to root at start of traversal?
            // Or fix `update` to not add VIRTUAL_LOSS if node is root?
            // Simpler: Apply VL to root.
            // In loop: `TreeNode* node = root.get(); node->apply_virtual_loss();`
            
            // Wait, path vector: `path.push_back(node)` (root).
            // `select` -> child. `path.push_back(child)`. `child->apply_vl()`.
            // So I need to apply to root too.
            // Actually, simpler:
            // Iterate `path` (which includes root and all selected nodes).
            // For each node in path:
            // `node->revert_virtual_loss()`.
            // Then `leaf->update_recursive_standard(value)`.
            // This decouples VL revert from value backup.
            // Let's modify TreeNode to allow this clean separation.
            
            // Refactored TreeNode logic for batching:
            // 1. apply_virtual_loss(): N++, W -= 1.0.
            // 2. revert_virtual_loss(): W += 1.0. (N stays ++).
            // 3. update_standard(val): W += val. (N stays).
            
            // Loop path: revert_virtual_loss().
            // Then leaf->update_recursive_standard(value). (With sign flipping).
            
            for (TreeNode* n : paths[i]) {
                n->total_value += TreeNode::VIRTUAL_LOSS;
            }
            
            // Now do standard backup
            // `value` is already adjusted to be passed to update_recursive (e.g. -leaf_value).
            leaf->update_recursive_no_virtual(value);
        }
    }

    std::pair<std::vector<int>, std::vector<double>> get_move_probs(const CublinoState& state, double temp) {
        int remaining = n_playout;
        while (remaining > 0) {
            int current_batch = std::min(batch_size, remaining);
            // Must cast const away or copy? `process_batch` takes `CublinoState&` (root state).
            // It needs to copy it for traversals.
            // `state` is const ref. Make a copy.
            CublinoState root_state = state.copy();
            process_batch(current_batch, root_state);
            remaining -= current_batch;
        }

        // ... return logic ...
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

// ... TreeNode extra methods needed ...
// In TreeNode:
// void update_recursive_no_virtual(double leaf_value) {
//     total_value += leaf_value;
//     if (parent) parent->update_recursive_no_virtual(-leaf_value);
// }

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