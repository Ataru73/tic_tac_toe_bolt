import gymnasium as gym
import os
import torch
import numpy as np

import argparse
import sys
import time
import json

os.environ['PYOPENGL_PLATFORM'] = 'glx'
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

try:
    from src.cublino_contra.model import PolicyValueNet
    from src.cublino_contra.mcts import MCTS, MCTS_CPP
    import src.cublino_contra # Register env
except ImportError:
    # If running as script from src/cublino_contra/
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.cublino_contra.model import PolicyValueNet
    from src.cublino_contra.mcts import MCTS, MCTS_CPP
    import src.cublino_contra # Register env

class HumanPlayer:
    def __init__(self):
        self.player = None
        self.selected_square = None # (row, col)
        self.legal_moves = [] # List of (target_row, target_col) for selected square
        self.pending_action = None # New: to store action from mouse callback
    
    def set_player_ind(self, p):
        self.player = p

    def get_action(self, env):
        if self.pending_action is not None:
            action = self.pending_action
            self.pending_action = None
            return action
        return None

class MCTSPlayer:
    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, use_cpp=False, model_path=None, device="cpu"):
        self._policy_value_function = policy_value_function
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._model_path = model_path
        self._device = device
        self._use_cpp = use_cpp

        self._init_mcts()

    def _init_mcts(self):
        if self._use_cpp and self._model_path is not None:
            try:
                self.mcts = MCTS_CPP(self._model_path, self._c_puct, self._n_playout, str(self._device))
                print("Using C++ MCTS for AI player.")
            except Exception as e:
                print(f"Failed to load C++ MCTS for AI: {e}. Falling back to Python MCTS.")
                self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout)
                self._use_cpp = False
        else:
            self.mcts = MCTS(self._policy_value_function, self._c_puct, self._n_playout)
            print("Using Python MCTS for AI player.")

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, env):
        legal_moves = env.unwrapped.get_legal_actions()
        if len(legal_moves) > 0:
            # Print NN evaluation
            if self._policy_value_function:
                _, value = self._policy_value_function(env)
                print(f"NN Evaluation: Value: {value:.4f}")

            acts, probs = self.mcts.get_move_probs(env, temp=1e-3)
            move = acts[np.argmax(probs)]
            return move
        else:
            print("WARNING: No sensible moves found!")
            return -1



camera_distance = 15.0
camera_angle_y = 45.0
camera_angle_x = 30.0

# Global variables for GLUT callbacks
current_env = None
human_player_instance = None
ai_player_instance = None
current_player_is_human = False
last_frame_time = time.time()
mouse_right_down = False
last_mouse_x = 0
last_mouse_y = 0
game_over = False
game_moves = []  # Track all moves for export

# Fixed camera presets for numpad keys (distance, angle_x, angle_y)
camera_presets = {
    # Angle X: 0 horizontal, 90 straight down
    # Angle Y: 0 Front (+Z), 90 Right (+X), 180 Back (-Z), 270 Left (-X)
    # Board is 0,0 (P1-Left) to 6,6 (P2-Right) in object coordinates.
    # After rotation: P1 is at Z=0, P2 is at Z=-6.
    # Front is along +Z (looking at P1), Back is along -Z (looking at P2).
    # Left is along -X, Right is along +X.

    b'5': (10.0, 89.0, 0.0),   # Top (looking straight down)

    b'1': (15.0, 30.0, -45.0),  # Front-Left
    b'3': (15.0, 30.0, 45.0),   # Front-Right
    b'7': (15.0, 30.0, -135.0), # Back-Left
    b'9': (15.0, 30.0, 135.0),  # Back-Right

    b'8': (15.0, 30.0, 180.0),  # Back (North)
    b'2': (15.0, 30.0, 0.0),    # Front (South)
    b'4': (15.0, 30.0, -90.0),  # Left (West)
    b'6': (15.0, 30.0, 90.0),   # Right (East)
}

class ReplayPlayer:
    def __init__(self, moves):
        self.moves = moves
        self.move_idx = 0
        self.player = None
        self.last_move_time = 0
        self.move_delay = 1.0 # Seconds between moves

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.move_idx = 0

    def get_action(self, env):
        if self.move_idx < len(self.moves):
            # Check delay
            if time.time() - self.last_move_time < self.move_delay:
                return None
            
            action = self.moves[self.move_idx]
            self.move_idx += 1
            self.last_move_time = time.time()
            return action
        return None

def generate_marble_texture(width=512, height=512):
    # Procedural Marble Texture using Fractal Noise
    # Formula: marble(x,y) = sin(f * (x + a * turb(x,y)))
    
    # 1. Generate Turbulence (Fractal Noise)
    turbulence = np.zeros((width, height))
    octaves = 4 # Reduced octaves to avoid high-freq noise
    
    for i in range(octaves):
        # Scale: Start low freq (large scale), go to high freq
        # i=0: scale=128, i=1: 64, i=2: 32, i=3: 16
        # Avoids 8, 4, 2 scales which cause graininess
        scale = 2 ** (7 - i) 
        
        # Generate small noise
        grid_w = int(np.ceil(width / scale))
        grid_h = int(np.ceil(height / scale))
        noise_layer = np.random.rand(grid_w, grid_h)
        
        # Upscale to full size
        full_noise = np.kron(noise_layer, np.ones((scale, scale)))
        full_noise = full_noise[:width, :height]
        
        # Smooth to remove blocks
        passes = int(scale * 1.5) # More smoothing passes
        for _ in range(passes):
             full_noise = (np.roll(full_noise, 1, axis=0) + np.roll(full_noise, -1, axis=0) +
                           np.roll(full_noise, 1, axis=1) + np.roll(full_noise, -1, axis=1) + 4*full_noise) / 8.0
        
        # Add to turbulence
        amplitude = 0.5 ** i
        turbulence += np.abs(full_noise - 0.5) * amplitude

    # Normalize turbulence
    turbulence = (turbulence - turbulence.min()) / (turbulence.max() - turbulence.min())
    
    # 2. Marble Formula
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Parameters
    freq = 3.0 * 2 * np.pi # Slightly fewer stripes
    amp = 2.0 # Slightly less distortion
    
    # Pattern: sin(f * (x + a * turb))
    pattern = np.sin(freq * ((X + Y) + amp * turbulence))
    
    # Map -1..1 to 0..1
    pattern = (pattern + 1) / 2.0
    
    # 3. Color Mapping
    texture = np.zeros((width, height, 3), dtype=np.float32)
    
    # Colors
    col_vein = np.array([0.6, 0.6, 0.65]) # Light Grey/Blue
    col_body = np.array([0.98, 0.98, 0.98]) # White
    
    # Sharpen veins (higher power = thinner veins, more white body)
    pattern = pattern ** 0.4 # Lower power biases towards 1 (white) for lighter look
    # Actually, to make veins distinct but smooth, we want a smooth transition.
    # Let's try a slight sigmoid or just keep it linear-ish but biased towards white.
    # pattern ** 0.5 biases towards 1 (white).
    
    for c in range(3):
        texture[:,:,c] = pattern * col_body[c] + (1-pattern) * col_vein[c]
        
    tex_data = (texture * 255).astype(np.uint8)
    
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data)
    
    return tex_id

def run_game(model_path=None, human_starts=True, difficulty=20, replay_file=None):
    # 1. Initialize Environment
    env = gym.make("CublinoContra-v0")
    env.reset()
    
    # 2. Initialize AI and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if replay_file:
        with open(replay_file, 'r') as f:
            data = json.load(f)
        
        initial_board = None
        initial_player = None

        if isinstance(data, list):
            # New format
            moves = [step['move'] for step in data if 'move' in step]
            winner = "Unknown"
            
            # Check for initial state in first record
            if len(data) > 0 and 'board_state' in data[0]:
                initial_board = data[0]['board_state']
                if 'current_player_at_step' in data[0]:
                    initial_player = data[0]['current_player_at_step']
        else:
            # Old format
            moves = data['moves']
            winner = data['winner']
            
        print(f"Replaying game from {replay_file}. Winner: {winner}")
        
        # ... (Rest of Replay setup) ...

        class SharedReplayState:
            def __init__(self, moves):
                self.moves = moves
                self.idx = 0
                self.last_time = time.time()
                
        shared_state = SharedReplayState(moves)
        
        class ReplayAgent:
            def __init__(self, shared_state):
                self.state = shared_state
                self.player = 0
                self.selected_square = None
                self.legal_moves = []
            def set_player_ind(self, p): self.player = p
            def reset_player(self): pass
            def get_action(self, env):
                if self.state.idx >= len(self.state.moves): return None
                if time.time() - self.state.last_time < 0.5: return None
                
                action = self.state.moves[self.state.idx]
                self.state.idx += 1
                self.state.last_time = time.time()
                return action

        human = ReplayAgent(shared_state)
        ai_player = ReplayAgent(shared_state)
        
        human.set_player_ind(1)
        ai_player.set_player_ind(-1)
        
    else:
        # ... (Normal play setup) ...
        # Load model
        policy_value_net = PolicyValueNet(board_size=7).to(device)
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'policy_value_net' in checkpoint:
                    policy_value_net.load_state_dict(checkpoint['policy_value_net'])
                else:
                    policy_value_net.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                 print(f"Could not load state dict from {model_path}: {e}")
                 print("Assuming it is a script model or invalid.")
        elif model_path:
            print(f"Warning: Model path {model_path} does not exist. Using untrained model.")
        else:
            print("No model path provided. Using untrained model.")
            
        # Save model as TorchScript for C++ MCTS
        model_path_cpp = "temp_play_model_cublino.pt"
        script_model = torch.jit.script(policy_value_net.cpu())
        script_model.save(model_path_cpp)
        policy_value_net.to(device) # Move back to device if needed
    
        def policy_value_fn(env):
            legal_actions = env.unwrapped.get_legal_actions() # Use unwrapped for accessing method
            # Use _get_obs() to get the full 12-channel observation (history of 4 states)
            obs = env.unwrapped._get_obs()
            board_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                log_act_probs, value = policy_value_net(board_tensor)
                act_probs = np.exp(log_act_probs.cpu().numpy().flatten())
                
            return zip(legal_actions, act_probs[legal_actions]), value.item()
    
        # Difficulty
        n_playout = difficulty * 20
        print(f"Difficulty Level: {difficulty} (Simulations: {n_playout})")
    
        # Players
        human = HumanPlayer()
        ai_player = MCTSPlayer(policy_value_fn, c_puct=5, n_playout=n_playout, use_cpp=True, model_path=model_path_cpp, device=device)
        
        human.set_player_ind(1 if human_starts else -1)
        ai_player.set_player_ind(-1 if human_starts else 1)
        
        ai_player.reset_player()

    # 3. Setup Global State for Callbacks
    global current_env, human_player_instance, ai_player_instance, current_player_is_human, game_moves
    current_env = env
    human_player_instance = human
    ai_player_instance = ai_player
    current_player_is_human = human_starts if not replay_file else False # In replay, neither is "human" in the interactive sense
    game_moves = []  # Reset game moves list

    # Set custom initial state if provided (for partial replay)
    if replay_file and 'initial_board' in locals() and initial_board is not None:
        env.unwrapped.board = np.array(initial_board, dtype=np.int8)
        if initial_player is not None:
             env.unwrapped.current_player = initial_player
        print("Loaded custom initial state from replay file.")

    # 4. Init OpenGL
    if not bool(glutInit):
        raise RuntimeError("GLUT is not available. Please ensure freeglut3-dev is installed.")
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(800, 800)
    glutCreateWindow(b"Cublino Contra 3D")
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)

    glClearColor(0.8, 0.8, 0.8, 1.0) # Light grey background

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1, 0.1, 100.0) # Field of view, aspect, near, far
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Light setup
    light_position = [10.0, 10.0, 10.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

    # Generate Marble Texture
    marble_tex_id = generate_marble_texture()

    # Helper for die logic
    _val_to_vec = {
        1: np.array([0, 0, 1]),
        6: np.array([0, 0, -1]),
        2: np.array([0, -1, 0]), 
        5: np.array([0, 1, 0]),  
        3: np.array([1, 0, 0]),  
        4: np.array([-1, 0, 0]) 
    }
    _vec_to_val = {tuple(v): k for k, v in _val_to_vec.items()}

    def draw_text_centered(text):
        glPushMatrix()
        scale = 0.003
        glScalef(scale, scale, scale)
        w = 0
        for c in text:
            w += glutStrokeWidth(GLUT_STROKE_ROMAN, ord(c))
        # 119.05 is the height of GLUT_STROKE_ROMAN
        glTranslatef(-w/2, -119.05/2, 0)
        glLineWidth(3.0)
        for c in text:
            glutStrokeCharacter(GLUT_STROKE_ROMAN, ord(c))
        glPopMatrix()

    def draw_die(p, top, south):
        # Calculate other faces
        v_top = _val_to_vec[top]
        v_south = _val_to_vec[south]
        v_east = np.cross(v_top, v_south)
        
        east = _vec_to_val.get(tuple(v_east), 0)
        north = 7 - south
        west = 7 - east
        bottom = 7 - top
        
        # Draw Cube
        if p == 1:
            glColor3f(0.8, 0.2, 0.2) # Redish
        else:
            glColor3f(0.2, 0.2, 0.8) # Bluish
        glutSolidCube(0.8)
        
        # Draw Numbers
        glColor3f(1.0, 1.0, 1.0) # White text
        # glDisable(GL_LIGHTING) # Optional: Make text self-luminous
        
        dist = 0.41
        
        # Top (+Z)
        glPushMatrix()
        glTranslatef(0, 0, dist)
        draw_text_centered(str(top))
        glPopMatrix()
        
        # Bottom (-Z)
        glPushMatrix()
        glTranslatef(0, 0, -dist)
        glRotatef(180, 1, 0, 0)
        draw_text_centered(str(bottom))
        glPopMatrix()
        
        # South (-Y)
        glPushMatrix()
        glTranslatef(0, -dist, 0)
        glRotatef(90, 1, 0, 0)
        draw_text_centered(str(south))
        glPopMatrix()
        
        # North (+Y)
        glPushMatrix()
        glTranslatef(0, dist, 0)
        glRotatef(-90, 1, 0, 0)
        glRotatef(180, 0, 0, 1) # Flip so it's upright relative to board? 
        # Actually standard: Text up is +Y (World Z).
        # Rotating -90 around X: Y -> Z. 
        # So text up (local Y) becomes world Z. This is correct.
        draw_text_centered(str(north))
        glPopMatrix()
        
        # East (+X)
        glPushMatrix()
        glTranslatef(dist, 0, 0)
        glRotatef(90, 0, 1, 0)
        glRotatef(90, 0, 0, 1) # Adjust orientation
        draw_text_centered(str(east))
        glPopMatrix()
        
        # West (-X)
        glPushMatrix()
        glTranslatef(-dist, 0, 0)
        glRotatef(-90, 0, 1, 0)
        glRotatef(-90, 0, 0, 1)
        draw_text_centered(str(west))
        glPopMatrix()

        # glEnable(GL_LIGHTING)


    # 5. Define Callbacks
    def display_callback():
        global camera_distance, camera_angle_x, camera_angle_y

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        gluLookAt(
            camera_distance * np.sin(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x)),
            camera_distance * np.sin(np.radians(camera_angle_x)),
            camera_distance * np.cos(np.radians(camera_angle_y)) * np.cos(np.radians(camera_angle_x)),
            3.5, 0.0, -3.5,   # Look at center of the board (accounting for -90° X rotation)
            0.0, 1.0, 0.0   # Up vector
        )
        
        glRotatef(-90, 1, 0, 0) # Rotate board to be horizontal

        # Draw Marble Board
        glEnable(GL_LIGHTING)
        
        # Marble Material
        mat_ambient = [0.1, 0.1, 0.1, 1.0]
        mat_diffuse = [0.9, 0.9, 0.95, 1.0] # Slightly bluish white
        mat_specular = [1.0, 1.0, 1.0, 1.0]
        mat_shininess = [100.0]

        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess)

        # Board Base
        # Board Base with Texture
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, marble_tex_id)
        
        glPushMatrix()
        glTranslatef(3.5, 3.5, -0.5) 
        glScalef(8.0, 8.0, 1.0) 
        
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        # Top Face
        glNormal3f( 0.0, 0.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-0.5, -0.5,  0.5)
        glTexCoord2f(1.0, 0.0); glVertex3f( 0.5, -0.5,  0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f( 0.5,  0.5,  0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f(-0.5,  0.5,  0.5)
        # Bottom Face
        glNormal3f( 0.0, 0.0,-1.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 0.0); glVertex3f( 0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f( 0.5,  0.5, -0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f(-0.5,  0.5, -0.5)
        # Sides
        glNormal3f( 0.0, 1.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-0.5,  0.5, -0.5)
        glTexCoord2f(1.0, 0.0); glVertex3f(-0.5,  0.5,  0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f( 0.5,  0.5,  0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f( 0.5,  0.5, -0.5)
        
        glNormal3f( 0.0,-1.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f(-0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f(-0.5, -0.5,  0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f( 0.5, -0.5,  0.5)
        glTexCoord2f(0.0, 0.0); glVertex3f( 0.5, -0.5, -0.5)

        glNormal3f( 1.0, 0.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f( 0.5, -0.5,  0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f( 0.5,  0.5,  0.5)
        glTexCoord2f(0.0, 0.0); glVertex3f( 0.5,  0.5, -0.5)

        glNormal3f(-1.0, 0.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-0.5, -0.5, -0.5)
        glTexCoord2f(1.0, 0.0); glVertex3f(-0.5, -0.5,  0.5)
        glTexCoord2f(1.0, 1.0); glVertex3f(-0.5,  0.5,  0.5)
        glTexCoord2f(0.0, 1.0); glVertex3f(-0.5,  0.5, -0.5)
        glEnd()

        glPopMatrix()
        glDisable(GL_TEXTURE_2D)

        # Draw grid lines on top
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D) # Ensure no texture interferes
        glColor3f(0.3, 0.3, 0.3) # Darker grey lines for contrast
        glLineWidth(2.0)
        
        z_offset = 0.01 # Slightly above the board
        
        for i in range(8):
            glBegin(GL_LINES)
            glVertex3f(i, 0, z_offset)
            glVertex3f(i, 7, z_offset)
            glEnd()
            glBegin(GL_LINES)
            glVertex3f(0, i, z_offset)
            glVertex3f(7, i, z_offset)
            glEnd()
        glEnable(GL_LIGHTING)

        # Draw dice
        for r in range(7):
            for c in range(7):
                p, top, south = env.unwrapped.board[r, c]
                if p != 0:
                    glPushMatrix()
                    glTranslatef(c + 0.5, r + 0.5, 0.5) # Center cube on grid square
                    draw_die(p, top, south)
                    glPopMatrix()

        # Highlight Selection
        if human.selected_square:
            r, c = human.selected_square
            glPushMatrix()
            glTranslatef(c + 0.5, r + 0.5, 0.5)
            glColor3f(0.0, 1.0, 0.0) # Green
            glutWireCube(0.9)
            glPopMatrix()
            
        # Highlight Legal Moves
        if human.legal_moves:
            for r, c, d in human.legal_moves:
                glPushMatrix()
                glTranslatef(c + 0.5, r + 0.5, 0.5)
                glColor4f(0.0, 1.0, 0.0, 0.3) # Green transparent
                glutSolidCube(0.9)
                glPopMatrix()

        glutSwapBuffers()

    def mouse_click(button, state, x, y):
        global current_player_is_human, mouse_right_down, last_mouse_x, last_mouse_y # Declare global for assignment
        
        # Camera control with Right Mouse Button
        if button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                mouse_right_down = True
                last_mouse_x = x
                last_mouse_y = y
            elif state == GLUT_UP:
                mouse_right_down = False
            return

        # Game interaction with Left Mouse Button
        if not current_player_is_human or button != GLUT_LEFT_BUTTON or state != GLUT_DOWN:
            return

        # Get current projection and modelview matrices
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        viewport_matrix = glGetIntegerv(GL_VIEWPORT)

        # Unproject mouse coordinates to world coordinates
        winX = float(x)
        winY = float(viewport_matrix[3] - y) # Invert Y-axis

        # Get 3D world coordinates for near and far clipping planes
        try:
            near_point = gluUnProject(winX, winY, 0.0, modelview_matrix, projection_matrix, viewport_matrix)
            far_point = gluUnProject(winX, winY, 1.0, modelview_matrix, projection_matrix, viewport_matrix)
        except Exception:
            return

        # Ray direction
        ray_dir = np.array(far_point) - np.array(near_point)
        norm = np.linalg.norm(ray_dir)
        if norm == 0: return
        ray_dir = ray_dir / norm
        ray_origin = np.array(near_point)

        # Intersection with Z=0 plane (board)
        if ray_dir[2] == 0: return
        t = -ray_origin[2] / ray_dir[2]
        
        if t < 0: # Intersection is behind camera
            human_player_instance.selected_square = None
            human_player_instance.legal_moves = []
            glutPostRedisplay()
            return

        intersect_point = ray_origin + t * ray_dir
        board_x, board_y = int(intersect_point[0]), int(intersect_point[1])

        if not (0 <= board_x < 7 and 0 <= board_y < 7):
            human_player_instance.selected_square = None
            human_player_instance.legal_moves = []
            glutPostRedisplay()
            return

        # Check if clicked on own piece
        if current_env.unwrapped.board[board_y, board_x, 0] == human_player_instance.player:
            human_player_instance.selected_square = (board_y, board_x)
            human_player_instance.legal_moves = []
            # Calculate legal moves for this piece
            for d in range(4):
                dr, dc = 0, 0
                if d == 0: dr = 1  # North (Row + 1)
                elif d == 1: dc = 1 # East (Col + 1)
                elif d == 2: dr = -1 # South (Row - 1)
                elif d == 3: dc = -1 # West (Col - 1)
                
                tr, tc = board_y + dr, board_x + dc
                if 0 <= tr < 7 and 0 <= tc < 7 and current_env.unwrapped.board[tr, tc, 0] == 0:
                    human_player_instance.legal_moves.append((tr, tc, d))

        elif human_player_instance.selected_square:
            # Check if clicked on a legal target
            for tr, tc, d in human_player_instance.legal_moves:
                if tr == board_y and tc == board_x:
                    # Execute move
                    sr, sc = human_player_instance.selected_square
                    action = (sr * 7 + sc) * 4 + d
                    human_player_instance.pending_action = action
                    human_player_instance.selected_square = None
                    human_player_instance.legal_moves = []
                    # game_loop_idle() # Process the move immediately - Removed to allow display update
                    break
            else:
                # Not a legal target, deselect
                human_player_instance.selected_square = None
                human_player_instance.legal_moves = []
        
        glutPostRedisplay()

    def keyboard_callback(key, x, y):
        global camera_distance, camera_angle_x, camera_angle_y # Declare global for assignment
        if key == b'w':
            camera_distance -= 0.5
        elif key == b's':
            camera_distance += 0.5
        elif key == b'a':
            camera_angle_y -= 5.0
        elif key == b'd':
            camera_angle_y += 5.0
        elif key == b'q': # Rotate camera up
            camera_angle_x += 5.0
            if camera_angle_x > 89.0: camera_angle_x = 89.0
        elif key == b'e': # Rotate camera down
            camera_angle_x -= 5.0
            if camera_angle_x < 1.0: camera_angle_x = 1.0
        
        elif key in camera_presets:
            camera_distance, camera_angle_x, camera_angle_y = camera_presets[key]

        if game_over:
             glutLeaveMainLoop()

        glutPostRedisplay()

    def mouse_motion(x, y):
        global mouse_right_down, last_mouse_x, last_mouse_y, camera_angle_x, camera_angle_y
        
        if mouse_right_down:
            dx = x - last_mouse_x
            dy = y - last_mouse_y
            
            camera_angle_y += dx * 0.5
            camera_angle_x += dy * 0.5
            
            # Clamp camera_angle_x
            if camera_angle_x > 89.0: camera_angle_x = 89.0
            if camera_angle_x < 1.0: camera_angle_x = 1.0
            
            last_mouse_x = x
            last_mouse_y = y
            
            glutPostRedisplay()

    def game_loop_idle():
        global current_player_is_human, last_frame_time, game_over # Declare global for assignment
        
        if game_over:
            time.sleep(0.1)
            return
        
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        # Avoid busy-waiting too much by introducing a small sleep
        if delta_time < 1/60.0: # Cap at 60 FPS
            time.sleep(1/60.0 - delta_time)

        current_player_idx = current_env.unwrapped.current_player
        player = human_player_instance if current_player_idx == human_player_instance.player else ai_player_instance
        
        if isinstance(player, HumanPlayer):
            current_player_is_human = True
            action = player.get_action(current_env) # Non-blocking call
        else:
            current_player_is_human = False
            # print("AI is thinking...") # Reduced verbosity
            action = player.get_action(current_env)

        if action is not None:
            # Track move for export (only for non-replay games)
            if not hasattr(ai_player_instance, 'state'):
                game_moves.append(int(action))
                ai_player_instance.mcts.update_with_move(action)

            obs, reward, terminated, truncated, info = current_env.step(action)
            
            if terminated or truncated:
                if truncated and info.get('reason') == "Invalid Move":
                    print(f"Game Over. Replay truncated due to invalid move: {info.get('error', 'Unknown error')}")
                    log_winner = 0 # No official winner for an invalid move truncation
                else:
                    winner = info.get('winner', 0)
                    if winner == 0:
                         print("Game Over. Draw!")
                    else:
                         print(f"Game Over. Winner: {winner}")
                    log_winner = winner
                
                # Export game log for non-replay games
                if not hasattr(ai_player_instance, 'state'):
                    timestamp = int(time.time())
                    filename = f"game_log_{timestamp}.json"
                    game_log_data = {
                        "winner": int(log_winner),
                        "moves": game_moves
                    }
                    try:
                        with open(filename, 'w') as f:
                            json.dump(game_log_data, f)
                        print(f"Game logged to {filename}")
                    except Exception as e:
                        print(f"Failed to save game log: {e}")
                
                print("Press any key to exit.")
                game_over = True
            
            glutPostRedisplay() # Redraw scene after move

    glutDisplayFunc(display_callback)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_motion)
    glutKeyboardFunc(keyboard_callback)
    glutIdleFunc(game_loop_idle)

    glutMainLoop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model .pth file")
    parser.add_argument("--ai_starts", action="store_true", help="If set, AI starts first")
    parser.add_argument("--difficulty", type=int, default=20, choices=range(1, 21), help="Difficulty level (1-20)")
    parser.add_argument("--replay", type=str, help="Path to game log JSON file for replay")
    args = parser.parse_args()
    
    run_game(model_path=args.model, human_starts=not args.ai_starts, difficulty=args.difficulty, replay_file=args.replay)
