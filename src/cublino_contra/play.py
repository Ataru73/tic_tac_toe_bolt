import gymnasium as gym
import os
import torch
import numpy as np

import argparse
import sys
import time

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

# Fixed camera presets for numpad keys (distance, angle_x, angle_y)
camera_presets = {
    # Angle X: 0 horizontal, 90 straight down
    # Angle Y: 0 east (+X), 90 north (+Y), 180 west (-X), 270 south (-Y)
    # Board is 0,0 (bottom-left) to 6,6 (top-right) in world coordinates.
    # Front is along +Y (P1 side), Back is along -Y (P2 side).
    # Left is along -X, Right is along +X.

    b'5': (10.0, 89.0, 45.0),   # Top (looking straight down)

    b'1': (15.0, 30.0, 135.0),  # Front-Left (from P1's side, left)
    b'3': (15.0, 30.0, 45.0),   # Front-Right (from P1's side, right)
    b'7': (15.0, 30.0, -135.0), # Back-Left (from P2's side, left)
    b'9': (15.0, 30.0, -45.0),  # Back-Right (from P2's side, right)

    b'8': (15.0, 30.0, -90.0),  # Back-Center (from P2's side, center)
    b'2': (15.0, 30.0, 90.0),   # Front-Center (from P1's side, center)
    b'4': (15.0, 30.0, 180.0),  # Left-Center
    b'6': (15.0, 30.0, 0.0),    # Right-Center
}

def run_game(model_path=None, human_starts=True, difficulty=20):
    # 1. Initialize Environment
    env = gym.make("CublinoContra-v0")
    env.reset()
    
    # 2. Initialize AI and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    policy_value_net = PolicyValueNet(board_size=7).to(device)
    if model_path and os.path.exists(model_path):
        try:
            policy_value_net.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except:
             print(f"Could not load state dict from {model_path}. Assuming it is a script model or invalid.")
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
        board = env.unwrapped.board
        board_tensor = torch.FloatTensor(board).permute(2, 0, 1).unsqueeze(0).to(device)
        
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
    global current_env, human_player_instance, ai_player_instance, current_player_is_human
    current_env = env
    human_player_instance = human
    ai_player_instance = ai_player
    current_player_is_human = human_starts

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
            3.0, 3.0, 0.0,  # Look at center of the board
            0.0, 1.0, 0.0   # Up vector
        )

        # Draw grid
        glDisable(GL_LIGHTING)
        glColor3f(0.5, 0.5, 0.5) # Grey lines
        glLineWidth(2.0)
        for i in range(8):
            glBegin(GL_LINES)
            glVertex3f(i, 0, 0)
            glVertex3f(i, 7, 0)
            glEnd()
            glBegin(GL_LINES)
            glVertex3f(0, i, 0)
            glVertex3f(7, i, 0)
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
                    game_loop_idle() # Process the move immediately
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
        global current_player_is_human, last_frame_time # Declare global for assignment
        
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
            ai_player_instance.mcts.update_with_move(action)

            obs, reward, terminated, truncated, info = current_env.step(action)
            
            if terminated:
                print(f"Game Over. Winner: {current_env.unwrapped.current_player}")
                glutLeaveMainLoop()
            elif truncated:
                print("Game Over. Draw!")
                glutLeaveMainLoop()
            
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
    args = parser.parse_args()
    
    run_game(model_path=args.model, human_starts=not args.ai_starts, difficulty=args.difficulty)
