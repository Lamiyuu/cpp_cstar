import pygame
import numpy as np
import math
import sys
import os
import glob

# --- CẤU HÌNH ---
DATASET_DIR = "AC300" 

WINDOW_SIZE = 800
SCALE = 1.0 
FPS = 30

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)          # Hố đã phát hiện
GHOST_GRAY = (230, 230, 230) # Hố ẩn
OUTER_COLOR = (100, 100, 100)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# START / GOAL (Dịch vào để tránh dính tường)
START_POS = np.array([2.0, 2.0]) 
GOAL_POS = np.array([48.0, 48.0]) # Goal tạm thời, sẽ cập nhật theo map nếu cần

# Tham số MCPP
EPSILON = 5.0           
EXPLORATION_C = 1.414  
GAMMA = 1.0            
MAX_ITERATIONS = 500    
SEARCH_DEPTH = 20      

# --- HÀM LOAD FILE THÔNG MINH (Sửa lỗi map bị nhọn) ---
def get_all_map_folders(root_dir):
    pattern = os.path.join(root_dir, "AC12_*") # Tìm tất cả map AC1...
    folders = sorted(glob.glob(pattern))
    return folders

def load_outer_polygon(filepath):
    """Đọc biên (Outer) - Luôn là 1 đa giác"""
    poly = []
    if not os.path.exists(filepath): return []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    x, y = float(parts[0]), float(parts[1])
                    poly.append((x, y))
                except ValueError: continue
    return poly

def load_holes_multi(filepath):
    """
    SỬA LỖI QUAN TRỌNG: Đọc nhiều hố trong cùng 1 file.
    Giả định: Các hố ngăn cách nhau bởi dòng trống hoặc định dạng NaN.
    Nếu file viết liền tù tì, ta cần logic khác. 
    Ở đây dùng logic: Nếu khoảng cách giữa 2 điểm liên tiếp quá xa (> 50% map), có thể là hố mới (Heuristic)
    HOẶC: Parse theo dòng trống (Cách chuẩn).
    """
    holes_list = []
    current_hole = []
    
    if not os.path.exists(filepath): return []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        
        # Nếu dòng trống hoặc có chữ 'NaN', ngắt hố hiện tại
        if not parts or 'NaN' in line: 
            if len(current_hole) > 2:
                holes_list.append(current_hole)
            current_hole = []
            continue

        try:
            x, y = float(parts[0]), float(parts[1])
            current_hole.append((x, y))
        except ValueError:
            continue
            
    # Lưu hố cuối cùng
    if len(current_hole) > 2:
        holes_list.append(current_hole)
        
    return holes_list

# --- HÀM HÌNH HỌC ---
def to_screen(pos):
    return int(pos[0] * SCALE), int(WINDOW_SIZE - (pos[1] * SCALE)) # Lật trục Y nếu cần (dataset này thường Y lên trên)

def dist(a, b):
    return np.linalg.norm(a - b)

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# --- CLASSES ---
class QNode:
    def __init__(self, parent, action):
        self.parent = parent; self.action = action        
        self.n = 0; self.Q = 0.0                
        self.child_v_node = None; self.r_cumulative = 0.0     

class VNode:
    def __init__(self, state):
        self.state = state; self.N = 0; self.V = 0.0; self.children = {}          

# --- PLANNER ---
class MCPP_Planner:
    def __init__(self, start, goal, outer_poly, known_holes):
        self.root = VNode(start)
        self.goal = goal
        self.outer_poly = outer_poly
        self.known_holes = known_holes # List các polygon

    def is_collision_belief(self, pos):
        # 1. Phải nằm TRONG Outer
        if not point_in_polygon(pos, self.outer_poly): return True
        # 2. Phải nằm NGOÀI tất cả các hố đã biết
        for hole in self.known_holes:
            if point_in_polygon(pos, hole): return True
        return False

    def get_action_ucb(self, v_node):
        best_score = -float('inf'); best_action = None
        for action, q_node in v_node.children.items():
            if q_node.n == 0: return action, q_node
            score = q_node.Q + EXPLORATION_C * math.sqrt(math.log(max(1, v_node.N)) / q_node.n)
            if score > best_score: best_score = score; best_action = (action, q_node)
        return best_action

    def expand(self, v_node):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(1.0, EPSILON) 
        dx = r * np.cos(angle); dy = r * np.sin(angle)
        new_pos = v_node.state + np.array([dx, dy])
        
        if self.is_collision_belief(new_pos): return None

        action = tuple(new_pos)
        if action not in v_node.children:
            v_node.children[action] = QNode(v_node, action)
        return action

    def rollout(self, state): return -dist(state, self.goal) 

    def simulate_v(self, v_node, depth):
        if depth == 0 or dist(v_node.state, self.goal) < 2.0: return self.rollout(v_node.state)
        if len(v_node.children) < 10:
            action = self.expand(v_node)
            if action:
                q_node = v_node.children[action]
                R = self.simulate_q(v_node, action, q_node, depth)
                v_node.N += 1; self.update_V(v_node)
                return R
        if not v_node.children: return self.rollout(v_node.state)
        action, q_node = self.get_action_ucb(v_node)
        R = self.simulate_q(v_node, action, q_node, depth)
        v_node.N += 1; self.update_V(v_node)
        return R
    
    def update_V(self, v_node):
        if v_node.N == 0: return
        weighted_sum = 0
        for action, q_node in v_node.children.items():
            weighted_sum += (q_node.n / v_node.N) * q_node.Q
        v_node.V = weighted_sum

    def simulate_q(self, parent_v_node, action, q_node, depth):
        step_dist = dist(parent_v_node.state, np.array(action))
        r_immediate = -step_dist 
        if q_node.child_v_node is None:
            q_node.child_v_node = VNode(np.array(action))
            future_val = self.rollout(q_node.child_v_node.state)
        else:
            future_val = self.simulate_v(q_node.child_v_node, depth - 1)
        total_return = r_immediate + GAMMA * future_val
        q_node.n += 1; q_node.r_cumulative += total_return
        q_node.Q = q_node.r_cumulative / q_node.n
        return total_return

    def search(self):
        for _ in range(MAX_ITERATIONS): self.simulate_v(self.root, SEARCH_DEPTH)
        if not self.root.children: return self.root.state
        best_action = max(self.root.children.items(), key=lambda item: item[1].Q)[0]
        return np.array(best_action)

def draw_tree(surface, v_node, parent_pos=None):
    if v_node is None: return
    current_pos_screen = to_screen(v_node.state)
    if parent_pos is not None:
        if v_node.N > 2: pygame.draw.line(surface, (180, 180, 180), parent_pos, current_pos_screen, 1)
    if dist(v_node.state, START_POS) < 30 or v_node.N > 5:
        for action, q_node in v_node.children.items():
            if q_node.child_v_node: draw_tree(surface, q_node.child_v_node, current_pos_screen)

# --- MAIN LOOP ---
def main():
    # --- SỬA LỖI SYNTAX Ở ĐÂY ---
    global SCALE # Khai báo global ở đầu hàm main nếu muốn gán lại nó trong main
    pygame.init()
    
    map_folders = get_all_map_folders(DATASET_DIR)
    if not map_folders:
        print(f"Lỗi: Không tìm thấy folder AC1_* trong '{DATASET_DIR}'!")
        sys.exit()
    
    current_map_idx = 0
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    font_large = pygame.font.SysFont("Arial", 24)

    # State variables
    outer_poly = []
    real_holes_list = [] # List các hố
    current_pos = START_POS
    path_history = []
    known_holes = [] # List các hố đã biết
    finished = False
    collision_effect = 0
    
    def reset_simulation(map_index):
        # --- SỬA LỖI SYNTAX Ở ĐÂY ---
        global SCALE # Biến toàn cục
        nonlocal outer_poly, real_holes_list, current_pos, path_history, known_holes, finished
        # ----------------------------
        
        folder_path = map_folders[map_index]
        print(f"\n--- Loading Map: {folder_path} ---")
        
        outer_path = os.path.join(folder_path, "outer_polygon")
        holes_path = os.path.join(folder_path, "holes")
        
        outer_poly = load_outer_polygon(outer_path)
        
        # SỬA: Load nhiều hố
        real_holes_list = load_holes_multi(holes_path)
        print(f"Detected {len(real_holes_list)} holes.")
        
        # Tính toán Scale tự động (Lật trục Y nếu cần)
        if outer_poly:
            xs = [p[0] for p in outer_poly]
            ys = [p[1] for p in outer_poly]
            max_size = max(max(xs), max(ys))
            SCALE = (WINDOW_SIZE - 50) / max_size
        else:
            SCALE = 8.0 

        current_pos = START_POS
        path_history = [START_POS]
        known_holes = []
        finished = False
        pygame.display.set_caption(f"POMDP Viewer - Map: {os.path.basename(folder_path)}")

    reset_simulation(current_map_idx)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    current_map_idx = (current_map_idx + 1) % len(map_folders)
                    reset_simulation(current_map_idx)
                elif event.key == pygame.K_p:
                    current_map_idx = (current_map_idx - 1) % len(map_folders)
                    reset_simulation(current_map_idx)
                elif event.key == pygame.K_r:
                    reset_simulation(current_map_idx)

        screen.fill(WHITE)

        # 1. DRAW MAP (Vẽ Outer)
        if outer_poly:
            pygame.draw.polygon(screen, OUTER_COLOR, [to_screen(p) for p in outer_poly], 3)
        
        # 2. DRAW HOLES (Vẽ từng hố riêng biệt)
        for hole in real_holes_list:
            if len(hole) < 3: continue # Bỏ qua hố lỗi
            
            # Check if known using reference check or parsing
            is_known = False
            if hole in known_holes: is_known = True
            
            hole_pts = [to_screen(p) for p in hole]
            color = RED if is_known else GHOST_GRAY
            
            # Vẽ vật cản
            pygame.draw.polygon(screen, color, hole_pts)

        pygame.draw.circle(screen, GREEN, to_screen(START_POS), 8)
        pygame.draw.circle(screen, BLUE, to_screen(GOAL_POS), 10)

        # 3. ROBOT LOGIC
        planner = None
        if not finished and outer_poly:
            if dist(current_pos, GOAL_POS) < 2.0:
                finished = True
                print("Goal Reached!")
            else:
                planner = MCPP_Planner(current_pos, GOAL_POS, outer_poly, known_holes)
                proposed_next_pos = planner.search() 
                
                # Check Collision Logic
                actual_collision = False
                hit_hole = None
                
                if not point_in_polygon(proposed_next_pos, outer_poly):
                    actual_collision = True
                
                if not actual_collision:
                    # Kiểm tra va chạm với từng hố
                    for hole in real_holes_list:
                        if point_in_polygon(proposed_next_pos, hole):
                            actual_collision = True
                            hit_hole = hole
                            break
                
                if actual_collision:
                    if hit_hole and (hit_hole not in known_holes):
                        known_holes.append(hit_hole)
                        collision_effect = 10
                        print("Hit hidden obstacle!")
                    
                    retreat_vec = (path_history[-2] - current_pos) if len(path_history) > 1 else np.array([0,0])
                    if np.linalg.norm(retreat_vec) > 0:
                        retreat_vec = retreat_vec / np.linalg.norm(retreat_vec) * 1.5
                        current_pos = current_pos + retreat_vec
                else:
                    current_pos = proposed_next_pos
                    path_history.append(current_pos)

        # VISUALS
        if planner and planner.root: draw_tree(screen, planner.root)
        if len(path_history) > 1:
            pygame.draw.lines(screen, BLACK, False, [to_screen(p) for p in path_history], 3)
        pygame.draw.circle(screen, BLACK, to_screen(current_pos), 6)

        if collision_effect > 0:
            s = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            s.set_alpha(50); s.fill(ORANGE)
            screen.blit(s, (0,0)); collision_effect -= 1

        # Text
        map_name = os.path.basename(map_folders[current_map_idx])
        screen.blit(font_large.render(f"Map [{current_map_idx+1}/{len(map_folders)}]: {map_name}", True, BLACK), (10, 10))
        screen.blit(font.render("Press 'N': Next | 'P': Prev | 'R': Replay", True, BLUE), (10, 40))
        status = "FINISHED" if finished else "RUNNING"
        screen.blit(font.render(f"Status: {status}", True, GREEN if finished else BLACK), (10, 65))

        pygame.display.flip()
        clock.tick(FPS) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()