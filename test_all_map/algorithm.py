import numpy as np
import math
import sys
import os
import glob
import time
import csv
import datetime
import random
import pygame
# --- CẤU HÌNH ---
DATASET_DIR = "AC300"
RESULT_DIR = "Results"
WINDOW_SIZE = 900
FPS = 60  # Tốc độ khung hình cao để RRT chạy mượt
CSV_FILE = os.path.join(RESULT_DIR, "benchmark_comparison.csv")

# --- CẤU HÌNH BỔ SUNG ---
AUTO_NEXT = False       # True: Tự động chuyển map khi xong. False: Chờ bấm phím 'N'
RESULT_DIR = "Results" # Thư mục lưu file kết quả

# --- THÔNG SỐ THUẬT TOÁN ---
# 1. RRT
RRT_STEP_LEN = 5.0      # Độ dài mỗi nhánh cây
RRT_MAX_ITER = 1000     # Số lần thử tối đa mỗi lần Re-plan
RRT_GOAL_SAMPLE_RATE = 0.1

# 2. MCPP
MCPP_EPSILON = 5.0
MCPP_C = 1.414
MCPP_ITER = 1000
MCPP_DEPTH = 25

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)          # Vật cản đã biết
GHOST_GRAY = (235, 235, 235) # Vật cản ẩn
GREEN = (0, 180, 0)          # Start / MCPP Tree
BLUE = (0, 0, 255)           # Goal / RRT Path
ORANGE = (255, 140, 0)       # Hiệu ứng va chạm
PURPLE = (128, 0, 128)       # RRT Tree

# Start/Goal (Sẽ cập nhật lại theo tỷ lệ map)
START_POS_BASE = np.array([2.0, 2.0])
GOAL_POS_BASE = np.array([100.0, 100.0])


# --- HÀM HÌNH HỌC & LOAD FILE ---
# --- HÀM HÌNH HỌC ---
def dist(a, b): return np.linalg.norm(a - b)

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

# --- LOAD DATA ---
def load_data(folder):
    outer = []; holes = []
    
    # Load Outer
    p_out = os.path.join(folder, "outer_polygon")
    if os.path.exists(p_out):
        with open(p_out, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts)>=2: 
                    try: outer.append((float(parts[0]), float(parts[1])))
                    except: pass
    
    # Load Holes (Multi)
    p_hole = os.path.join(folder, "holes")
    if os.path.exists(p_hole):
        curr = []
        with open(p_hole, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts or 'NaN' in line:
                    if len(curr)>2: holes.append(curr)
                    curr = []
                else:
                    try: curr.append((float(parts[0]), float(parts[1])))
                    except: pass
        if len(curr)>2: holes.append(curr)
        
    return outer, holes

# ==========================================
# THUẬT TOÁN 1: RRT (Rapidly-exploring Random Tree)
# ==========================================
class RRT_Planner:
    class Node:
        def __init__(self, x, y):
            self.x = x; self.y = y; self.parent = None
            
    def __init__(self, start, goal, outer, known_holes, bounds):
        self.start = RRT_Planner.Node(start[0], start[1])
        self.goal = RRT_Planner.Node(goal[0], goal[1])
        self.outer = outer
        self.known_holes = known_holes
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.node_list = []

    def is_collision(self, x, y):
        # Check known obstacles
        pt = (x, y)
        if not point_in_polygon(pt, self.outer): return True
        for h in self.known_holes:
            if point_in_polygon(pt, h): return True
        return False

    def plan(self):
        self.node_list = [self.start]
        for i in range(RRT_MAX_ITER):
            # 1. Sample
            if random.random() < RRT_GOAL_SAMPLE_RATE:
                rnd = self.goal
            else:
                rnd = RRT_Planner.Node(
                    random.uniform(self.min_x, self.max_x),
                    random.uniform(self.min_y, self.max_y)
                )

            # 2. Nearest
            dists = [(node.x - rnd.x)**2 + (node.y - rnd.y)**2 for node in self.node_list]
            nearest_ind = dists.index(min(dists))
            nearest_node = self.node_list[nearest_ind]

            # 3. Steer
            theta = math.atan2(rnd.y - nearest_node.y, rnd.x - nearest_node.x)
            new_node = RRT_Planner.Node(
                nearest_node.x + RRT_STEP_LEN * math.cos(theta),
                nearest_node.y + RRT_STEP_LEN * math.sin(theta)
            )
            new_node.parent = nearest_node

            # 4. Check Collision
            if not self.is_collision(new_node.x, new_node.y):
                self.node_list.append(new_node)
                
                # Check Goal
                dx = new_node.x - self.goal.x
                dy = new_node.y - self.goal.y
                if math.sqrt(dx*dx + dy*dy) <= RRT_STEP_LEN:
                    final_node = RRT_Planner.Node(self.goal.x, self.goal.y)
                    final_node.parent = new_node
                    return self.extract_path(final_node)
        return None

    def extract_path(self, node):
        path = []
        while node:
            path.append(np.array([node.x, node.y]))
            node = node.parent
        return path[::-1] # Reverse

# ==========================================
# THUẬT TOÁN 2: MCPP (Monte-Carlo Path Planning)
# ==========================================
class MCPP_Planner:
    class VNode:
        def __init__(self, state):
            self.state = state; self.N=0; self.children={}
    class QNode:
        def __init__(self, parent, action):
            self.parent=parent; self.action=action; self.n=0; self.Q=0.0; self.child_v=None; self.cum_r=0.0

    def __init__(self, start, goal, outer, known_holes):
        self.root = self.VNode(start)
        self.goal = goal
        self.outer = outer
        self.known_holes = known_holes

    def is_valid(self, pos):
        if not point_in_polygon(pos, self.outer): return False
        for h in self.known_holes:
            if point_in_polygon(pos, h): return False
        return True

    def get_action_ucb(self, v):
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            s = q.Q + MCPP_C * math.sqrt(math.log(max(1, v.N))/q.n)
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        ang = random.uniform(0, 2*math.pi)
        r = random.uniform(1.0, MCPP_EPSILON)
        nxt = v.state + np.array([r*math.cos(ang), r*math.sin(ang)])
        if not self.is_valid(nxt): return None
        act = tuple(nxt)
        if act not in v.children: v.children[act] = self.QNode(v, act)
        return act

    def sim_v(self, v, d):
        if d==0 or dist(v.state, self.goal)<2.0: return -dist(v.state, self.goal)
        if len(v.children)<8:
            act = self.expand(v)
            if act: return self.sim_q(v.children[act], d)
        if not v.children: return -dist(v.state, self.goal)
        _, q = self.get_action_ucb(v)
        return self.sim_q(q, d)

    def sim_q(self, q, d):
        if not q.child_v: q.child_v = self.VNode(np.array(q.action)); return -dist(q.child_v.state, self.goal)
        r = -dist(q.parent.state, np.array(q.action)) + self.sim_v(q.child_v, d-1) # Gamma=1
        q.n+=1; q.cum_r+=r; q.Q=q.cum_r/q.n
        q.parent.N+=1
        return r

    def search(self):
        for _ in range(MCPP_ITER): self.sim_v(self.root, MCPP_DEPTH)
        if not self.root.children: return None
        best = max(self.root.children.items(), key=lambda i:i[1].Q)[0]
        return np.array(best)
def get_valid_goal_pos(outer_poly, holes):
    """
    Tìm một vị trí đích hợp lệ:
    1. Nằm TRONG outer_polygon
    2. Nằm NGOÀI tất cả các holes
    """
    if not outer_poly:
        return np.array([90.0, 90.0]) # Fallback

    # Tìm giới hạn bản đồ
    xs = [p[0] for p in outer_poly]
    ys = [p[1] for p in outer_poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Ưu tiên 1: Thử góc trên bên phải (cách biên 10%)
    # Đây là vị trí thường dùng để test path planning đường dài
    candidate = np.array([
        min_x + (max_x - min_x) * 0.9,
        min_y + (max_y - min_y) * 0.9
    ])

    # Hàm kiểm tra điểm có hợp lệ không
    def is_valid(pos):
        if not point_in_polygon(pos, outer_poly): return False
        for hole in holes:
            if point_in_polygon(pos, hole): return False
        return True

    if is_valid(candidate):
        return candidate

    # Ưu tiên 2: Nếu góc phải bị chặn, thử Random trong vùng bounding box
    # Thử tối đa 100 lần để tìm điểm hợp lệ
    for _ in range(100):
        rx = random.uniform(min_x + 5, max_x - 5)
        ry = random.uniform(min_y + 5, max_y - 5)
        candidate = np.array([rx, ry])
        if is_valid(candidate):
            return candidate
            
    # Nếu xui quá không tìm được (map quá đặc), trả về giữa bản đồ
    return np.array([(min_x+max_x)/2, (min_y+max_y)/2])   
# ==========================================
# MAIN APP
# ==========================================
    
# def main():
#     global SCALE 
#     pygame.init()
    
#     # Setup Màn hình
#     screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
#     pygame.display.set_caption("Algorithm Arena: RRT vs MCPP")
#     clock = pygame.time.Clock()
#     font = pygame.font.SysFont("Consolas", 16)
#     font_big = pygame.font.SysFont("Consolas", 24)

#     # Load danh sách map
#     map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC11_*")))
#     if not map_folders:
#         print(f"Lỗi: Không tìm thấy map trong {DATASET_DIR}")
#         return

#     # --- STATE VARIABLES ---
#     current_map_idx = 0
#     algo_mode = "MCPP" 
    
#     # Simulation Data
#     outer_poly = []; real_holes = []
    
#     # Robot State
#     current_pos = np.array([0.0, 0.0])
#     goal_pos = np.array([0.0, 0.0]) # Goal sẽ được lưu giữ giữa các lần switch algo
#     known_holes = []
#     path_history = []
    
#     # Algo Specific
#     rrt_path = []         
#     rrt_planner_vis = None 
#     mcpp_planner_vis = None 
    
#     # Stats
#     start_time = 0
#     elapsed_time = 0
#     total_dist = 0
#     finished = False
    
#     # --- RESET FUNCTION CẢI TIẾN ---
#     def reset_simulation(new_map=False):
#         """
#         new_map=True:  Load map mới, tạo Start/Goal mới.
#         new_map=False: Giữ nguyên Map và Goal cũ, chỉ reset vị trí Robot về Start.
#         """
#         global SCALE
#         nonlocal outer_poly, real_holes, current_pos, goal_pos, known_holes
#         nonlocal path_history, rrt_path, rrt_planner_vis, mcpp_planner_vis
#         nonlocal start_time, elapsed_time, total_dist, finished

#         # 1. Load Data (Luôn load lại để đảm bảo biến sạch, nhưng Goal xử lý sau)
#         folder = map_folders[current_map_idx]
#         if new_map: 
#             print(f"Loading Map: {folder}")
#         outer_poly, real_holes = load_data(folder)

#         # 2. Calc Scale & Start Pos
#         if outer_poly:
#             xs = [p[0] for p in outer_poly]
#             ys = [p[1] for p in outer_poly]
#             max_dim = max(max(xs), max(ys))
#             SCALE = (WINDOW_SIZE - 80) / max_dim 
            
#             # Start: Luôn đặt lại về góc dưới trái
#             min_x, min_y = min(xs), min(ys)
#             current_pos = np.array([min_x + 2.0, min_y + 2.0])
            
#             # 3. XỬ LÝ GOAL (Quan trọng!)
#             # Chỉ tạo Goal mới nếu là Map mới hoặc Goal chưa từng được tạo (lần đầu chạy)
#             if new_map or (goal_pos[0] == 0 and goal_pos[1] == 0):
#                 goal_pos = get_valid_goal_pos(outer_poly, real_holes)
#             # Nếu new_map=False (tức là Replay hoặc Switch Algo), goal_pos giữ nguyên giá trị cũ
#         else:
#             SCALE = 1.0
#             current_pos = np.array([2.0, 2.0])
#             goal_pos = np.array([90.0, 90.0])

#         # 4. Reset State
#         known_holes = []
#         path_history = [current_pos]
#         rrt_path = []
#         rrt_planner_vis = None
#         mcpp_planner_vis = None
        
#         finished = False
#         start_time = time.time()
#         elapsed_time = 0
#         total_dist = 0

#     # Khởi chạy lần đầu (new_map=True để tạo Goal)
#     reset_simulation(new_map=True)

#     # Hàm phụ trợ vẽ
#     def to_scr(pos):
#         return int(pos[0] * SCALE) + 40, int(WINDOW_SIZE - (pos[1] * SCALE)) - 40

#     def draw_mcpp_tree_rec(v, p_pos=None):
#         if not v: return
#         c_pos = to_scr(v.state)
#         if p_pos:
#             pygame.draw.line(screen, (50, 200, 50), p_pos, c_pos, 1)
#         if v.N > 2:
#             for _, q in v.children.items():
#                 if q.child_v: draw_mcpp_tree_rec(q.child_v, c_pos)

#     # --- GAME LOOP ---
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT: running = False
#             elif event.type == pygame.KEYDOWN:
#                 # N: Next Map -> Cần tạo Goal mới -> new_map=True
#                 if event.key == pygame.K_n: 
#                     current_map_idx = (current_map_idx + 1) % len(map_folders)
#                     reset_simulation(new_map=True)
                
#                 # P: Prev Map -> Cần tạo Goal mới -> new_map=True
#                 elif event.key == pygame.K_p: 
#                     current_map_idx = (current_map_idx - 1) % len(map_folders)
#                     reset_simulation(new_map=True)
                
#                 # R: Replay -> Giữ Goal cũ -> new_map=False
#                 elif event.key == pygame.K_r: 
#                     reset_simulation(new_map=False)
                
#                 # TAB: Switch Algo -> Giữ Goal cũ -> new_map=False
#                 elif event.key == pygame.K_TAB: 
#                     algo_mode = "RRT" if algo_mode == "MCPP" else "MCPP"
#                     reset_simulation(new_map=False) 

#         # 2. LOGIC UPDATE
#         if not finished and outer_poly:
#             elapsed_time = time.time() - start_time
            
#             if dist(current_pos, goal_pos) < 3.0:
#                 finished = True
#                 print(f"DONE! {algo_mode} - Time: {elapsed_time:.2f}s")
#             else:
#                 next_pos = None
                
#                 # === MCPP LOGIC ===
#                 if algo_mode == "MCPP":
#                     planner = MCPP_Planner(current_pos, goal_pos, outer_poly, known_holes)
#                     next_pos = planner.search()
#                     mcpp_planner_vis = planner 

#                 # === RRT LOGIC ===
#                 elif algo_mode == "RRT":
#                     if not rrt_path:
#                         bounds = [0, WINDOW_SIZE/SCALE, 0, WINDOW_SIZE/SCALE]
#                         planner = RRT_Planner(current_pos, goal_pos, outer_poly, known_holes, bounds)
#                         full_path = planner.plan()
#                         rrt_planner_vis = planner
#                         if full_path and len(full_path) > 1:
#                             rrt_path = full_path[1:] 
                    
#                     if rrt_path:
#                         target = rrt_path[0]
#                         vec = target - current_pos
#                         d = np.linalg.norm(vec)
#                         step_val = 2.0 
#                         if d > step_val:
#                             next_pos = current_pos + vec/d * step_val
#                         else:
#                             next_pos = target
#                             rrt_path.pop(0)

#                 # === EXECUTION ===
#                 if next_pos is not None:
#                     collided = False
#                     hit_obj = None
                    
#                     if not point_in_polygon(next_pos, outer_poly): collided = True
#                     if not collided:
#                         for h in real_holes:
#                             if point_in_polygon(next_pos, h):
#                                 collided = True; hit_obj = h; break
                    
#                     if collided:
#                         if hit_obj and (hit_obj not in known_holes):
#                             known_holes.append(hit_obj)
                        
#                         if algo_mode == "RRT":
#                             rrt_path = [] 
                        
#                         if len(path_history) > 1:
#                             vec_back = path_history[-2] - current_pos
#                             if np.linalg.norm(vec_back) > 0:
#                                 vec_back = vec_back / np.linalg.norm(vec_back)
#                                 current_pos += vec_back * 1.5
#                     else:
#                         total_dist += dist(current_pos, next_pos)
#                         current_pos = next_pos
#                         path_history.append(current_pos)

#         # 3. RENDERING
#         screen.fill(WHITE)
#         if outer_poly:
#             pygame.draw.polygon(screen, (50, 50, 50), [to_scr(p) for p in outer_poly], 2)
#         for h in real_holes:
#             col = RED if h in known_holes else GHOST_GRAY
#             pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

#         if algo_mode == "MCPP" and mcpp_planner_vis:
#             draw_mcpp_tree_rec(mcpp_planner_vis.root)
#         elif algo_mode == "RRT" and rrt_planner_vis:
#             for node in rrt_planner_vis.node_list:
#                 if node.parent:
#                     pygame.draw.line(screen, (200, 200, 255), 
#                                      to_scr((node.x, node.y)), to_scr((node.parent.x, node.parent.y)), 1)
#             if rrt_path:
#                 pts = [to_scr(current_pos)] + [to_scr(p) for p in rrt_path]
#                 if len(pts) > 1: pygame.draw.lines(screen, BLUE, False, pts, 2)

#         pygame.draw.circle(screen, GREEN, to_scr(path_history[0]), 6)
#         pygame.draw.circle(screen, BLUE, to_scr(goal_pos), 8) # Goal vẽ tại vị trí cố định

#         if len(path_history) > 1:
#             pygame.draw.lines(screen, BLACK, False, [to_scr(p) for p in path_history], 3)
#         pygame.draw.circle(screen, BLACK, to_scr(current_pos), 6)

#         # GUI
#         pygame.draw.rect(screen, (245, 245, 245), (10, 10, 320, 130))
#         pygame.draw.rect(screen, BLACK, (10, 10, 320, 130), 2)
#         map_name = os.path.basename(map_folders[current_map_idx])
#         screen.blit(font_big.render(f"Map: {map_name}", True, BLACK), (20, 20))
#         screen.blit(font.render(f"Mode: {algo_mode}", True, BLUE if algo_mode=="RRT" else GREEN), (20, 50))
#         screen.blit(font.render(f"Time: {elapsed_time:.2f} s", True, BLACK), (20, 70))
#         screen.blit(font.render(f"Dist: {total_dist:.2f} m", True, BLACK), (20, 90))
#         status_col = GREEN if finished else RED
#         status_txt = "FINISHED" if finished else "RUNNING"
#         screen.blit(font.render(f"Status: {status_txt}", True, status_col), (20, 110))
#         help_txt = "[TAB]: Switch | [N]: Next | [R]: Reset"
#         screen.blit(font.render(help_txt, True, (100, 100, 100)), (20, WINDOW_SIZE - 30))

#         pygame.display.flip()
#         clock.tick(FPS)

#     pygame.quit()
#     sys.exit()
    
# --- HÀM MAIN ĐÃ NÂNG CẤP ---
def main():
    global SCALE 
    pygame.init()
    
    # Tạo thư mục kết quả nếu chưa có
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # Tạo file CSV mới với timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(RESULT_DIR, f"Benchmark_{timestamp}.csv")
    
    # Viết Header cho file CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Map Name", "Algorithm", "Time (s)", "Distance (m)", "Obstacles Found", "Status"])
    
    print(f"--- Đã tạo file kết quả: {csv_filename} ---")

    # Setup Màn hình
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption(f"Arena Benchmark - Saving to {csv_filename}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    font_big = pygame.font.SysFont("Consolas", 24)

    # Load danh sách map
    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC10_*")))
    if not map_folders:
        print(f"Lỗi: Không tìm thấy map trong {DATASET_DIR}")
        return

    # --- STATE VARIABLES ---
    current_map_idx = 0
    algo_mode = "MCPP" 
    
    # Simulation Data
    outer_poly = []; real_holes = []
    
    # Robot State
    current_pos = np.array([0.0, 0.0])
    goal_pos = np.array([0.0, 0.0])
    known_holes = []
    path_history = []
    
    # Algo Specific
    rrt_path = []         
    rrt_planner_vis = None 
    mcpp_planner_vis = None 
    
    # Stats
    start_time = 0
    elapsed_time = 0
    total_dist = 0
    finished = False
    data_saved = False # Cờ để đảm bảo chỉ lưu 1 lần
    
    # Timer cho Auto Next
    finish_cooldown = 0 

    # --- RESET FUNCTION ---
    def reset_simulation(new_map=False):
        global SCALE
        nonlocal outer_poly, real_holes, current_pos, goal_pos, known_holes
        nonlocal path_history, rrt_path, rrt_planner_vis, mcpp_planner_vis
        nonlocal start_time, elapsed_time, total_dist, finished, data_saved, finish_cooldown

        # 1. Load Data
        folder = map_folders[current_map_idx]
        if new_map: print(f"Loading Map: {folder}")
        outer_poly, real_holes = load_data(folder)

        # 2. Calc Scale & Start Pos
        if outer_poly:
            xs = [p[0] for p in outer_poly]
            ys = [p[1] for p in outer_poly]
            max_dim = max(max(xs), max(ys))
            SCALE = (WINDOW_SIZE - 80) / max_dim 
            min_x, min_y = min(xs), min(ys)
            current_pos = np.array([min_x + 2.0, min_y + 2.0])
            
            if new_map or (goal_pos[0] == 0 and goal_pos[1] == 0):
                goal_pos = get_valid_goal_pos(outer_poly, real_holes)
        else:
            SCALE = 1.0
            current_pos = np.array([2.0, 2.0])
            goal_pos = np.array([90.0, 90.0])

        # 3. Reset State
        known_holes = []
        path_history = [current_pos]
        rrt_path = []
        rrt_planner_vis = None
        mcpp_planner_vis = None
        
        finished = False
        data_saved = False # Reset cờ lưu
        finish_cooldown = 0
        start_time = time.time()
        elapsed_time = 0
        total_dist = 0

    # Khởi chạy lần đầu
    reset_simulation(new_map=True)

    def to_scr(pos):
        return int(pos[0] * SCALE) + 40, int(WINDOW_SIZE - (pos[1] * SCALE)) - 40

    def draw_mcpp_tree_rec(v, p_pos=None):
        if not v: return
        c_pos = to_scr(v.state)
        if p_pos: pygame.draw.line(screen, (50, 200, 50), p_pos, c_pos, 1)
        if v.N > 2:
            for _, q in v.children.items():
                if q.child_v: draw_mcpp_tree_rec(q.child_v, c_pos)

    # --- HÀM LƯU KẾT QUẢ ---
    def save_results():
        map_name = os.path.basename(map_folders[current_map_idx])
        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                map_name, 
                algo_mode, 
                f"{elapsed_time:.4f}", 
                f"{total_dist:.2f}", 
                len(known_holes), 
                "SUCCESS"
            ])
        print(f">> Saved: {map_name} | {algo_mode} | {elapsed_time:.2f}s")

    # --- GAME LOOP ---
    running = True
    while running:
        dt = clock.tick(FPS) # Delta time

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    current_map_idx = (current_map_idx + 1) % len(map_folders)
                    reset_simulation(new_map=True)
                elif event.key == pygame.K_p: 
                    current_map_idx = (current_map_idx - 1) % len(map_folders)
                    reset_simulation(new_map=True)
                elif event.key == pygame.K_r: 
                    reset_simulation(new_map=False)
                elif event.key == pygame.K_TAB: 
                    algo_mode = "RRT" if algo_mode == "MCPP" else "MCPP"
                    reset_simulation(new_map=False) 

        # 2. LOGIC UPDATE
        if not finished and outer_poly:
            elapsed_time = time.time() - start_time
            
            # KIỂM TRA ĐÍCH
            if dist(current_pos, goal_pos) < 3.0:
                finished = True
                
                # --- LƯU FILE KHI VỀ ĐÍCH ---
                if not data_saved:
                    save_results()
                    data_saved = True
                    finish_cooldown = pygame.time.get_ticks() # Bắt đầu đếm ngược
            else:
                next_pos = None
                
                # ... [GIỮ NGUYÊN LOGIC MCPP/RRT CŨ TỪ ĐÂY] ...
                if algo_mode == "MCPP":
                    planner = MCPP_Planner(current_pos, goal_pos, outer_poly, known_holes)
                    next_pos = planner.search()
                    mcpp_planner_vis = planner 
                elif algo_mode == "RRT":
                    if not rrt_path:
                        bounds = [0, WINDOW_SIZE/SCALE, 0, WINDOW_SIZE/SCALE]
                        planner = RRT_Planner(current_pos, goal_pos, outer_poly, known_holes, bounds)
                        full_path = planner.plan()
                        rrt_planner_vis = planner
                        if full_path and len(full_path) > 1:
                            rrt_path = full_path[1:] 
                    if rrt_path:
                        target = rrt_path[0]
                        vec = target - current_pos
                        d = np.linalg.norm(vec)
                        step_val = 2.0 
                        if d > step_val: next_pos = current_pos + vec/d * step_val
                        else: next_pos = target; rrt_path.pop(0)

                # Collision Check
                if next_pos is not None:
                    collided = False; hit_obj = None
                    if not point_in_polygon(next_pos, outer_poly): collided = True
                    if not collided:
                        for h in real_holes:
                            if point_in_polygon(next_pos, h):
                                collided = True; hit_obj = h; break
                    
                    if collided:
                        if hit_obj and (hit_obj not in known_holes): known_holes.append(hit_obj)
                        if algo_mode == "RRT": rrt_path = [] 
                        if len(path_history) > 1:
                            vec_back = path_history[-2] - current_pos
                            if np.linalg.norm(vec_back) > 0:
                                current_pos += (vec_back / np.linalg.norm(vec_back)) * 1.5
                    else:
                        total_dist += dist(current_pos, next_pos)
                        current_pos = next_pos
                        path_history.append(current_pos)
                # ... [HẾT PHẦN LOGIC CŨ] ...

        # 3. AUTO NEXT LOGIC (Nếu bật)
        if finished and AUTO_NEXT:
            # Chờ 1 giây (1000ms) rồi chuyển
            if pygame.time.get_ticks() - finish_cooldown > 1000:
                # Logic: Chạy MCPP -> Chạy RRT -> Next Map -> Lặp lại
                if algo_mode == "MCPP":
                    algo_mode = "RRT"
                    reset_simulation(new_map=False) # Chạy lại map cũ với RRT
                else:
                    algo_mode = "MCPP"
                    current_map_idx = (current_map_idx + 1) % len(map_folders)
                    reset_simulation(new_map=True) # Qua map mới

        # 4. RENDERING
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50, 50, 50), [to_scr(p) for p in outer_poly], 2)
        for h in real_holes:
            col = RED if h in known_holes else GHOST_GRAY
            pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

        if algo_mode == "MCPP" and mcpp_planner_vis:
            draw_mcpp_tree_rec(mcpp_planner_vis.root)
        elif algo_mode == "RRT" and rrt_planner_vis:
            for node in rrt_planner_vis.node_list:
                if node.parent:
                    pygame.draw.line(screen, (200, 200, 255), to_scr((node.x, node.y)), to_scr((node.parent.x, node.parent.y)), 1)
            if rrt_path:
                pts = [to_scr(current_pos)] + [to_scr(p) for p in rrt_path]
                if len(pts) > 1: pygame.draw.lines(screen, BLUE, False, pts, 2)

        pygame.draw.circle(screen, GREEN, to_scr(path_history[0]), 6)
        pygame.draw.circle(screen, BLUE, to_scr(goal_pos), 8)
        if len(path_history) > 1: pygame.draw.lines(screen, BLACK, False, [to_scr(p) for p in path_history], 3)
        pygame.draw.circle(screen, BLACK, to_scr(current_pos), 6)

        # GUI
        pygame.draw.rect(screen, (245, 245, 245), (10, 10, 350, 140))
        pygame.draw.rect(screen, BLACK, (10, 10, 350, 140), 2)
        map_name = os.path.basename(map_folders[current_map_idx])
        screen.blit(font_big.render(f"Map: {map_name}", True, BLACK), (20, 20))
        screen.blit(font.render(f"Mode: {algo_mode}", True, BLUE if algo_mode=="RRT" else GREEN), (20, 50))
        screen.blit(font.render(f"Time: {elapsed_time:.2f} s", True, BLACK), (20, 70))
        screen.blit(font.render(f"Dist: {total_dist:.2f} m", True, BLACK), (20, 90))
        
        status_txt = "FINISHED - SAVED" if (finished and data_saved) else "RUNNING"
        screen.blit(font.render(f"Status: {status_txt}", True, GREEN if finished else RED), (20, 115))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()
    
if __name__ == "__main__":
    main()