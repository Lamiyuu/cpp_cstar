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
MAGENTA = (255, 0, 255)
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

def check_line_collision(p1, p2, outer, holes):
    """Kiểm tra va chạm đoạn thẳng (cho việc tối ưu)"""
    d = dist(p1, p2)
    if d == 0: return False
    steps = int(d / 1.0) + 1 
    for i in range(steps + 1):
        t = i / steps
        pt = p1 + (p2 - p1) * t
        if not point_in_polygon(pt, outer): return True
        for h in holes:
            if point_in_polygon(pt, h): return True
    return False

def optimize_path(path, outer, holes):
    """Cắt ngắn đường đi (Shortcut Pruning)"""
    if len(path) < 3: return path
    
    optimized = [path[0]]
    current_idx = 0
    
    # Duyệt từ điểm hiện tại, cố gắng nối với điểm xa nhất có thể
    while current_idx < len(path) - 1:
        next_idx = current_idx + 1
        for i in range(len(path) - 1, current_idx + 1, -1):
            if not check_line_collision(path[current_idx], path[i], outer, holes):
                next_idx = i
                break
        optimized.append(path[next_idx])
        current_idx = next_idx
    return optimized
def calculate_path_len(path):
    l = 0
    for i in range(len(path)-1):
        l += dist(path[i], path[i+1])
    return l

# ==========================================
# MAIN APP HOÀN CHỈNH
# ==========================================
# ==========================================
# MAIN APP (SINGLE GOAL)
# ==========================================
def main():
    global SCALE 
    pygame.init()
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    csv_file = os.path.join(RESULT_DIR, f"SingleGoal_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Map", "Algo", "Original Len", "Optimized Len", "Improvement (%)", "Collisions"])

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Single Goal: RRT vs MCPP with Optimization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    font_big = pygame.font.SysFont("Consolas", 24)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC10_*")))
    if not map_folders: 
        print(f"Lỗi: Không tìm thấy map trong {DATASET_DIR}")
        return

    # --- STATE ---
    current_map_idx = 0
    algo_mode = "MCPP"
    
    outer_poly = []; real_holes = []
    current_pos = np.array([0.,0.])
    goal_pos = np.array([0.,0.])
    
    known_holes = []
    path_history = [] 
    optimized_path = [] # Đường đi tối ưu (chỉ có khi finish)
    
    rrt_path = []; rrt_vis = None; mcpp_vis = None
    
    start_time = 0; elapsed_time = 0; collisions_count = 0
    finished = False; data_saved = False; finish_cooldown = 0

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_pos, goal_pos, known_holes, path_history
        nonlocal rrt_path, rrt_vis, mcpp_vis, finished, data_saved, start_time, collisions_count, optimized_path
        global SCALE

        folder = map_folders[current_map_idx]
        if new_map: print(f"Loading Map: {folder}")
        outer_poly, real_holes = load_data(folder)

        if outer_poly:
            xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
            mx = max(max(xs), max(ys)); SCALE = (WINDOW_SIZE - 80) / mx
            min_x, min_y = min(xs), min(ys)
            
            start_pos = np.array([min_x+2.0, min_y+2.0])
            current_pos = start_pos
            
            # Chỉ tạo goal mới nếu là map mới
            if new_map or (goal_pos[0]==0 and goal_pos[1]==0):
                goal_pos = get_valid_goal_pos(outer_poly, real_holes)
        else: SCALE = 1.0; current_pos = np.array([2.,2.]); goal_pos = np.array([90.,90.])

        # Reset Runtime Data
        known_holes = []
        path_history = [current_pos]
        optimized_path = []
        
        rrt_path = []; rrt_vis = None; mcpp_vis = None
        
        finished = False; data_saved = False; finish_cooldown = 0
        collisions_count = 0; start_time = time.time()

    reset_sim(new_map=True)

    def to_scr(pos):
        return int(pos[0]*SCALE)+40, int(WINDOW_SIZE - (pos[1]*SCALE))-40

    def draw_mcpp_rec(v, p_pos=None):
        if not v: return
        c = to_scr(v.state)
        if p_pos: pygame.draw.line(screen, GREEN, p_pos, c, 1)
        if v.N > 2:
            for _, q in v.children.items():
                if q.child_v: draw_mcpp_rec(q.child_v, c)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    current_map_idx = (current_map_idx + 1) % len(map_folders)
                    reset_sim(new_map=True)
                elif event.key == pygame.K_p: 
                    current_map_idx = (current_map_idx - 1) % len(map_folders)
                    reset_sim(new_map=True)
                elif event.key == pygame.K_r: reset_sim(new_map=False)
                elif event.key == pygame.K_TAB: 
                    algo_mode = "RRT" if algo_mode == "MCPP" else "MCPP"
                    reset_sim(new_map=False) 

        # LOGIC UPDATE
        if not finished and outer_poly:
            elapsed_time = time.time() - start_time
            
            # --- 1. KIỂM TRA ĐẾN ĐÍCH ---
            if dist(current_pos, goal_pos) < 3.0:
                finished = True
                
                # CHẠY TỐI ƯU HÓA NGAY LẬP TỨC
                if not optimized_path:
                    print("Goal Reached! Optimizing path...")
                    # Dùng real_holes để tối ưu (Ground Truth optimization)
                    optimized_path = optimize_path(path_history, outer_poly, real_holes)
                    
                    l_old = calculate_path_len(path_history)
                    l_new = calculate_path_len(optimized_path)
                    imp = (l_old - l_new) / l_old * 100 if l_old > 0 else 0
                    
                    if not data_saved:
                        with open(csv_file, 'a', newline='') as f:
                            csv.writer(f).writerow([
                                os.path.basename(map_folders[current_map_idx]),
                                algo_mode, f"{l_old:.2f}", f"{l_new:.2f}", f"{imp:.2f}", collisions_count
                            ])
                        data_saved = True
                        finish_cooldown = pygame.time.get_ticks()

            # --- 2. PLANNING ---
            else:
                next_pos = None
                
                # MCPP
                if algo_mode == "MCPP":
                    planner = MCPP_Planner(current_pos, goal_pos, outer_poly, known_holes)
                    next_pos = planner.search()
                    mcpp_vis = planner 
                
                # RRT
                elif algo_mode == "RRT":
                    if not rrt_path:
                        bounds = [0, WINDOW_SIZE/SCALE, 0, WINDOW_SIZE/SCALE]
                        planner = RRT_Planner(current_pos, goal_pos, outer_poly, known_holes, bounds)
                        full_path = planner.plan()
                        rrt_vis = planner
                        if full_path and len(full_path)>1: rrt_path = full_path[1:] 
                    if rrt_path:
                        tgt = rrt_path[0]; vec = tgt-current_pos; d = np.linalg.norm(vec)
                        step = 2.0 
                        if d > step: next_pos = current_pos + vec/d*step
                        else: next_pos = tgt; rrt_path.pop(0)

                # --- 3. COLLISION & MOVE ---
                if next_pos is not None:
                    col = False; hit = None
                    if not point_in_polygon(next_pos, outer_poly): col = True
                    if not col:
                        for h in real_holes:
                            if point_in_polygon(next_pos, h): col=True; hit=h; break
                    
                    if col:
                        collisions_count += 1
                        if hit and hit not in known_holes: known_holes.append(hit)
                        if algo_mode == "RRT": rrt_path = [] 
                        if len(path_history)>1:
                            back = path_history[-2]-current_pos
                            if np.linalg.norm(back)>0: current_pos += back/np.linalg.norm(back)*1.5
                    else:
                        current_pos = next_pos
                        path_history.append(current_pos)

        # AUTO NEXT
        if finished and AUTO_NEXT:
            if pygame.time.get_ticks() - finish_cooldown > 1000:
                if algo_mode == "MCPP": algo_mode = "RRT"; reset_sim(new_map=False)
                else: algo_mode = "MCPP"; current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim(new_map=True)

        # RENDERING
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        for h in real_holes:
            pygame.draw.polygon(screen, RED if h in known_holes else GHOST_GRAY, [to_scr(p) for p in h])

        # Draw Trees
        if algo_mode == "MCPP" and mcpp_vis and not finished: draw_mcpp_rec(mcpp_vis.root)
        elif algo_mode == "RRT" and rrt_vis and not finished:
            for n in rrt_vis.node_list:
                if n.parent: pygame.draw.line(screen, (200, 200, 255), to_scr((n.x, n.y)), to_scr((n.parent.x, n.parent.y)), 1)
            if rrt_path:
                pts = [to_scr(current_pos)] + [to_scr(p) for p in rrt_path]
                if len(pts)>1: pygame.draw.lines(screen, BLUE, False, pts, 2)

        # Draw Goal
        pygame.draw.circle(screen, BLUE, to_scr(goal_pos), 8)
        
        # DRAW PATHS
        # 1. Đường thô (Đen -> Xám khi xong)
        if len(path_history)>1:
            col = (180,180,180) if finished else BLACK
            pygame.draw.lines(screen, col, False, [to_scr(p) for p in path_history], 3)
        
        # 2. ĐƯỜNG TỐI ƯU (Tím Đậm)
        if finished and len(optimized_path)>1:
            pygame.draw.lines(screen, MAGENTA, False, [to_scr(p) for p in optimized_path], 4)
            for p in optimized_path: pygame.draw.circle(screen, MAGENTA, to_scr(p), 4)

        pygame.draw.circle(screen, BLACK, to_scr(current_pos), 6)

        # GUI
        mname = os.path.basename(map_folders[current_map_idx])
        screen.blit(font_big.render(f"Map: {mname}", True, BLACK), (10, 10))
        screen.blit(font.render(f"Mode: {algo_mode}", True, BLUE), (10, 40))
        
        if finished:
            l_old = calculate_path_len(path_history)
            l_new = calculate_path_len(optimized_path)
            imp = (l_old - l_new) / l_old * 100 if l_old > 0 else 0
            screen.blit(font.render(f"Raw: {l_old:.1f}m | Opt: {l_new:.1f}m", True, BLACK), (10, 70))
            screen.blit(font.render(f"Better: {imp:.1f}%", True, GREEN), (10, 90))
            screen.blit(font.render("DONE - PRESS 'N'", True, RED), (10, 120))
        else:
            screen.blit(font.render("RUNNING...", True, GREEN), (10, 70))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()