import numpy as np
import math
import sys
import os
import glob
import time
import csv
import random
import pygame

# ==========================================
# 1. CẤU HÌNH & THAM SỐ
# ==========================================
WINDOW_SIZE = 900
FPS = 60
DATASET_DIR = "AC300"
RESULT_DIR = "Results"

# --- THÔNG SỐ XE (KINEMATIC) ---
CAR_L = 4.0           # Chiều dài cơ sở
CAR_WIDTH = 2.5        # Chiều rộng xe
MAX_STEER = 0.6        # Góc lái tối đa (~35 độ)
VELOCITY = 5.0        # Vận tốc xe
DT = 0.2               # Bước thời gian
SIM_TIME = 1.0         # Thời gian xe chạy cho 1 bước

# --- THÔNG SỐ ĐÍCH ĐẾN ---
GOAL_RADIUS = 1.5     # Bán kính vùng đích

# --- CẢM BIẾN & TÁI QUY HOẠCH ---
LOOKAHEAD_STEPS = 5   # Nhìn trước 10 bước

# --- THAM SỐ THUẬT TOÁN ---
# 1. RRT
RRT_MAX_ITER = 2000
RRT_GOAL_PROB = 0.3    

# 2. MCPP (THÔNG SỐ MỚI)
MCPP_EPSILON = 5.0     # (Dùng để giới hạn không gian hoặc reward)
MCPP_C = 1.414         # Hằng số cân bằng Explore/Exploit (UCB)
MCPP_ITER = 1000       # Số lần duyệt tối đa
MCPP_DEPTH = 30        # Độ sâu mô phỏng (Rollout)
MCPP_BRANCHES = 5
# MÀU SẮC
WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (255, 0, 0)
GRAY = (200, 200, 200); GREEN = (0, 200, 0); BLUE = (0, 0, 255)
CAR_COLOR = (50, 50, 200); PATH_COLOR = (50, 50, 50)
LOOKAHEAD_COLOR = (255, 140, 0) 
GHOST_GRAY = (240, 240, 240)

# ==========================================
# 2. HÀM HỖ TRỢ HÌNH HỌC
# ==========================================
def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def normalize_angle(angle):
    while angle > math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    if n == 0: return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def load_data(folder):
    outer = []; holes = []
    if not os.path.exists(folder): return [], []
    
    p_out = os.path.join(folder, "outer_polygon")
    if os.path.exists(p_out):
        with open(p_out, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts)>=2: 
                    try: outer.append((float(parts[0]), float(parts[1])))
                    except: pass
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

def get_valid_random_pos(outer, holes, bounds):
    min_x, max_x, min_y, max_y = bounds
    for _ in range(100):
        rx = random.uniform(min_x, max_x)
        ry = random.uniform(min_y, max_y)
        if not point_in_polygon((rx, ry), outer): continue
        valid = True
        for h in holes:
            if point_in_polygon((rx, ry), h):
                valid = False; break
        if valid: return np.array([rx, ry])
    return np.array([50, 50])

# ==========================================
# 3. MÔ HÌNH XE & VA CHẠM
# ==========================================
def get_car_corners(x, y, yaw):
    rear_overhang = 2.0; front_overhang = 2.0
    local_corners = [
        (-rear_overhang, CAR_WIDTH/2), (-rear_overhang, -CAR_WIDTH/2),
        (CAR_L + front_overhang, -CAR_WIDTH/2), (CAR_L + front_overhang, CAR_WIDTH/2)
    ]
    world_corners = []
    c, s = math.cos(yaw), math.sin(yaw)
    for lx, ly in local_corners:
        wx = lx * c - ly * s + x
        wy = lx * s + ly * c + y
        world_corners.append((wx, wy))
    return world_corners

def check_collision_with_index(x, y, yaw, outer, holes):
    corners = get_car_corners(x, y, yaw)
    for p in corners:
        if outer and not point_in_polygon(p, outer): return True, -1
        for i, h in enumerate(holes):
            if point_in_polygon(p, h): return True, i
    return False, -2

def check_path_collision(path_x, path_y, path_yaw, outer, holes):
    step = 2 
    for i in range(0, len(path_x), step):
        collided, _ = check_collision_with_index(path_x[i], path_y[i], path_yaw[i], outer, holes)
        if collided: return True
    return False

def simulate_step(x, y, yaw, steer, sim_time=SIM_TIME):
    """Mô phỏng xe chạy với góc lái steer trong sim_time"""
    path_x, path_y, path_yaw = [x], [y], [yaw]
    steps = int(sim_time / DT)
    for _ in range(steps):
        # Vật lý xe (Bicycle Model)
        x += VELOCITY * math.cos(yaw) * DT
        y += VELOCITY * math.sin(yaw) * DT
        yaw += (VELOCITY / CAR_L) * math.tan(steer) * DT
        yaw = normalize_angle(yaw)
        
        path_x.append(x); path_y.append(y); path_yaw.append(yaw)
    return x, y, yaw, path_x, path_y, path_yaw

# ==========================================
# 4. THUẬT TOÁN TÌM ĐƯỜNG
# ==========================================
class Node:
    def __init__(self, x, y, yaw, parent=None):
        self.x = x; self.y = y; self.yaw = yaw
        self.parent = parent
        self.path_x = []; self.path_y = []; self.path_yaw = []

# --- A. KINEMATIC RRT ---
class KinematicRRT:
    def __init__(self, start, goal, outer, known_holes, bounds):
        self.start = Node(*start)
        self.goal = goal
        self.outer = outer; self.known_holes = known_holes 
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.node_list = [self.start]

    def plan_step(self):
        if random.random() < RRT_GOAL_PROB: rnd = (self.goal[0], self.goal[1])
        else: rnd = (random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))
        
        dists = [(node.x - rnd[0])**2 + (node.y - rnd[1])**2 for node in self.node_list]
        nearest = self.node_list[dists.index(min(dists))]
        
        dx = rnd[0] - nearest.x; dy = rnd[1] - nearest.y
        target_yaw = math.atan2(dy, dx)
        diff = normalize_angle(target_yaw - nearest.yaw)
        steer = max(-MAX_STEER, min(MAX_STEER, diff))
        
        nx, ny, nyaw, px, py, pyaw = simulate_step(nearest.x, nearest.y, nearest.yaw, steer)
        
        if not check_path_collision(px, py, pyaw, self.outer, self.known_holes):
            new_node = Node(nx, ny, nyaw, nearest)
            new_node.path_x = px; new_node.path_y = py; new_node.path_yaw = pyaw
            self.node_list.append(new_node)
            
            # Check Goal (Đầu xe)
            fx = nx + CAR_L * math.cos(nyaw)
            fy = ny + CAR_L * math.sin(nyaw)
            if dist((fx, fy), self.goal) < GOAL_RADIUS:
                return self.extract_path(new_node)
        return None

    def extract_path(self, node):
        full_path = []
        while node.parent:
            segment = list(zip(node.path_x, node.path_y, node.path_yaw))
            full_path = segment + full_path
            node = node.parent
        return full_path

# --- B. KINEMATIC MCPP (Monte Carlo Tree Search) ---
class KinematicMCPP:
    class VNode: # Node trạng thái
        def __init__(self, state):
            self.state = state # (x, y, yaw)
            self.N = 0 # Số lần thăm
            self.children = {} # Map[action] -> QNode
            # Để vẽ
            self.parent_node = None 
            self.path_x = []; self.path_y = []

    class QNode: # Node hành động
        def __init__(self, parent, action):
            self.parent = parent
            self.action = action # steer
            self.n = 0 # Số lần chọn
            self.Q = 0.0 # Giá trị trung bình
            self.child_v = None # Node con (kết quả)

    def __init__(self, start, goal, outer, known_holes):
        self.root = self.VNode(start)
        self.goal = goal
        self.outer = outer; self.known_holes = known_holes
        self.node_list = [self.root] # Để vẽ cây

    def get_action_ucb(self, v):
        # Chọn hành động theo công thức UCB1
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            # UCB = Q + C * sqrt(ln(N)/n)
            s = q.Q + MCPP_C * math.sqrt(math.log(max(1, v.N)) / q.n)
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        # Thử một hành động lái ngẫu nhiên
        steer = random.uniform(-MAX_STEER, MAX_STEER)
        
        # Mô phỏng kinematic
        nx, ny, nyaw, px, py, pyaw = simulate_step(v.state[0], v.state[1], v.state[2], steer)
        
        # Check va chạm
        if check_path_collision(px, py, pyaw, self.outer, self.known_holes):
            return None # Hành động không hợp lệ
            
        # Thêm vào cây
        action_key = round(steer, 2) # Làm tròn để gom nhóm
        if action_key not in v.children:
            qnode = self.QNode(v, action_key)
            v.children[action_key] = qnode
            return action_key
        return None

    def sim_v(self, v, d):
        # 1. Check Goal (Đầu xe)
        fx = v.state[0] + CAR_L * math.cos(v.state[2])
        fy = v.state[1] + CAR_L * math.sin(v.state[2])
        dist_to_goal = dist((fx, fy), self.goal)
        
        if d == 0 or dist_to_goal < GOAL_RADIUS:
            # Reward: Càng gần đích càng tốt (Reward âm)
            return -dist_to_goal 

        # 2. Expand (Mở rộng cây)
        if len(v.children) < MCPP_BRANCHES:
            act = self.expand(v)
            if act is not None:
                return self.sim_q(v.children[act], d)
        
        if not v.children: 
            return -dist_to_goal # Kẹt

        # 3. Select (Chọn nhánh tốt nhất)
        _, q = self.get_action_ucb(v)
        return self.sim_q(q, d)

    def sim_q(self, q, d):
        if not q.child_v:
            # Thực thi hành động -> Tạo trạng thái mới
            nx, ny, nyaw, px, py, pyaw = simulate_step(q.parent.state[0], q.parent.state[1], q.parent.state[2], q.action)
            q.child_v = self.VNode((nx, ny, nyaw))
            
            # Lưu thông tin để vẽ cây
            q.child_v.parent_node = q.parent
            q.child_v.path_x = px
            q.child_v.path_y = py
            self.node_list.append(q.child_v)
            
            # 4. Simulation (Rollout ngẫu nhiên)
            curr_state = (nx, ny, nyaw)
            for _ in range(5): # Đi thêm 5 bước ngẫu nhiên
                rnd_steer = random.uniform(-MAX_STEER, MAX_STEER)
                rx, ry, ryaw, rpx, rpy, rpyaw = simulate_step(curr_state[0], curr_state[1], curr_state[2], rnd_steer)
                if check_path_collision(rpx, rpy, rpyaw, self.outer, self.known_holes): break
                curr_state = (rx, ry, ryaw)
            
            # Tính reward tại điểm cuối của Rollout
            r_fx = curr_state[0] + CAR_L * math.cos(curr_state[2])
            r_fy = curr_state[1] + CAR_L * math.sin(curr_state[2])
            return -dist((r_fx, r_fy), self.goal)

        # 5. Backpropagation (Lan truyền ngược)
        r = self.sim_v(q.child_v, d - 1)
        q.n += 1
        q.Q += (r - q.Q) / q.n # Cập nhật trung bình động
        q.parent.N += 1
        return r

    def plan_step(self):
        # Chạy 1 lần MCTS (Duyệt từ gốc)
        self.sim_v(self.root, MCPP_DEPTH)
        
        # Kiểm tra xem trong cây đã có node nào tới đích chưa
        # (Duyệt node_list để tìm)
        for node in self.node_list:
            fx = node.state[0] + CAR_L * math.cos(node.state[2])
            fy = node.state[1] + CAR_L * math.sin(node.state[2])
            if dist((fx, fy), self.goal) < GOAL_RADIUS:
                return self.extract_path(node)
        return None

    def extract_path(self, node):
        full_path = []
        while node.parent_node:
            segment = list(zip(node.path_x, node.path_y, [node.state[2]]*len(node.path_x))) # Yaw xấp xỉ
            full_path = segment + full_path
            node = node.parent_node
        return full_path

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    pygame.init()
    if not os.path.exists("Results"): os.makedirs("Results")
    
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Kinematic Car: RRT vs MCTS (Expanded)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC11_*")))
    if not map_folders: print(f"No maps found in {DATASET_DIR}!"); return

    # --- STATE ---
    current_map_idx = 0
    algo_mode = "RRT"
    outer_poly = []; real_holes = []
    
    # Dùng set để lưu index các hố đã biết
    known_hole_indices = set() 
    planner_holes_geom = [] 
    
    current_state = (0, 0, 0)
    goal_pos = np.array([0, 0])
    
    planner = None
    planned_path = []
    path_index = 0
    is_planning = True
    path_history = []
    
    scale = 1.0

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_state, goal_pos, known_hole_indices, planner_holes_geom, path_history
        nonlocal planner, planned_path, path_index, is_planning, scale

        folder = map_folders[current_map_idx]
        if new_map: print(f"Loading: {folder}")
        outer_poly, real_holes = load_data(folder)

        if outer_poly:
            xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
            mx = max(max(xs), max(ys))
            scale = (WINDOW_SIZE - 80) / mx
            min_x, min_y = min(xs), min(ys)
            
            start_pos = np.array([min_x+10.0, min_y+10.0])
            current_state = (start_pos[0], start_pos[1], 0.0)
            
            if new_map or (goal_pos[0]==0):
                goal_pos = get_valid_random_pos(outer_poly, real_holes, [min_x, max(xs), min_y, max(ys)])
            
            bounds = [0, mx, 0, mx]
            
            # Init Planner
            if algo_mode == "RRT":
                planner = KinematicRRT(current_state, goal_pos, outer_poly, planner_holes_geom, bounds)
            else:
                planner = KinematicMCPP(current_state, goal_pos, outer_poly, planner_holes_geom)
        else: scale = 1.0
        
        if new_map: 
            known_hole_indices = set()
            planner_holes_geom = []
            
        planned_path = []; path_index = 0
        path_history = []
        is_planning = True

    reset_sim(new_map=True)

    def to_scr(pos):
        return int(pos[0]*scale)+40, int(WINDOW_SIZE - (pos[1]*scale)-40)

    def draw_car(state, color=CAR_COLOR):
        x, y, yaw = state
        corners = get_car_corners(x, y, yaw)
        scr_corners = [to_scr(p) for p in corners]
        pygame.draw.polygon(screen, color, scr_corners)
        pygame.draw.polygon(screen, BLACK, scr_corners, 1)
        
        fx = x + CAR_L * math.cos(yaw); fy = y + CAR_L * math.sin(yaw)
        f_scr = to_scr((fx, fy)); r_scr = to_scr((x, y))
        
        # Chấm Đen (Trục Sau)
        pygame.draw.circle(screen, BLACK, r_scr, 4) 
        # Chấm ĐỎ (Đầu Xe)
        pygame.draw.circle(screen, RED, f_scr, 6)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim(True)
                elif event.key == pygame.K_r: reset_sim(False) # Re-run same map
                elif event.key == pygame.K_TAB: algo_mode = "MCPP" if algo_mode == "RRT" else "RRT"; reset_sim(False)

        # --- LOGIC ---
        if is_planning:
            for _ in range(50): 
                path = planner.plan_step()
                if path:
                    planned_path = path
                    is_planning = False
                    path_index = 0
                    print("Path Found!")
                    break
            
            # Giới hạn số lần lặp
            limit = RRT_MAX_ITER if algo_mode == "RRT" else MCPP_ITER * 10
            if hasattr(planner, 'node_list') and len(planner.node_list) > limit:
                print("Retry planning...")
                reset_sim(False) 

        elif planned_path and path_index < len(planned_path):
            # LOOKAHEAD
            collision_detected = False
            look_limit = min(path_index + LOOKAHEAD_STEPS, len(planned_path))
            
            for i in range(path_index, look_limit):
                future_state = planned_path[i]
                collided, hit_idx = check_collision_with_index(
                    future_state[0], future_state[1], future_state[2], outer_poly, real_holes
                )
                
                if collided:
                    collision_detected = True
                    if hit_idx != -1 and hit_idx not in known_hole_indices:
                        known_hole_indices.add(hit_idx)
                        planner_holes_geom.append(real_holes[hit_idx])
                        print(f"Obstacle #{hit_idx} Discovered! Re-planning...")
                    elif hit_idx == -1:
                        print("Wall collision! Re-planning...")
                    break
            
            if collision_detected:
                is_planning = True
                planned_path = []
                if algo_mode == "RRT":
                    bounds = [planner.min_x, planner.max_x, planner.min_y, planner.max_y]
                    planner = KinematicRRT(current_state, goal_pos, outer_poly, planner_holes_geom, bounds)
                else:
                    planner = KinematicMCPP(current_state, goal_pos, outer_poly, planner_holes_geom)
            else:
                current_state = planned_path[path_index]
                path_history.append((current_state[0], current_state[1]))
                path_index += 1

        # --- DRAWING ---
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        
        for i, h in enumerate(real_holes):
            col = RED if i in known_hole_indices else GHOST_GRAY
            pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

        # Vẽ cây (cho cả RRT và MCPP)
        if is_planning:
            for node in planner.node_list:
                # Với MCPP, node có parent_node
                parent = getattr(node, 'parent', None) or getattr(node, 'parent_node', None)
                if parent:
                    pts = [to_scr((px, py)) for px, py in zip(node.path_x, node.path_y)]
                    if len(pts)>1: pygame.draw.lines(screen, (200, 200, 255), False, pts, 1)

        if not is_planning and planned_path:
            pts = [to_scr((p[0], p[1])) for p in planned_path[path_index:]]
            if len(pts) > 1: pygame.draw.lines(screen, GREEN, False, pts, 2)
            
            look_end = min(path_index + LOOKAHEAD_STEPS, len(planned_path))
            if path_index < look_end:
                look_pts = [to_scr((p[0], p[1])) for p in planned_path[path_index:look_end]]
                if len(look_pts) > 1: pygame.draw.lines(screen, LOOKAHEAD_COLOR, False, look_pts, 4)

        if len(path_history) > 1:
            pts = [to_scr(p) for p in path_history]
            pygame.draw.lines(screen, BLACK, False, pts, 1)

        # Đích (Vùng Xanh)
        pygame.draw.circle(screen, BLUE, to_scr(goal_pos), int(GOAL_RADIUS * scale))
        
        draw_car(current_state)

        # GUI
        screen.blit(font.render(f"Mode: {algo_mode}", True, BLUE), (10, 10))
        status = "PLANNING..." if is_planning else "MOVING"
        
        # Check Finish (Đầu xe)
        fx_curr = current_state[0] + CAR_L * math.cos(current_state[2])
        fy_curr = current_state[1] + CAR_L * math.sin(current_state[2])
        if not is_planning and dist((fx_curr, fy_curr), goal_pos) < GOAL_RADIUS:
            status = "FINISHED"
            
        screen.blit(font.render(f"Status: {status}", True, RED if is_planning else GREEN), (10, 30))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()