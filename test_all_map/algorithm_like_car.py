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
WINDOW_SIZE = 700
FPS = 60
DATASET_DIR = "AC300"
RESULT_DIR = "Results"

# --- THÔNG SỐ XE (KINEMATIC) ---
CAR_L = 3.0           # Chiều dài cơ sở
CAR_WIDTH = 1.5       
MAX_STEER = 0.6       # ~40 độ
VELOCITY_MAX = 5.0    
VELOCITY_MIN = 2.0    # Tốc độ lùi/cua gắt
DT = 0.2              
SIM_TIME = 1.0        

# --- THÔNG SỐ DUBINS ---
MIN_TURN_RADIUS = CAR_L / math.tan(MAX_STEER)
DUBINS_STEP_SIZE = VELOCITY_MAX * DT 
DUBINS_CONNECT_DIST = 50.0 

# --- THÔNG SỐ ĐÍCH ---
GOAL_RADIUS = 3.0     

# --- CẢM BIẾN ---
LOOKAHEAD_STEPS = 5   

# --- THAM SỐ THUẬT TOÁN ---
RRT_MAX_ITER = 5000   
RRT_GOAL_PROB = 0.05  
PROB_REVERSE = 0.4   # 30% tỷ lệ đi lùi

MCPP_EPSILON = 5.0    
MCPP_C = 1.414        
MCPP_ITER = 2000      
MCPP_DEPTH = 30       
MCPP_BRANCHES = 5

# MÀU SẮC
WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (255, 0, 0)
GRAY = (200, 200, 200); GREEN = (0, 200, 0); BLUE = (0, 0, 255)
CAR_COLOR = (50, 50, 200)
LOOKAHEAD_COLOR = (255, 140, 0) 
GHOST_GRAY = (245, 245, 245)
DUBINS_COLOR = (180, 0, 180) # TÍM
REVERSE_COLOR = (255, 100, 0) # CAM

# ==========================================
# 2. HÀM HÌNH HỌC & DUBINS
# ==========================================
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def mod2pi(theta): 
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)

def dist(a, b): 
    return np.linalg.norm(np.array(a) - np.array(b))

def point_in_polygon(point, polygon):
    x, y = point
    if not polygon: return False
    n = len(polygon)
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

# --- DUBINS LOGIC ---
class DubinsPath:
    def __init__(self, t, p, q, length, mode):
        self.t = t; self.p = p; self.q = q
        self.length = length; self.mode = mode
        self.x = []; self.y = []; self.yaw = []

def dubins_path_planning(sx, sy, syaw, ex, ey, eyaw, c):
    ex = ex - sx; ey = ey - sy
    lex = math.cos(-syaw) * ex - math.sin(-syaw) * ey
    ley = math.cos(-syaw) * ey + math.sin(-syaw) * ex
    leyaw = mod2pi(eyaw - syaw)
    D = math.sqrt(lex**2 + ley**2); d = D / c
    theta = mod2pi(math.atan2(ley, lex))
    alpha = mod2pi(-theta); beta = mod2pi(leyaw - theta)

    planners = [LSL, RSR, LSR, RSL, RLR, LRL]
    best_path = None; min_len = float('inf')

    for planner in planners:
        res = planner(alpha, beta, d)
        if res:
            t, p, q, mode = res
            length = (abs(t) + abs(p) + abs(q)) * c
            if length < min_len:
                min_len = length
                best_path = DubinsPath(t, p, q, length, mode)
    
    if best_path:
        px, py, pyaw = generate_dubins_points(best_path, c, sx, sy, syaw)
        best_path.x = px; best_path.y = py; best_path.yaw = pyaw
        return best_path
    return None

def LSL(alpha, beta, d): 
    sa = math.sin(alpha); sb = math.sin(beta); ca = math.cos(alpha); cb = math.cos(beta)
    p_sq = 2 + d*d - (2*math.cos(alpha - beta)) + (2*d*(sa - sb))
    if p_sq < 0: return None
    tmp = math.atan2((cb - ca), (d + sa - sb))
    return mod2pi(-alpha + tmp), math.sqrt(p_sq), mod2pi(beta - tmp), ["L","S","L"]
def RSR(alpha, beta, d):
    sa = math.sin(alpha); sb = math.sin(beta); ca = math.cos(alpha); cb = math.cos(beta)
    p_sq = 2 + d*d - (2*math.cos(alpha - beta)) + (2*d*(sb - sa))
    if p_sq < 0: return None
    tmp = math.atan2((ca - cb), (d - sa + sb))
    return mod2pi(alpha - tmp), math.sqrt(p_sq), mod2pi(-beta + tmp), ["R","S","R"]
def LSR(alpha, beta, d):
    sa = math.sin(alpha); sb = math.sin(beta); ca = math.cos(alpha); cb = math.cos(beta)
    p_sq = -2 + d*d + (2*math.cos(alpha - beta)) + (2*d*(sa + sb))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    return mod2pi(-alpha + tmp), p, mod2pi(-mod2pi(beta) + tmp), ["L","S","R"]
def RSL(alpha, beta, d):
    sa = math.sin(alpha); sb = math.sin(beta); ca = math.cos(alpha); cb = math.cos(beta)
    p_sq = (d*d) - 2 + (2*math.cos(alpha - beta)) - (2*d*(sa + sb))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    return mod2pi(alpha - tmp), p, mod2pi(beta - tmp), ["R","S","L"]
def RLR(alpha, beta, d):
    sa = math.sin(alpha); sb = math.sin(beta); ca = math.cos(alpha); cb = math.cos(beta)
    tmp = (6.0 - d*d + 2.0*math.cos(alpha - beta) + 2.0*d*(sa - sb)) / 8.0
    if abs(tmp) > 1.0: return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, ["R","L","R"]
def LRL(alpha, beta, d):
    sa = math.sin(alpha); sb = math.sin(beta); ca = math.cos(alpha); cb = math.cos(beta)
    tmp = (6.0 - d*d + 2.0*math.cos(alpha - beta) + 2.0*d*(sb - sa)) / 8.0
    if abs(tmp) > 1.0: return None
    p = mod2pi(2 * math.pi - math.acos(tmp))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))
    return t, p, q, ["L","R","L"]

def generate_dubins_points(path, c, sx, sy, syaw):
    px, py, pyaw = [sx], [sy], [syaw]
    lengths = [path.t, path.p, path.q]
    step = DUBINS_STEP_SIZE
    curr_yaw = syaw; curr_x = sx; curr_y = sy
    
    for i, mode in enumerate(path.mode):
        seg_len = lengths[i] * c
        n_steps = int(seg_len / step)
        steer = 1 if mode == 'L' else (-1 if mode == 'R' else 0)
        phys_steer = math.atan(CAR_L / c) * steer
        
        for _ in range(n_steps):
            curr_x += VELOCITY_MAX * math.cos(curr_yaw) * (step/VELOCITY_MAX)
            curr_y += VELOCITY_MAX * math.sin(curr_yaw) * (step/VELOCITY_MAX)
            if mode != 'S':
                curr_yaw += (VELOCITY_MAX / CAR_L) * math.tan(phys_steer) * (step/VELOCITY_MAX)
            px.append(curr_x); py.append(curr_y); pyaw.append(curr_yaw)
    return px, py, pyaw

# ==========================================
# 3. MÔI TRƯỜNG & MÔ PHỎNG
# ==========================================
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
        rx = random.uniform(min_x, max_x); ry = random.uniform(min_y, max_y)
        if not point_in_polygon((rx, ry), outer): continue
        valid = True
        for h in holes:
            if point_in_polygon((rx, ry), h): valid = False; break
        if valid: return np.array([rx, ry])
    return np.array([50, 50])

def get_car_corners(x, y, yaw):
    rear_overhang = 1.0; front_overhang = 1.0
    local = [(-rear_overhang, CAR_WIDTH/2), (-rear_overhang, -CAR_WIDTH/2),
             (CAR_L + front_overhang, -CAR_WIDTH/2), (CAR_L + front_overhang, CAR_WIDTH/2)]
    world = []
    c, s = math.cos(yaw), math.sin(yaw)
    for lx, ly in local:
        world.append((lx*c - ly*s + x, lx*s + ly*c + y))
    return world

def check_collision_with_index(x, y, yaw, outer, holes):
    corners = get_car_corners(x, y, yaw)
    for p in corners:
        if outer and not point_in_polygon(p, outer): return True, -1
        for i, h in enumerate(holes):
            if point_in_polygon(p, h): return True, i
    return False, -2

def check_path_collision(path_x, path_y, path_yaw, outer, holes):
    step = 5 
    for i in range(0, len(path_x), step):
        collided, _ = check_collision_with_index(path_x[i], path_y[i], path_yaw[i], outer, holes)
        if collided: return True
    return False

# --- SIMULATE CÓ DIRECTION ---
def simulate_step(x, y, yaw, steer, direction, sim_time=SIM_TIME):
    path_x, path_y, path_yaw = [x], [y], [yaw]
    steps = int(sim_time / DT)
    
    base_vel = VELOCITY_MIN if direction == -1 else VELOCITY_MAX
    steer_factor = abs(steer) / MAX_STEER
    current_vel = base_vel * (1.0 - 0.5 * steer_factor)
    v = current_vel * direction

    for _ in range(steps):
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += (v / CAR_L) * math.tan(steer) * DT
        yaw = normalize_angle(yaw)
        path_x.append(x); path_y.append(y); path_yaw.append(yaw)
    return x, y, yaw, path_x, path_y, path_yaw

# ==========================================
# 4. PLANNERS
# ==========================================
class Node:
    def __init__(self, x, y, yaw, parent=None, is_dubins=False, direction=1):
        self.x = x; self.y = y; self.yaw = yaw
        self.parent = parent
        self.path_x = []; self.path_y = []; self.path_yaw = []
        self.is_dubins = is_dubins
        self.direction = direction

# --- A. KINEMATIC RRT + DUBINS + REVERSE ---
class KinematicRRT:
    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds):
        self.start = Node(start[0], start[1], start[2])
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes 
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.node_list = [self.start]

    def plan_step(self):
        if random.random() < RRT_GOAL_PROB: rnd = (self.goal_pos[0], self.goal_pos[1])
        else: rnd = (random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))
        
        dists = [(node.x - rnd[0])**2 + (node.y - rnd[1])**2 for node in self.node_list]
        nearest = self.node_list[dists.index(min(dists))]
        
        dx = rnd[0] - nearest.x; dy = rnd[1] - nearest.y
        target_yaw = math.atan2(dy, dx)
        
        diff_head = normalize_angle(target_yaw - nearest.yaw)
        diff_tail = normalize_angle(target_yaw - (nearest.yaw + math.pi))
        
        direction = 1
        if random.random() < PROB_REVERSE: 
            direction = -1; diff = diff_tail
        else: 
            direction = 1; diff = diff_head
            
        steer = max(-MAX_STEER, min(MAX_STEER, diff))
        nx, ny, nyaw, px, py, pyaw = simulate_step(nearest.x, nearest.y, nearest.yaw, steer, direction)
        
        if not check_path_collision(px, py, pyaw, self.outer, self.known_holes):
            new_node = Node(nx, ny, nyaw, nearest, is_dubins=False, direction=direction)
            new_node.path_x = px; new_node.path_y = py; new_node.path_yaw = pyaw
            self.node_list.append(new_node)
            
            # DUBINS CONNECT
            if dist((nx, ny), self.goal_pos) <= DUBINS_CONNECT_DIST:
                dpath = dubins_path_planning(nx, ny, nyaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
                if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes):
                    goal_node = Node(self.goal_pos[0], self.goal_pos[1], self.goal_yaw, new_node, is_dubins=True)
                    goal_node.path_x = dpath.x; goal_node.path_y = dpath.y; goal_node.path_yaw = dpath.yaw
                    return self.extract_path(goal_node)
            
            if dist((nx, ny), self.goal_pos) < GOAL_RADIUS:
                return self.extract_path(new_node)
        return None

    def extract_path(self, node):
        full_path = []
        while node.parent:
            points = list(zip(node.path_x, node.path_y, node.path_yaw))
            segment = {'points': points, 'is_dubins': node.is_dubins, 'direction': node.direction}
            full_path.insert(0, segment)
            node = node.parent
        return full_path

# --- B. KINEMATIC MCPP + DUBINS + REVERSE ---
class KinematicMCPP:
    class VNode: 
        def __init__(self, state, is_dubins=False, direction=1):
            self.state = state 
            self.N = 0; self.children = {}
            self.parent_node = None
            # QUAN TRỌNG: Giữ các danh sách này để hàm main() có thể vẽ
            self.path_x = []; self.path_y = []; self.path_yaw = []
            self.is_dubins = is_dubins
            self.direction = direction
    
    class QNode: 
        def __init__(self, parent, action):
            self.parent = parent; self.action = action
            self.n = 0; self.Q = 0.0; self.child_v = None

    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes):
        self.root = self.VNode(start)
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes
        # node_list này chính là danh sách chứa các "đường mảnh" để vẽ
        self.node_list = [self.root]

    def get_dist_to_nearest_obstacle(self, state):
        min_d = 50.0 
        px, py = state[0], state[1]
        for hole in self.known_holes:
            for vertex in hole:
                d = math.sqrt((px - vertex[0])**2 + (py - vertex[1])**2)
                if d < min_d: min_d = d
        return min_d

    def get_action_ucb(self, v):
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            # Tăng C khi gần vật cản để kích thích "tìm hướng khác"
            curr_c = MCPP_C * (1.5 if self.get_dist_to_nearest_obstacle(v.state) < 10.0 else 1.0)
            s = q.Q + curr_c * math.sqrt(math.log(max(1, v.N)) / q.n)
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        d_obs = self.get_dist_to_nearest_obstacle(v.state)
        # Né tránh chủ động: Thử các góc lái gắt khi ở gần vật cản
        if d_obs < 15.0:
            steer = random.choice([-MAX_STEER, MAX_STEER, random.uniform(-MAX_STEER, MAX_STEER)])
        else:
            steer = random.uniform(-MAX_STEER, MAX_STEER)
            
        direction = -1 if random.random() < PROB_REVERSE else 1
        nx, ny, nyaw, px, py, pyaw = simulate_step(v.state[0], v.state[1], v.state[2], steer, direction)
        
        if check_path_collision(px, py, pyaw, self.outer, self.known_holes): return None
        
        action_key = (round(steer, 2), direction)
        if action_key not in v.children:
            qnode = self.QNode(v, action_key)
            v.children[action_key] = qnode
            return action_key
        return None

    def sim_v(self, v, d):
        # Thử Dubins
        if dist(v.state[:2], self.goal_pos) < DUBINS_CONNECT_DIST:
            dpath = dubins_path_planning(v.state[0], v.state[1], v.state[2], self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
            if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes):
                goal_v = self.VNode((self.goal_pos[0], self.goal_pos[1], self.goal_yaw), is_dubins=True)
                goal_v.parent_node = v
                goal_v.path_x, goal_v.path_y, goal_v.path_yaw = dpath.x, dpath.y, dpath.yaw
                self.node_list.append(goal_v) 
                return 2000.0

        if d == 0 or dist(v.state[:2], self.goal_pos) < GOAL_RADIUS:
            d_goal = dist(v.state[:2], self.goal_pos)
            d_obs = self.get_dist_to_nearest_obstacle(v.state)
            return -(1.0 * d_goal) + (2.0 * d_obs if d_obs < 10.0 else 0.5 * d_obs)

        # Dynamic Branching khi gần vật cản
        max_b = MCPP_BRANCHES * (2 if self.get_dist_to_nearest_obstacle(v.state) < 10.0 else 1)
        if len(v.children) < max_b:
            act = self.expand(v)
            if act: return self.sim_q(v.children[act], d)
        
        if not v.children: return -dist(v.state[:2], self.goal_pos)
        
        res = self.get_action_ucb(v)
        if res: return self.sim_q(res[1], d)
        return -dist(v.state[:2], self.goal_pos)

    def sim_q(self, q, d):
        steer, direction = q.action
        if not q.child_v:
            nx, ny, nyaw, px, py, pyaw = simulate_step(q.parent.state[0], q.parent.state[1], q.parent.state[2], steer, direction)
            q.child_v = self.VNode((nx, ny, nyaw), is_dubins=False, direction=direction)
            q.child_v.parent_node = q.parent
            q.child_v.path_x, q.child_v.path_y, q.child_v.path_yaw = px, py, pyaw
            # Đưa vào node_list để hàm vẽ có thể truy cập
            self.node_list.append(q.child_v)
            
            # Rollout đơn giản
            curr = (nx, ny, nyaw)
            for _ in range(3):
                rx, ry, ryaw, rpx, rpy, _ = simulate_step(curr[0], curr[1], curr[2], random.uniform(-MAX_STEER, MAX_STEER), 1)
                if check_path_collision(rpx, rpy, [0]*len(rpx), self.outer, self.known_holes): break
                curr = (rx, ry, ryaw)
            return -dist(curr[:2], self.goal_pos) + (0.5 * self.get_dist_to_nearest_obstacle(curr))

        r = self.sim_v(q.child_v, d - 1)
        q.n += 1; q.Q += (r - q.Q) / q.n; q.parent.N += 1
        return r

    def plan_step(self):
        # Chạy giả lập để xây dựng cây
        for _ in range(MCPP_ITER // 40):
            self.sim_v(self.root, MCPP_DEPTH)
        
        # Kiểm tra nếu đã chạm đích
        for node in self.node_list:
            if dist(node.state[:2], self.goal_pos) < GOAL_RADIUS:
                return self.extract_path(node)
        return None

    def extract_path(self, node):
        full_path = []
        curr = node
        while curr and curr.parent_node:
            points = list(zip(curr.path_x, curr.path_y, curr.path_yaw))
            full_path.insert(0, {'points': points, 'is_dubins': curr.is_dubins, 'direction': curr.direction})
            curr = curr.parent_node
        return full_path

# ==========================================
# 5. MAIN LOOP
# ==========================================
def main():
    pygame.init()
    if not os.path.exists("Results"): os.makedirs("Results")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Kinematic RRT/MCPP + Dubins + Reverse + Colors")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC12_*")))
    
    current_map_idx = 0
    algo_mode = "RRT"
    outer_poly = []; real_holes = []
    known_hole_indices = set(); planner_holes_geom = []
    current_state = (0, 0, 0)
    goal_pos = np.array([0, 0]); goal_yaw = 0.0
    planner = None
    
    planned_path = []; flat_planned_path = []; path_index = 0; path_history = []
    is_planning = True
    scale = 1.0

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_state, goal_pos, goal_yaw, known_hole_indices, planner_holes_geom
        nonlocal planner, planned_path, flat_planned_path, path_index, is_planning, scale, path_history

        use_dummy = False
        if map_folders:
            folder = map_folders[current_map_idx]
            if new_map: print(f"Loading: {folder}")
            outer_poly, real_holes = load_data(folder)
            if not outer_poly: use_dummy = True
        else: use_dummy = True
        
        if use_dummy:
             outer_poly = [(0,0), (WINDOW_SIZE,0), (WINDOW_SIZE,WINDOW_SIZE), (0,WINDOW_SIZE)]
             real_holes = [[(300,300), (500,300), (500,500), (300,500)]]

        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        mx = max(max(xs), max(ys))
        scale = (WINDOW_SIZE - 80) / mx
        min_x, min_y = min(xs), min(ys)
        
        start_pos = np.array([min_x+20.0, min_y+20.0])
        current_state = (start_pos[0], start_pos[1], 0.0)
        
        if new_map or (goal_pos[0]==0):
            goal_pos = get_valid_random_pos(outer_poly, real_holes, [min_x, max(xs), min_y, max(ys)])
            goal_yaw = random.uniform(-math.pi, math.pi)
            
        bounds = [0, mx, 0, mx]
        if algo_mode == "RRT":
            planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, bounds)
        else:
            planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom)
        
        if new_map: 
            known_hole_indices = set(); planner_holes_geom = []
            
        planned_path = []; flat_planned_path = []; path_index = 0; path_history = []
        is_planning = True

    reset_sim(new_map=True)

    def to_scr(pos): return int(pos[0]*scale)+40, int(WINDOW_SIZE - (pos[1]*scale)-40)
    
    def draw_car(state, color=CAR_COLOR):
        x, y, yaw = state
        corners = get_car_corners(x, y, yaw)
        scr_corners = [to_scr(p) for p in corners]
        pygame.draw.polygon(screen, color, scr_corners)
        pygame.draw.polygon(screen, BLACK, scr_corners, 1)
        fx = x + CAR_L * math.cos(yaw); fy = y + CAR_L * math.sin(yaw)
        pygame.draw.line(screen, BLACK, to_scr((x,y)), to_scr((fx,fy)), 2)

    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    if map_folders: current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim(True)
                elif event.key == pygame.K_r: reset_sim(False)
                elif event.key == pygame.K_TAB: algo_mode = "MCPP" if algo_mode == "RRT" else "RRT"; reset_sim(False)

        if is_planning:
            for _ in range(20): 
                path_segments = planner.plan_step()
                if path_segments:
                    planned_path = path_segments
                    flat_planned_path = []
                    for seg in path_segments:
                        flat_planned_path.extend(seg['points'])
                    is_planning = False
                    path_index = 0
                    print("Path Found!")
                    break
            
            limit = RRT_MAX_ITER if algo_mode == "RRT" else MCPP_ITER * 10
            if len(planner.node_list) > limit:
                print("Retry planning...")
                reset_sim(False) 

        elif flat_planned_path and path_index < len(flat_planned_path):
            collision_detected = False
            look_limit = min(path_index + LOOKAHEAD_STEPS, len(flat_planned_path))
            
            for i in range(path_index, look_limit):
                fs = flat_planned_path[i]
                collided, hit_idx = check_collision_with_index(fs[0], fs[1], fs[2], outer_poly, real_holes)
                if collided:
                    collision_detected = True
                    # Cập nhật bản đồ vật cản ngay khi nhìn thấy từ xa
                    if hit_idx != -1 and hit_idx not in known_hole_indices:
                        known_hole_indices.add(hit_idx)
                        planner_holes_geom.append(real_holes[hit_idx])
                    break
            
            if collision_detected:
                # THAY ĐỔI TẠI ĐÂY: Không reset sim hoàn toàn, mà yêu cầu tìm nhánh mới
                is_planning = True 
                # Giữ nguyên path_history nhưng xóa con đường cũ đang đi bị kẹt
                planned_path = []
                flat_planned_path = []
                
                # Khởi tạo lại Planner từ vị trí HIỆN TẠI của xe
                # Khi này Planner sẽ thấy phía trước có vật cản và tự ưu tiên nhánh lùi (PROB_REVERSE)
                if algo_mode == "RRT":
                    planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, [0, 700, 0, 700])
                else:
                    planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom)
                
                print("Obstacle ahead! Re-routing or reversing...")
            else:
                # Di chuyển bình thường
                current_state = flat_planned_path[path_index]
                path_history.append((current_state[0], current_state[1]))
                path_index += 1

        # --- DRAW ---
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        for i, h in enumerate(real_holes):
            col = RED if i in known_hole_indices else GHOST_GRAY
            pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

        if is_planning:
            for node in planner.node_list:
                parent = getattr(node, 'parent', None) or getattr(node, 'parent_node', None)
                if parent:
                    pts = [to_scr((px, py)) for px, py in zip(node.path_x, node.path_y)]
                    if len(pts)>1: pygame.draw.lines(screen, (200, 200, 255), False, pts, 1)

        if not is_planning and planned_path:
            for seg in planned_path:
                points = seg['points']
                if len(points) > 1:
                    pts_scr = [to_scr((p[0], p[1])) for p in points]
                    if seg['is_dubins']: col = DUBINS_COLOR; w = 4
                    elif seg['direction'] == -1: col = REVERSE_COLOR; w = 2
                    else: col = GREEN; w = 2
                    pygame.draw.lines(screen, col, False, pts_scr, w)

        if len(path_history) > 1:
            pygame.draw.lines(screen, BLACK, False, [to_scr(p) for p in path_history], 1)

        g_scr = to_scr(goal_pos)
        pygame.draw.circle(screen, BLUE, g_scr, int(GOAL_RADIUS * scale))
        arrow_end = (goal_pos[0] + 4.0*math.cos(goal_yaw), goal_pos[1] + 4.0*math.sin(goal_yaw))
        pygame.draw.line(screen, BLUE, g_scr, to_scr(arrow_end), 3)

        draw_car(current_state)

        screen.blit(font.render(f"Mode: {algo_mode} | [TAB] Switch | [N] Next Map", True, BLUE), (10, 10))
        status = "PLANNING..." if is_planning else "MOVING"
        if not is_planning and dist(current_state[:2], goal_pos) < GOAL_RADIUS: status = "FINISHED"
        screen.blit(font.render(f"Status: {status}", True, RED if is_planning else GREEN), (10, 30))
        screen.blit(font.render("GREEN: Fwd | ORANGE: Rev | PURPLE: Dubins", True, BLACK), (10, 50))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()