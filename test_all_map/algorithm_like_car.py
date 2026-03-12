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
DUBINS_CONNECT_DIST = 20.0 

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

# --- THAM SỐ GRID SHELL (MCPP) ---
BIG_GRID_SIZE = 20.0    # Kích thước 1 ô lớn (tùy chỉnh theo bản đồ)
MAX_STEPS_PER_GRID = 25  # Nếu kẹt trong ô này 50 bước, ép đổi ô khác

# --- THÔNG SỐ CHƯỚNG NGẠI VẬT ĐỘNG ---
NUM_DYN_OBS = 6       # Tăng số lượng lên chút cho thú vị
DYN_OBS_RADIUS = 2.0  # THU BÉ BÓNG LẠI (từ 3.5 xuống 2.0)
DYN_OBS_SPEED = 12.0  

# MÀU SẮC
WHITE = (255, 255, 255); BLACK = (0, 0, 0); RED = (255, 0, 0)
GRAY = (200, 200, 200); GREEN = (0, 200, 0); BLUE = (0, 0, 255)
CAR_COLOR = (50, 50, 200)
LOOKAHEAD_COLOR = (255, 140, 0) 
GHOST_GRAY = (245, 245, 245)
DUBINS_COLOR = (180, 0, 180) # TÍM
REVERSE_COLOR = (255, 100, 0) # CAM
DYN_COLOR = (255, 100, 150) # HỒNG

# ==========================================
# CHƯỚNG NGẠI VẬT ĐỘNG (CLASS TỐI ƯU VẬT LÝ)
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y, radius, vx, vy):
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = vx
        self.vy = vy

    def move(self, dt, bounds, outer_poly, holes, robot_state, robot_safe_radius):
        # Tính vị trí tương lai nếu đi tiếp
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        
        # 1. Va chạm viền bản đồ (Bounding Box)
        min_x, max_x, min_y, max_y = bounds
        hit_bound = False
        if nx - self.radius < min_x or nx + self.radius > max_x:
            self.vx *= -1; hit_bound = True
        if ny - self.radius < min_y or ny + self.radius > max_y:
            self.vy *= -1; hit_bound = True
            
        if hit_bound:
            nx = self.x + self.vx * dt
            ny = self.y + self.vy * dt

        # 2. Va chạm với Đa giác tĩnh (SỬA LỖI LÚN TƯỜNG)
        hit_solid = False
        
        # Tạo 8 điểm cảm biến nằm ngay trên "vỏ" của quả bóng
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        check_pts = [(nx + self.radius * math.cos(a), ny + self.radius * math.sin(a)) for a in angles]
        check_pts.append((nx, ny)) # Thêm cả tâm bóng cho chắc
        
        # Quét xem có điểm cảm biến nào lọt vào vật cản tĩnh không
        for pt in check_pts:
            if outer_poly and not point_in_polygon(pt, outer_poly): 
                hit_solid = True
                break
            for h in holes:
                if point_in_polygon(pt, h):
                    hit_solid = True
                    break
            if hit_solid: break
                    
        # 3. Check va chạm với Robot (Bóng nảy vào xe)
        if not hit_solid and robot_state:
            rx, ry, _ = robot_state
            # Cộng gộp bán kính bóng và bán kính an toàn của xe
            if math.hypot(nx - rx, ny - ry) < (self.radius + robot_safe_radius):
                hit_solid = True

        # --- XỬ LÝ PHẢN XẠ NẾU CÓ VA CHẠM ---
        if hit_solid:
            # Đảo ngược lực văng
            self.vx *= -1
            self.vy *= -1
            
            # Xoay vector vận tốc đi một góc ngẫu nhiên để bóng không bị kẹt lặp lại 1 đường nảy
            angle = math.atan2(self.vy, self.vx) + random.uniform(-0.5, 0.5)
            speed = math.hypot(self.vx, self.vy)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            
            # DỪNG BÓNG TẠI CHỖ 1 FRAME: Điều này rất quan trọng để bóng không tiếp tục lấn tới
            nx = self.x
            ny = self.y

        # Chốt vị trí cuối cùng
        self.x = nx
        self.y = ny

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
class RSPath:
    def __init__(self, lengths, modes, total_len):
        self.lengths = lengths  
        self.modes = modes      
        self.length = total_len
        self.x, self.y, self.yaw = [], [], []

def LSL(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = 2 + d*d - (2*math.cos(a - b)) + (2*d*(sa - sb))
    if p_sq < 0: return None
    tmp = math.atan2((cb - ca), (d + sa - sb))
    return mod2pi(-a + tmp), math.sqrt(p_sq), mod2pi(b - tmp), ["L", "S", "L"]

def RSR(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = 2 + d*d - (2*math.cos(a - b)) + (2*d*(sb - sa))
    if p_sq < 0: return None
    tmp = math.atan2((ca - cb), (d - sa + sb))
    return mod2pi(a - tmp), math.sqrt(p_sq), mod2pi(-b + tmp), ["R", "S", "R"]

def LSR(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = -2 + d*d + (2*math.cos(a - b)) + (2*d*(sa + sb))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    return mod2pi(-a + tmp), p, mod2pi(-mod2pi(b) + tmp), ["L", "S", "R"]

def RSL(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = d*d - 2 + (2*math.cos(a - b)) - (2*d*(sa + sb))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    return mod2pi(a - tmp), p, mod2pi(b - tmp), ["R", "S", "L"]

def CCC_FBF(alpha, beta, d):
    x = d * math.cos(alpha)
    y = -d * math.sin(alpha)
    phi = normalize_angle(beta - alpha)
    
    best_t, best_p, best_q, best_mode = None, None, None, None
    min_len = float('inf')

    def calc_LRL(X, Y, PHI):
        u1 = math.sqrt((X - math.sin(PHI))**2 + (Y - 1.0 + math.cos(PHI))**2)
        if u1 <= 4.0:
            P = -2.0 * math.asin(0.25 * u1)
            T = mod2pi(math.atan2(Y - 1.0 + math.cos(PHI), X - math.sin(PHI)) + 0.5 * P + math.pi)
            Q = mod2pi(PHI - T + P)
            return T, P, Q
        return None

    res_LRL = calc_LRL(x, y, phi)
    if res_LRL:
        t, p, q = res_LRL
        l = abs(t) + abs(p) + abs(q)
        if l < min_len:
            min_len = l
            best_t, best_p, best_q, best_mode = t, p, q, ["L", "R", "L"]

    res_RLR = calc_LRL(x, -y, -phi)
    if res_RLR:
        t, p, q = res_RLR
        l = abs(t) + abs(p) + abs(q)
        if l < min_len:
            best_t, best_p, best_q, best_mode = t, p, q, ["R", "L", "R"]

    if best_mode:
        return best_t, best_p, best_q, best_mode
    return None

def reeds_shepp_planning(sx, sy, syaw, ex, ey, eyaw, c):
    dx = ex - sx
    dy = ey - sy
    
    lex = math.cos(syaw) * dx + math.sin(syaw) * dy
    ley = math.sin(syaw) * dx - math.cos(syaw) * dy  
    leyaw = normalize_angle(-(eyaw - syaw)) 
    
    D = math.sqrt(lex**2 + ley**2)
    d = D / c 
    phi = math.atan2(ley, lex)
    
    alpha = normalize_angle(-phi)
    beta = normalize_angle(leyaw - phi)

    best_p = None; min_l = float('inf')
    
    def flip_modes(modes):
        return ["R" if m == "L" else ("L" if m == "R" else "S") for m in modes]

    symmetries = [
        (alpha, beta, False, False), 
        (normalize_angle(-alpha - math.pi), normalize_angle(-beta - math.pi), True, False), 
        (normalize_angle(-alpha), normalize_angle(-beta), False, True), 
        (normalize_angle(alpha + math.pi), normalize_angle(beta + math.pi), True, True) 
    ]
    
    for f in [LSL, RSR, LSR, RSL, CCC_FBF]:
        for a, b, is_timeflip, is_reflect in symmetries:
            res = f(a, b, d)
            if res:
                t, p, q, m = res
                if is_timeflip: t, p, q = -t, -p, -q
                if is_reflect: m = flip_modes(m)
                    
                l = (abs(t) + abs(p) + abs(q)) * c
                if l < min_l and l < (D * 3.0 + 25.0): 
                    min_l = l
                    best_p = RSPath([t, p, q], m, l)

    if best_p:
        best_p.x, best_p.y, best_p.yaw = generate_rs_points(best_p, c, sx, sy, syaw)
        return best_p
    return None

def generate_rs_points(path, c, sx, sy, syaw):
    px, py, pyaw = [sx], [sy], [syaw]
    cx, cy, cyaw = sx, sy, syaw
    
    for i, mode in enumerate(path.modes):
        L = path.lengths[i] * c
        gear = 1 if L >= 0 else -1
        abs_L = abs(L)
        
        step = 0.1 
        n = max(1, int(abs_L / step))
        d_step = abs_L / n
        steer = -1 if mode == 'L' else (1 if mode == 'R' else 0)
        
        for _ in range(n):
            cx += d_step * gear * math.cos(cyaw)
            cy += d_step * gear * math.sin(cyaw)
            if mode != 'S':
                cyaw += (d_step * gear / c) * steer
            
            px.append(cx); py.append(cy); pyaw.append(normalize_angle(cyaw))
            
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

def check_collision_with_index(x, y, yaw, outer, holes, dyn_obs=None):
    corners = get_car_corners(x, y, yaw)
    # Va chạm Tĩnh
    for p in corners:
        if outer and not point_in_polygon(p, outer): return True, -1
        for i, h in enumerate(holes):
            if point_in_polygon(p, h): return True, i
            
    # Va chạm Động
    if dyn_obs:
        safe_radius = math.hypot(CAR_L/2 + 1.0, CAR_WIDTH/2) 
        for obs in dyn_obs:
            if math.hypot(x - obs.x, y - obs.y) < obs.radius + safe_radius:
                return True, -3 
    return False, -2

def check_path_collision(path_x, path_y, path_yaw, outer, holes, dyn_obs=None):
    step = 5 
    for i in range(0, len(path_x), step):
        collided, _ = check_collision_with_index(path_x[i], path_y[i], path_yaw[i], outer, holes, dyn_obs)
        if collided: return True
    return False

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

class KinematicRRT:
    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs=None):
        self.start = Node(start[0], start[1], start[2])
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes 
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.node_list = [self.start]
        self.dyn_obs = dyn_obs 

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
        
        if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
            new_node = Node(nx, ny, nyaw, nearest, is_dubins=False, direction=direction)
            new_node.path_x = px; new_node.path_y = py; new_node.path_yaw = pyaw
            self.node_list.append(new_node)
            
            if dist((nx, ny), self.goal_pos) <= DUBINS_CONNECT_DIST:
                dpath = reeds_shepp_planning(nx, ny, nyaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
                if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
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

class KinematicMCPP:
    class VNode: 
        def __init__(self, state, is_dubins=False, direction=1):
            self.state = state 
            self.N = 0; self.children = {}
            self.parent_node = None
            self.path_x = []; self.path_y = []; self.path_yaw = []
            self.is_dubins = is_dubins
            self.direction = direction
    
    class QNode: 
        def __init__(self, parent, action):
            self.parent = parent; self.action = action
            self.n = 0; self.Q = 0.0; self.child_v = None

    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, global_penalties=None, dyn_obs=None):
        self.root = self.VNode(start)
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes
        self.node_list = [self.root]
        self.dyn_obs = dyn_obs
        
        self.grid_visits = {}     
        self.grid_penalties = global_penalties if global_penalties is not None else {} 

    def get_dist_to_nearest_obstacle(self, state):
        min_d = 50.0 
        px, py = state[0], state[1]
        for hole in self.known_holes:
            for vertex in hole:
                d = math.sqrt((px - vertex[0])**2 + (py - vertex[1])**2)
                if d < min_d: min_d = d
                
        # Né tránh bóng động từ xa
        if self.dyn_obs:
            for obs in self.dyn_obs:
                d = math.hypot(px - obs.x, py - obs.y) - obs.radius
                if d < min_d: min_d = max(0.1, d)
        return min_d

    def get_action_ucb(self, v):
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            curr_c = MCPP_C * (1.5 if self.get_dist_to_nearest_obstacle(v.state) < 10.0 else 1.0)
            s = q.Q + curr_c * math.sqrt(math.log(max(1, v.N)) / q.n)
            
            if q.child_v:
                gid = (int(q.child_v.state[0] // BIG_GRID_SIZE), int(q.child_v.state[1] // BIG_GRID_SIZE))
                s -= self.grid_penalties.get(gid, 0)
                
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        d_obs = self.get_dist_to_nearest_obstacle(v.state)
        if d_obs < 15.0:
            steer = random.choice([-MAX_STEER, MAX_STEER, random.uniform(-MAX_STEER, MAX_STEER)])
        else:
            steer = random.uniform(-MAX_STEER, MAX_STEER)
            
        direction = -1 if random.random() < PROB_REVERSE else 1
        nx, ny, nyaw, px, py, pyaw = simulate_step(v.state[0], v.state[1], v.state[2], steer, direction)
        
        if check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs): return None
        
        action_key = (round(steer, 2), direction)
        if action_key not in v.children:
            qnode = self.QNode(v, action_key)
            v.children[action_key] = qnode
            return action_key
        return None

    def sim_v(self, v, d):
        sx, sy, syaw = v.state[0], v.state[1], v.state[2]
        
        if dist((sx, sy), self.goal_pos) < DUBINS_CONNECT_DIST:
            dpath = reeds_shepp_planning(sx, sy, syaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
            
            if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
                goal_v = self.VNode((self.goal_pos[0], self.goal_pos[1], self.goal_yaw), is_dubins=True)
                goal_v.parent_node = v
                goal_v.path_x, goal_v.path_y, goal_v.path_yaw = dpath.x, dpath.y, dpath.yaw
                goal_v.direction = -1 if any(l < 0 for l in dpath.lengths) else 1
                self.node_list.append(goal_v) 
                return 2000.0

        if d == 0 or dist(v.state[:2], self.goal_pos) < GOAL_RADIUS:
            d_goal = dist(v.state[:2], self.goal_pos)
            d_obs = self.get_dist_to_nearest_obstacle(v.state)
            return -(1.0 * d_goal) + (2.0 * d_obs if d_obs < 10.0 else 0.5 * d_obs)

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
            self.node_list.append(q.child_v)
            
            gid = (int(nx // BIG_GRID_SIZE), int(ny // BIG_GRID_SIZE))
            self.grid_visits[gid] = self.grid_visits.get(gid, 0) + 1
            
            if self.grid_visits[gid] > MAX_STEPS_PER_GRID:
                self.grid_penalties[gid] = self.grid_penalties.get(gid, 0) + 2000.0
                self.grid_visits[gid] = 0 
            
            curr = (nx, ny, nyaw)
            survive_steps = 0
            for _ in range(3):
                rx, ry, ryaw, rpx, rpy, _ = simulate_step(curr[0], curr[1], curr[2], random.uniform(-MAX_STEER, MAX_STEER), 1)
                if check_path_collision(rpx, rpy, [0]*len(rpx), self.outer, self.known_holes, self.dyn_obs): break
                curr = (rx, ry, ryaw)
            return -dist(curr[:2], self.goal_pos) + (0.5 * self.get_dist_to_nearest_obstacle(curr))

        r = self.sim_v(q.child_v, d - 1)
        q.n += 1; q.Q += (r - q.Q) / q.n; q.parent.N += 1
        return r

    def plan_step(self):
        for _ in range(MCPP_ITER // 40):
            self.sim_v(self.root, MCPP_DEPTH)
        
        for node in self.node_list:
            if dist(node.state[:2], self.goal_pos) < GOAL_RADIUS and abs(normalize_angle(node.state[2] - self.goal_yaw)) < 0.2:
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
    pygame.display.set_caption("Kinematic Planner + Bouncing Obstacles")
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
    global_grid_penalties = {}
    dyn_obstacles = [] 
    
    click_step = 0 
    is_planning = False
    scale = 1.0

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_state, goal_pos, goal_yaw, known_hole_indices, planner_holes_geom
        nonlocal planner, planned_path, flat_planned_path, path_index, is_planning, scale, path_history, click_step
        nonlocal global_grid_penalties, dyn_obstacles

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
        
        click_step = 0
        is_planning = False
        planner = None
        current_state = (0.0, 0.0, 0.0)
        goal_pos = np.array([0.0, 0.0])
        goal_yaw = 0.0
        
        if new_map: 
            known_hole_indices = set(); planner_holes_geom = []
            global_grid_penalties = {} 
            dyn_obstacles.clear()
            
            bounds = [0, mx, 0, mx]
            for _ in range(NUM_DYN_OBS):
                rp = get_valid_random_pos(outer_poly, real_holes, bounds)
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(DYN_OBS_SPEED/2, DYN_OBS_SPEED)
                dyn_obstacles.append(DynamicObstacle(rp[0], rp[1], DYN_OBS_RADIUS, math.cos(angle)*speed, math.sin(angle)*speed))
            
        planned_path = []; flat_planned_path = []; path_index = 0; path_history = []

    reset_sim(new_map=True)

    def to_scr(pos): return int(pos[0]*scale)+40, int(WINDOW_SIZE - (pos[1]*scale)-40)
    def from_scr(sx, sy): return (sx - 40) / scale, (WINDOW_SIZE - sy - 40) / scale
    
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
        dt_frame = 1.0 / FPS
        
        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        bounds = [0, max(max(xs), max(ys)), 0, max(max(xs), max(ys))] if outer_poly else [0, 700, 0, 700]
        
        # Bán kính xe an toàn để tính va chạm vật lý với bóng
        safe_car_radius = math.hypot(CAR_L/2 + 1.0, CAR_WIDTH/2)
        
        # Gọi hàm di chuyển bóng với đầy đủ check va chạm
        for obs in dyn_obstacles:
            # Nếu xe chưa click xong điểm bắt đầu thì không truyền current_state vào check va chạm
            active_robot_state = current_state if click_step >= 1 else None
            obs.move(dt_frame, bounds, outer_poly, real_holes, active_robot_state, safe_car_radius)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    if map_folders: current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim(True)
                elif event.key == pygame.K_r: reset_sim(False)
                elif event.key == pygame.K_TAB: algo_mode = "MCPP" if algo_mode == "RRT" else "RRT"; reset_sim(False)
            
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if click_step < 2:
                    wx, wy = from_scr(event.pos[0], event.pos[1])
                    if click_step == 0:
                        current_state = (wx, wy, 0.0); click_step = 1
                    elif click_step == 1:
                        goal_pos = np.array([wx, wy]); goal_yaw = 0.0 
                        click_step = 2; is_planning = True
                        
                        if algo_mode == "RRT":
                            planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, bounds, dyn_obstacles)
                        else:
                            planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, global_grid_penalties, dyn_obstacles)

        if is_planning and click_step == 2:
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

        elif flat_planned_path and path_index < len(flat_planned_path) and click_step == 2:
            collision_detected = False
            look_limit = min(path_index + LOOKAHEAD_STEPS, len(flat_planned_path))
            
            for i in range(path_index, look_limit):
                fs = flat_planned_path[i]
                collided, hit_idx = check_collision_with_index(fs[0], fs[1], fs[2], outer_poly, real_holes, dyn_obstacles)
                if collided:
                    collision_detected = True
                    if hit_idx >= 0 and hit_idx not in known_hole_indices:
                        known_hole_indices.add(hit_idx)
                        planner_holes_geom.append(real_holes[hit_idx])
                        hit_obstacle = real_holes[hit_idx]
                        for pt in hit_obstacle:
                            gid = (int(pt[0] // BIG_GRID_SIZE), int(pt[1] // BIG_GRID_SIZE))
                            global_grid_penalties[gid] = global_grid_penalties.get(gid, 0) + 5000.0 
                    break
            
            if collision_detected:
                is_planning = True 
                planned_path = []; flat_planned_path = []
                
                if algo_mode == "RRT":
                    planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, bounds, dyn_obstacles)
                else:
                    planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, global_grid_penalties, dyn_obstacles)
            else:
                if path_index < len(flat_planned_path):
                    current_state = flat_planned_path[path_index]
                    path_history.append((current_state[0], current_state[1]))
                    path_index += 1
                else:
                    current_state = flat_planned_path[-1]

        # --- DRAW ---
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        for i, h in enumerate(real_holes):
            col = RED if i in known_hole_indices else GHOST_GRAY
            pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

        # Vẽ vật cản động
        for obs in dyn_obstacles:
            pygame.draw.circle(screen, DYN_COLOR, to_scr((obs.x, obs.y)), int(obs.radius * scale))

        if is_planning and click_step == 2:
            for node in planner.node_list:
                parent = getattr(node, 'parent', None) or getattr(node, 'parent_node', None)
                if parent:
                    pts = [to_scr((px, py)) for px, py in zip(node.path_x, node.path_y)]
                    if len(pts)>1: pygame.draw.lines(screen, (200, 200, 255), False, pts, 1)

        if not is_planning and planned_path and click_step == 2:
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

        if click_step >= 2:
            g_scr = to_scr(goal_pos)
            pygame.draw.circle(screen, BLUE, g_scr, int(GOAL_RADIUS * scale))
            arrow_end = (goal_pos[0] + 4.0*math.cos(goal_yaw), goal_pos[1] + 4.0*math.sin(goal_yaw))
            pygame.draw.line(screen, BLUE, g_scr, to_scr(arrow_end), 3)

        if click_step >= 1:
            draw_car(current_state)

        screen.blit(font.render(f"Mode: {algo_mode} | [TAB] Switch | [N] Next Map | [R] Reset Map", True, BLUE), (10, 10))
        
        if click_step == 0: status = "VUI LONG CLICK CHON DIEM DAU"; color_st = BLUE
        elif click_step == 1: status = "VUI LONG CLICK CHON DIEM DICH"; color_st = BLUE
        else:
            reached_end_of_path = (flat_planned_path and path_index >= len(flat_planned_path))
            if not is_planning and reached_end_of_path:
                status = "FINISHED"; color_st = GREEN
            elif is_planning:
                status = "PLANNING..."; color_st = RED
            else:
                status = "MOVING"; color_st = GREEN

        screen.blit(font.render(f"Status: {status}", True, color_st), (10, 30))
        screen.blit(font.render("GREEN: Fwd | ORANGE: Rev | PURPLE: Dubins", True, BLACK), (10, 50))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()