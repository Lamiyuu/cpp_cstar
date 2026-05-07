import random
import math
import numpy as np
from config import *
from core_math import dist, normalize_angle, simulate_step, reeds_shepp_planning
from environment import check_path_collision
import heapq
import sys
sys.setrecursionlimit(5000)

def angle_between_vectors(v1, v2):
    """Tính góc giữa 2 vector (Trả về radian)"""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 * mag2 == 0: return 0.0
    val = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.acos(val)

def amugopia_sabo_ordering(start_pos, goals):
    if not goals: return []
    unvisited = goals.copy()
    ordered_path = []
    current = start_pos
    
    all_nodes = [start_pos] + goals
    cx = sum(p[0] for p in all_nodes) / len(all_nodes)
    cy = sum(p[1] for p in all_nodes) / len(all_nodes)
    centroid = (cx, cy)
    
    while unvisited:
        if len(unvisited) < 3:
            unvisited.sort(key=lambda p: math.hypot(p[0]-current[0], p[1]-current[1]))
            next_node = unvisited.pop(0)
            ordered_path.append(next_node)
            current = next_node
            continue
            
        unvisited.sort(key=lambda p: math.hypot(p[0]-current[0], p[1]-current[1]))
        pot1 = unvisited[0]
        pot2 = unvisited[1]
        temp_goal = unvisited[2]
        
        r = (current[0] - centroid[0], current[1] - centroid[1])
        s1 = (pot1[0] - centroid[0], pot1[1] - centroid[1])
        s2 = (pot2[0] - centroid[0], pot2[1] - centroid[1])
        alpha1 = angle_between_vectors(r, s1)
        alpha2 = angle_between_vectors(r, s2)
        
        t = (current[0] - start_pos[0], current[1] - start_pos[1])
        u1 = (pot1[0] - start_pos[0], pot1[1] - start_pos[1])
        u2 = (pot2[0] - start_pos[0], pot2[1] - start_pos[1])
        beta1 = angle_between_vectors(t, u1)
        beta2 = angle_between_vectors(t, u2)
        
        w = (temp_goal[0] - current[0], temp_goal[1] - current[1])
        v1 = (pot1[0] - current[0], pot1[1] - current[1])
        v2 = (pot2[0] - current[0], pot2[1] - current[1])
        gamma1 = angle_between_vectors(v1, w)
        gamma2 = angle_between_vectors(v2, w)
        
        if alpha1 < alpha2: next_node = pot1
        elif alpha1 > alpha2 and gamma1 > gamma2: next_node = pot1
        elif alpha1 > alpha2 and gamma1 < gamma2 and beta1 < beta2: next_node = pot1
        elif alpha1 > alpha2 and gamma1 < gamma2 and beta1 > beta2: next_node = pot2
        else: next_node = pot1
            
        ordered_path.append(next_node)
        unvisited.remove(next_node)
        current = next_node
        
    # --- ĐÃ XÓA DÒNG ÉP KHÉP KÍN CHU TRÌNH ---
    return ordered_path

def get_topological_path(start_pos, goal_pos, bounds, clearance_func, grid_res=2.0):
    """ TẦNG 1: Tìm xương sống bằng A* kết hợp Hàm phạt (Penalty Cost) """
    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    start_grid = (int(start_pos[0] // grid_res), int(start_pos[1] // grid_res))
    goal_grid = (int(goal_pos[0] // grid_res), int(goal_pos[1] // grid_res))
    
    open_set = []
    heapq.heappush(open_set, (0, start_grid))
    came_from = {}
    g_score = {start_grid: 0}
    
    best_node = start_grid
    min_h = heuristic(start_grid, goal_grid)
    
    neighbors = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        h_curr = heuristic(current, goal_grid)
        if h_curr < min_h:
            min_h = h_curr
            best_node = current
            
        if current == goal_grid or h_curr < 2:
            best_node = current
            break
            
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            nx_real, ny_real = nxt[0] * grid_res, nxt[1] * grid_res
            
            clearance = clearance_func((nx_real, ny_real))
            
            # ĐIỀU KIỆN CỨNG: Chạm vạch (khoảng cách < 1.0) -> Chặn tuyệt đối
            if clearance < 1.0:
                continue
                
            # ĐIỀU KIỆN MỀM: Phạt cực nặng nếu đi gần vạch (dưới 5.0)
            # Điều này ép A* luôn phải lách ra giữa hành lang mà không lo bị kẹt
            penalty = 0.0
            if clearance < 5.0:
                penalty = (5.0 - clearance) * 10.0 # Hệ số phạt khổng lồ

            tentative_g = g_score[current] + math.hypot(dx, dy) + penalty
            
            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + heuristic(nxt, goal_grid)
                heapq.heappush(open_set, (f_score, nxt))
                
    path = []
    curr = best_node
    while curr in came_from:
        path.append((curr[0] * grid_res, curr[1] * grid_res))
        curr = came_from[curr]
    path.reverse()
    
    if not path:
        return [goal_pos]
        
    # =========================================================
    # LÀM MƯỢT ĐƯỜNG CAM CHUẨN KINEMATIC (Moving Average Filter)
    # Biến góc gãy 90 độ thành vòng cung mềm mại cho xe dễ bám
    # =========================================================
    smoothed_path = path[:]
    for _ in range(5): # Chạy 5 lớp lọc để đường thật mượt
        temp = [smoothed_path[0]]
        for i in range(1, len(smoothed_path)-1):
            nx = smoothed_path[i][0]*0.5 + (smoothed_path[i-1][0]+smoothed_path[i+1][0])*0.25
            ny = smoothed_path[i][1]*0.5 + (smoothed_path[i-1][1]+smoothed_path[i+1][1])*0.25
            temp.append((nx, ny))
        temp.append(smoothed_path[-1])
        smoothed_path = temp
        
    return smoothed_path + [goal_pos]

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
        self.dyn_obs = None 

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


class KinematicRRT:
    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs=None):
        self.start = Node(start[0], start[1], start[2])
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes 
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.node_list = [self.start]
        self.dyn_obs = None 

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
        def __init__(self, state, wp_idx=0, is_dubins=False, direction=1):
            self.state = state 
            self.wp_idx = wp_idx # ĐÃ BỔ SUNG BIẾN TRÍ NHỚ TIẾN ĐỘ
            self.N = 0; self.children = {}
            self.parent_node = None
            self.path_x = []; self.path_y = []; self.path_yaw = []
            self.is_dubins = is_dubins
            self.direction = direction
    
    class QNode: 
        def __init__(self, parent, action):
            self.parent = parent; self.action = action
            self.n = 0; self.Q = 0.0; self.child_v = None

    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds, waypoints, dyn_obs=None):
        self.root = self.VNode(start, wp_idx=0)
        self.goal_pos = goal_pos
        self.goal_yaw = goal_yaw
        self.outer = outer
        self.known_holes = known_holes
        self.bounds = bounds 
        self.dyn_obs = dyn_obs
        self.node_list = [self.root] 
        self.grid_visits = {}
        self.waypoints = waypoints

    def macro_step(self, sx, sy, syaw, steer, direction, num_steps=3):
        cx, cy, cyaw = sx, sy, syaw
        full_px, full_py, full_pyaw = [], [], []
        for _ in range(num_steps):
            nx, ny, nyaw, px, py, pyaw = simulate_step(cx, cy, cyaw, steer, direction)
            full_px.extend(px)
            full_py.extend(py)
            full_pyaw.extend(pyaw)
            cx, cy, cyaw = nx, ny, nyaw
        return cx, cy, cyaw, full_px, full_py, full_pyaw
    
    def get_action_ucb(self, v):
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            curr_c = MCPP_C * 2.0 
            s = q.Q + curr_c * math.sqrt(math.log(max(1, v.N)) / q.n)
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        cx, cy, cyaw = v.state[0], v.state[1], v.state[2]
        
        min_dist = float('inf')
        closest_idx = v.wp_idx
        search_end = min(len(self.waypoints), v.wp_idx + 15)
        for i in range(v.wp_idx, search_end):
            wp = self.waypoints[i]
            d = math.hypot(cx - wp[0], cy - wp[1])
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        v.wp_idx = closest_idx 
        
        target_idx = closest_idx
        look_ahead_dist = 4.0 
        while target_idx < len(self.waypoints):
            wp = self.waypoints[target_idx]
            if math.hypot(cx - wp[0], cy - wp[1]) < look_ahead_dist:
                target_idx += 1
            else:
                break
                
        if target_idx < len(self.waypoints):
            guide_x, guide_y = self.waypoints[target_idx]
        else:
            guide_x, guide_y = self.goal_pos[0], self.goal_pos[1]
            
        dx = guide_x - cx
        dy = guide_y - cy
        angle_to_target = math.atan2(dy, dx)
        
        diff_head = normalize_angle(angle_to_target - cyaw)
        diff_tail = normalize_angle(angle_to_target - (cyaw + math.pi))
        
        if abs(diff_head) > math.pi / 2.0:
            direction = -1
            diff = diff_tail
        else:
            direction = 1
            diff = diff_head
            
        ideal_steer = max(-MAX_STEER, min(MAX_STEER, diff))

        candidates = []
        candidates.append((ideal_steer, direction))
        candidates.append((MAX_STEER, direction))
        candidates.append((-MAX_STEER, direction))
        candidates.append((max(-MAX_STEER, min(MAX_STEER, ideal_steer + 0.1)), direction))
        candidates.append((max(-MAX_STEER, min(MAX_STEER, ideal_steer - 0.1)), direction))
        alt_dir = -direction
        candidates.append((-ideal_steer, alt_dir)) 
        candidates.append((MAX_STEER, alt_dir))    
        candidates.append((-MAX_STEER, alt_dir))   

        for steer, dir_val in candidates:
            steer = round(steer, 2)
            action_key = (steer, dir_val)
            if action_key in v.children: continue
            
            nx, ny, nyaw, px, py, pyaw = self.macro_step(cx, cy, cyaw, steer, dir_val, num_steps=3)
            
            if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                qnode = self.QNode(v, action_key)
                v.children[action_key] = qnode
                return action_key
                
        return None

    def sim_v(self, v, d):
        waypoints_left = len(self.waypoints) - 1 - v.wp_idx
        dist_to_goal = math.hypot(v.state[0] - self.goal_pos[0], v.state[1] - self.goal_pos[1])
        
        if d <= 0 or dist_to_goal < GOAL_RADIUS:
            return -(waypoints_left * 20.0 + dist_to_goal) 

        if len(v.children) < 8: 
            act = self.expand(v)
            if act:
                return self.sim_q(v.children[act], d - 1)
        
        res = self.get_action_ucb(v)
        if res:
            return self.sim_q(res[1], d - 1)
            
        return -(waypoints_left * 20.0 + dist_to_goal)

    def sim_q(self, q, d):
        if not q.child_v:
            steer, direction = q.action
            nx, ny, nyaw, px, py, pyaw = self.macro_step(
                q.parent.state[0], q.parent.state[1], q.parent.state[2], 
                steer, direction, num_steps=3
            )
            
            min_wp_dist = float('inf')
            closest_idx = q.parent.wp_idx
            search_end = min(len(self.waypoints), q.parent.wp_idx + 15)
            for i in range(q.parent.wp_idx, search_end):
                wp = self.waypoints[i]
                d_wp = math.hypot(nx - wp[0], ny - wp[1])
                if d_wp < min_wp_dist:
                    min_wp_dist = d_wp
                    closest_idx = i
            
            q.child_v = self.VNode((nx, ny, nyaw), wp_idx=closest_idx, is_dubins=False, direction=direction)
            q.child_v.parent_node = q.parent
            q.child_v.path_x, q.child_v.path_y, q.child_v.path_yaw = px, py, pyaw
            self.node_list.append(q.child_v)
            
            grid_x = int(nx // 2.0)
            grid_y = int(ny // 2.0)
            grid_yaw = int(math.degrees(normalize_angle(nyaw)) // 15.0)
            gid = (grid_x, grid_y, grid_yaw)
            
            if gid in self.grid_visits:
                return -99999.0 
            self.grid_visits[gid] = True
            
            waypoints_left = len(self.waypoints) - 1 - closest_idx
            dist_to_goal = dist((nx, ny), self.goal_pos)
            
            if waypoints_left <= 6 and dist_to_goal < 40.0:  
                dpath = reeds_shepp_planning(nx, ny, nyaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
                if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
                    goal_v = self.VNode((self.goal_pos[0], self.goal_pos[1], self.goal_yaw), wp_idx=len(self.waypoints)-1, is_dubins=True)
                    goal_v.parent_node = q.child_v
                    goal_v.path_x, goal_v.path_y, goal_v.path_yaw = dpath.x, dpath.y, dpath.yaw
                    goal_v.direction = -1 if any(l < 0 for l in dpath.lengths) else 1
                    self.node_list.append(goal_v) 
                    return 50000.0 
            
            cost = (waypoints_left * 20.0) + (min_wp_dist * 15.0)
            return -cost

        r = self.sim_v(q.child_v, d)
        q.n += 1
        q.Q += (r - q.Q) / q.n
        q.parent.N += 1
        return r

    def plan_step(self, iterations=50):
        # [TỐI ƯU NON-BLOCKING]: Chỉ chạy số lượng nhánh được chỉ định mỗi Frame
        for _ in range(iterations): 
            self.sim_v(self.root, 1000) 
        
        # In log để theo dõi tiến độ đâm rễ
        max_wp = max([n.wp_idx for n in self.node_list])
        if __import__("random").random() < 0.15: 
            print(f"⏳ MCPP đang đâm rễ: Chạm Waypoint {max_wp}/{len(self.waypoints)-1}")
            
        best_node = None
        for node in self.node_list:
            if node.is_dubins:
                best_node = node
                break
            elif (len(self.waypoints) - 1 - node.wp_idx) <= 2 and dist(node.state[:2], self.goal_pos) < GOAL_RADIUS:
                if best_node is None: best_node = node
        
        if best_node and best_node != self.root:
            return self.extract_path(best_node)
        return None

    def extract_path(self, node):
        full_path = []
        curr = node
        while curr and curr.parent_node:
            points = list(zip(curr.path_x, curr.path_y, curr.path_yaw))
            full_path.insert(0, {'points': points, 'is_dubins': curr.is_dubins, 'direction': curr.direction})
            curr = curr.parent_node
        return full_path