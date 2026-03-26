import random
import math
import numpy as np
from config import *
from core_math import dist, normalize_angle, simulate_step, reeds_shepp_planning
from environment import check_path_collision

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

    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs=None):
        self.root = self.VNode(start)
        self.goal_pos = goal_pos
        self.goal_yaw = goal_yaw
        self.outer = outer
        self.known_holes = known_holes
        self.bounds = bounds 
        self.dyn_obs = dyn_obs
        self.node_list = [self.root] 
        self.grid_visits = {}
        self.grid_penalties = {}

    def get_dist_to_nearest_obstacle(self, state):
        min_d = 50.0 
        px, py = state[0], state[1]
        for hole in self.known_holes:
            for vertex in hole:
                d = math.sqrt((px - vertex[0])**2 + (py - vertex[1])**2)
                if d < min_d: min_d = d
        return min_d
    
    def macro_step(self, sx, sy, syaw, steer, direction, num_steps=4):
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
            
            if q.child_v:
                gid = (int(q.child_v.state[0] // 5.0), int(q.child_v.state[1] // 5.0))
                s -= self.grid_penalties.get(gid, 0)
                
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        cx, cy, cyaw = v.state[0], v.state[1], v.state[2]
        
        # ==========================================================
        # 1. TÌM "ĐIỂM MỒI" (WAYPOINT) ĐỂ KÉO XE THOÁT CHUỒNG
        # ==========================================================
        if random.random() < 0.2: 
            # 20% nhắm thẳng đích
            target_x, target_y = self.goal_pos[0], self.goal_pos[1]
        else:
            safe_target_found = False
            # Thử tối đa 10 lần để tìm ra một điểm nằm chình ình giữa đường
            for _ in range(10):
                # TỐI ƯU CỦA LÂM: Không lấy bừa toàn map nữa, chỉ lấy xung quanh xe 
                # (bán kính 40 đơn vị) để tạo lực hút cục bộ kéo xe ra khỏi chuồng
                tx = cx + random.uniform(-15.0, 15.0)
                ty = cy + random.uniform(-15.0, 15.0)
                
                # Đảm bảo điểm không văng ra ngoài sa bàn
                tx = max(self.bounds[0], min(self.bounds[1], tx))
                ty = max(self.bounds[2], min(self.bounds[3], ty))
                
                # ĐIỀU KIỆN VÀNG: Điểm này PHẢI nằm giữa hành lang (cách vạch > 12 đơn vị)
                if self.get_dist_to_nearest_obstacle((tx, ty)) > 8.0:
                    target_x, target_y = tx, ty
                    safe_target_found = True
                    break # Tìm thấy điểm an toàn là chốt luôn!
            
            if not safe_target_found:
                # Nếu xui xẻo 10 lần không tìm được, lấy đại một điểm để tránh lỗi
                target_x = random.uniform(self.bounds[0], self.bounds[1])
                target_y = random.uniform(self.bounds[2], self.bounds[3])
                
        # ==========================================================
        # 2. TÍNH TOÁN GÓC LÁI TỰ NHIÊN 
        # ==========================================================
        dx = target_x - cx
        dy = target_y - cy
        angle_to_target = math.atan2(dy, dx)
        
        # Tính góc lệch nếu đi tiến (diff_head) và nếu đi lùi (diff_tail)
        diff_head = normalize_angle(angle_to_target - cyaw)
        diff_tail = normalize_angle(angle_to_target - (cyaw + math.pi))
        
        # ==========================================================
        # 3. LOGIC SỐ TIẾN / SỐ LÙI THÔNG MINH (Chống kẹt chuồng)
        # ==========================================================
        # Nếu "điểm mồi" an toàn đang nằm ở phía sau lưng xe (góc > 90 độ)
        if abs(diff_head) > math.pi / 2.0: 
            direction = -1        # Tự động gài số lùi để bám theo điểm mồi
            diff = diff_tail
        else:
            direction = 1         # Gài số tiến
            diff = diff_head
            
        # Thỉnh thoảng (20%) vẫn phá lệ random số tiến/lùi để đảm bảo tính đa dạng của MCTS
        if random.random() < 0.2:
            direction = -1 if random.random() < PROB_REVERSE else 1
            diff = diff_tail if direction == -1 else diff_head
            
        steer = max(-MAX_STEER, min(MAX_STEER, diff))
        
        # ==========================================================
        # 4. MÔ PHỎNG VÀ THÊM VÀO CÂY
        # ==========================================================
        nx, ny, nyaw, px, py, pyaw = self.macro_step(cx, cy, cyaw, steer, direction, num_steps=4)
        
        if check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
            return None
        
        action_key = (round(steer, 2), direction)
        if action_key not in v.children:
            qnode = self.QNode(v, action_key)
            v.children[action_key] = qnode
            return action_key
        return None

    def sim_v(self, v, d):
        sx, sy, syaw = v.state[0], v.state[1], v.state[2]
        dist_to_goal = dist((sx, sy), self.goal_pos)

        # ĐIỀU KIỆN DỪNG
        if d <= 0 or dist_to_goal < GOAL_RADIUS:
            return -(20.0 * dist_to_goal) 

        # EXPANSION
        if len(v.children) < MCPP_BRANCHES:
            act = self.expand(v)
            if act:
                return self.sim_q(v.children[act], d - 1)
        
        # SELECTION
        res = self.get_action_ucb(v)
        if res:
            return self.sim_q(res[1], d - 1)
            
        return -dist_to_goal

    def sim_q(self, q, d):
        if not q.child_v:
            steer, direction = q.action
            nx, ny, nyaw, px, py, pyaw = self.macro_step(
                q.parent.state[0], q.parent.state[1], q.parent.state[2], 
                steer, direction, num_steps=3
            )
            
            q.child_v = self.VNode((nx, ny, nyaw), is_dubins=False, direction=direction)
            q.child_v.parent_node = q.parent
            q.child_v.path_x, q.child_v.path_y, q.child_v.path_yaw = px, py, pyaw
            self.node_list.append(q.child_v)
            
            # --- TRÍ NHỚ CHỐNG KẸT (Tăng đô) ---
            gid = (int(nx // 5.0), int(ny // 5.0))
            self.grid_visits[gid] = self.grid_visits.get(gid, 0) + 1
            # Ép phạt 50.0 điểm cho mỗi lần loanh quanh ô cũ!
            self.grid_penalties[gid] = self.grid_visits[gid] * 50.0 
            
            dist_to_goal = dist((nx, ny), self.goal_pos)
            
            # --- BẮT DUBINS ---
            if dist_to_goal < 20.0:  
                dpath = reeds_shepp_planning(nx, ny, nyaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
                if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
                    goal_v = self.VNode((self.goal_pos[0], self.goal_pos[1], self.goal_yaw), is_dubins=True)
                    goal_v.parent_node = q.child_v
                    goal_v.path_x, goal_v.path_y, goal_v.path_yaw = dpath.x, dpath.y, dpath.yaw
                    goal_v.direction = -1 if any(l < 0 for l in dpath.lengths) else 1
                    self.node_list.append(goal_v) 
                    return 50000.0 
            
            # --- HÀNH LANG AN TOÀN (Thu hẹp lại cho vừa map) ---
            obs_dist = self.get_dist_to_nearest_obstacle((nx, ny))
            repulsion = 0.0
            
            if dist_to_goal > 25.0:
                SAFE_DIST = 8.0 # Giảm từ 14 xuống 8 để có khoảng hở giữa hành lang
                if obs_dist < SAFE_DIST:
                    repulsion = 30.0 * (SAFE_DIST - obs_dist)**2

            return -(10.0 * dist_to_goal + repulsion)

        r = self.sim_v(q.child_v, d)
        q.n += 1
        q.Q += (r - q.Q) / q.n
        q.parent.N += 1
        return r

    def plan_step(self):
        for _ in range(30): 
            self.sim_v(self.root, MCPP_DEPTH)
        
        best_node = None
        for node in self.node_list:
            if node.is_dubins:
                best_node = node
                break
            elif dist(node.state[:2], self.goal_pos) < GOAL_RADIUS:
                if best_node is None: best_node = node
        
        if best_node:
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