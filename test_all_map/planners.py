import random
import math
import numpy as np
from config import *
from core_math import dist, normalize_angle, simulate_step, reeds_shepp_planning
from environment import check_path_collision
import heapq

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

    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds, waypoints, dyn_obs=None):
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
        
        # BỔ SUNG TỪ BÀI BÁO: Dữ liệu dẫn đường
        self.waypoints = waypoints
        self.current_wp_idx = 0

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
        # 1. TÌM ĐIỂM NHÌN TRƯỚC ĐỘC LẬP (SỬA LỖI XUNG ĐỘT NHÁNH MCPP)
        # ==========================================================
        # Tìm waypoint gần xe nhất (Tính lại từ đầu cho mỗi Node)
        min_dist = float('inf')
        closest_idx = 0
        for i, wp in enumerate(self.waypoints):
            d = math.hypot(cx - wp[0], cy - wp[1])
            if d < min_dist:
                min_dist = d
                closest_idx = i
                
        # Nhìn xa ra phía trước một khoảng an toàn (Look-ahead)
        target_idx = closest_idx
        look_ahead_dist = 6.0 # 6.0 là cự ly vàng để bo cua
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
            
        # ==========================================================
        # 2. TÍNH GÓC LÁI LÝ TƯỞNG (PURE PURSUIT)
        # ==========================================================
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

        # ==========================================================
        # 3. CHIẾN THUẬT QUÉT GÓC LÁI (THAY VÌ RANDOM 1 LẦN DỄ CHẾT)
        # Tạo danh sách các hướng nên thử, ưu tiên đi mượt trước, lách sau
        # ==========================================================
        candidates = []
        
        # Nhóm 1: Đi sát đường cam nhất có thể (Băm nhỏ sai số)
        steer_offsets = [0.0, 0.05, -0.05, 0.15, -0.15, 0.3, -0.3, 0.5, -0.5]
        for offset in steer_offsets:
            s = max(-MAX_STEER, min(MAX_STEER, ideal_steer + offset))
            candidates.append((s, direction))
            
        # Nhóm 2: Hết lái để ôm cua gắt vòng rộng hoặc thoát kẹt
        candidates.append((MAX_STEER, direction))
        candidates.append((-MAX_STEER, direction))
        
        # Nhóm 3: Bí quá thì gài số lùi 1 nhịp để "xào" xe lại ngay
        candidates.append((-ideal_steer, -direction)) 

        # ==========================================================
        # 4. TÌM VÀ TRẢ VỀ NHÁNH SỐNG ĐẦU TIÊN
        # ==========================================================
        for steer, dir_val in candidates:
            steer = round(steer, 2)
            action_key = (steer, dir_val)
            
            # Bỏ qua nếu nhánh này đã được sinh ra ở Node này rồi
            if action_key in v.children:
                continue
                
            # Mô phỏng thử 3 bước nhỏ cho chính xác
            nx, ny, nyaw, px, py, pyaw = self.macro_step(cx, cy, cyaw, steer, dir_val, num_steps=3)
            
            # NẾU KHÔNG ĐÂM TƯỜNG -> SINH RỄ VÀ TRẢ VỀ NGAY
            if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                qnode = self.QNode(v, action_key)
                v.children[action_key] = qnode
                return action_key
                
        # Chỉ trả về None khi xe thực sự bị bao vây (Mọi góc lái đều đâm tường)
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
            
            # --- TRÍ NHỚ CHỐNG KẸT ---
            gid = (int(nx // 5.0), int(ny // 5.0))
            self.grid_visits[gid] = self.grid_visits.get(gid, 0) + 1
            
            # --- BẮT DUBINS (Giữ nguyên) ---
            dist_to_goal = dist((nx, ny), self.goal_pos)
            if dist_to_goal < 20.0:  
                dpath = reeds_shepp_planning(nx, ny, nyaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
                if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
                    goal_v = self.VNode((self.goal_pos[0], self.goal_pos[1], self.goal_yaw), is_dubins=True)
                    goal_v.parent_node = q.child_v
                    goal_v.path_x, goal_v.path_y, goal_v.path_yaw = dpath.x, dpath.y, dpath.yaw
                    goal_v.direction = -1 if any(l < 0 for l in dpath.lengths) else 1
                    self.node_list.append(goal_v) 
                    return 50000.0 
            
            # =========================================================
            # ĐỒNG BỘ MCPP VỚI KINEMATIC (THAY THẾ HOÀN TOÀN REPULSION)
            # Ép xe đánh giá nhánh rễ dựa trên tiến độ bám đường cam
            # =========================================================
            min_wp_dist = float('inf')
            closest_idx = 0
            for i, wp in enumerate(self.waypoints):
                d_wp = math.hypot(nx - wp[0], ny - wp[1])
                if d_wp < min_wp_dist:
                    min_wp_dist = d_wp
                    closest_idx = i
                    
            # Đếm số điểm mồi còn lại (Càng ít chứng tỏ tiến càng sâu)
            waypoints_left = len(self.waypoints) - closest_idx
            
            # Hàm Chi phí Mới: Phạt nếu lùi lại phía sau (waypoints_left lớn) 
            # HOẶC phạt nếu chệch khỏi vạch cam (min_wp_dist lớn)
            cost = (waypoints_left * 20.0) + (min_wp_dist * 15.0)

            # Phạt đi lặp lại ô cũ để chống kẹt
            if self.grid_visits[gid] > 1:
                cost += self.grid_visits[gid] * 50.0

            return -cost

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