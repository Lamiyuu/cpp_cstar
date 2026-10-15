import math
import random
import numpy as np
from core_math import dist, normalize_angle, simulate_step
from environment import check_path_collision

# Hằng số Năng lượng Kinematic
C_LINEAR = 1.0
C_STEER = 2.5
C_SWITCH = 15.0

class KinematicNode:
    def __init__(self, x, y, yaw, parent=None, direction=1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent = parent
        self.cost = 0.0 # Lưu trữ Năng lượng tiêu thụ
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.is_dubins = False # Đã bỏ Dubins/RS
        self.direction = direction

class PaperPlannerKinematicBase:
    def __init__(self, start_state, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs=None):
        self.start = KinematicNode(start_state[0], start_state[1], start_state[2])
        self.goal_pos = goal_pos
        self.goal_yaw = goal_yaw
        self.outer = outer
        self.known_holes = known_holes
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.dyn_obs = dyn_obs
        
        self.node_list_start = [self.start]
        self.node_list_goal = [KinematicNode(goal_pos[0], goal_pos[1], goal_yaw)]
        self.solved_node = None
        self.max_steer = 0.5
        self.connect_dist = 10.0 # Ngưỡng khoảng cách để coi là kết nối thành công (Thuần RRT)

    def calc_kinematic_energy(self, parent_node, path_x, path_y, path_yaw, direction):
        e_linear, e_steer, e_switch = 0.0, 0.0, 0.0
        for i in range(1, len(path_x)):
            dx, dy = path_x[i] - path_x[i-1], path_y[i] - path_y[i-1]
            e_linear += math.hypot(dx, dy)
            e_steer += abs(normalize_angle(path_yaw[i] - path_yaw[i-1]))
        if parent_node and parent_node.direction != direction:
            e_switch = C_SWITCH
        return (e_linear * C_LINEAR) + (e_steer * C_STEER) + e_switch

    def sample_free(self):
        if random.random() < 0.15:  
            return (self.goal_pos[0], self.goal_pos[1])
        return (random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))

    def nearest(self, tree, pt):
        dists = [(n.x - pt[0])**2 + (n.y - pt[1])**2 for n in tree]
        return tree[dists.index(min(dists))]

    def steer_kinematic(self, nearest, target_pt, step_factor=1.0):
        dx = target_pt[0] - nearest.x
        dy = target_pt[1] - nearest.y
        target_yaw = math.atan2(dy, dx)
        diff_head = normalize_angle(target_yaw - nearest.yaw)
        diff_tail = normalize_angle(target_yaw - (nearest.yaw + math.pi))
        
        direction = -1 if random.random() < 0.2 else 1
        diff = diff_tail if direction == -1 else diff_head
        steer = max(-self.max_steer, min(self.max_steer, diff))
        
        nx, ny, nyaw, px, py, pyaw = simulate_step(nearest.x, nearest.y, nearest.yaw, steer, direction)
        
        if step_factor < 1.0:
            keep_len = max(1, int(len(px) * step_factor))
            px, py, pyaw = px[:keep_len], py[:keep_len], pyaw[:keep_len]
            nx, ny, nyaw = px[-1], py[-1], pyaw[-1]
            
        return nx, ny, nyaw, px, py, pyaw

    def extract_path(self, node):
        full_path = []
        while node.parent:
            points = list(zip(node.path_x, node.path_y, node.path_yaw))
            full_path.insert(0, {'points': points, 'is_dubins': False, 'direction': node.direction})
            node = node.parent
        return full_path

    def extract_bidirectional_path(self, node_from_start, node_from_goal):
        """Hợp nhất 2 cây RRT kèm theo một đoạn cong Bezier nối mượt mà ở giữa"""
        # 1. Trích xuất nửa đường đi từ Cây Xuất Phát
        path_start = self.extract_path(node_from_start)
        
        # 2. TẠO ĐOẠN CẦU NỐI ĐỘNG HỌC (BRIDGE SEGMENT) GIỮA 2 NODE
        d_gap = math.hypot(node_from_goal.x - node_from_start.x, node_from_goal.y - node_from_start.y)
        bridge_segment = []
        
        # Nếu 2 điểm nối không trùng khít nhau, sinh đường cong Bezier để lấp khoảng trống
        if d_gap > 0.1:
            steps = max(100, int(d_gap / 1.0)) # Độ phân giải: 1 pixel/điểm để xe chạy mượt
            k = d_gap / 2.0
            
            # Vector tiếp tuyến tại điểm ra của Cây Xuất Phát
            p0_x, p0_y = node_from_start.x, node_from_start.y
            p1_x = p0_x + k * math.cos(node_from_start.yaw) * node_from_start.direction
            p1_y = p0_y + k * math.sin(node_from_start.yaw) * node_from_start.direction
            
            # Vector tiếp tuyến tại điểm vào của Cây Đích
            p3_x, p3_y = node_from_goal.x, node_from_goal.y
            p2_x = p3_x + k * math.cos(node_from_goal.yaw) * node_from_goal.direction
            p2_y = p3_y + k * math.sin(node_from_goal.yaw) * node_from_goal.direction
            
            bx, by, byaw = [], [], []
            ydiff = normalize_angle(node_from_goal.yaw - node_from_start.yaw)
            
            # Tính toán các điểm trên đường cong Bezier
            for i in range(steps):
                t = i / (steps - 1)
                mt = 1 - t
                x = (mt**3)*p0_x + 3*(mt**2)*t*p1_x + 3*mt*(t**2)*p2_x + (t**3)*p3_x
                y = (mt**3)*p0_y + 3*(mt**2)*t*p1_y + 3*mt*(t**2)*p2_y + (t**3)*p3_y
                
                bx.append(x)
                by.append(y)
                byaw.append(normalize_angle(node_from_start.yaw + t * ydiff))
                
            bridge_segment = [{'points': list(zip(bx, by, byaw)), 'is_dubins': False, 'direction': node_from_start.direction}]

        # 3. Trích xuất nửa đường đi từ Cây Đích (Goal Tree)
        path_goal = []
        curr = node_from_goal
        while curr.parent:
            # Lật ngược mảng tọa độ để đi từ điểm giao tiến dần về đích
            points = list(zip(curr.path_x[::-1], curr.path_y[::-1], curr.path_yaw[::-1]))
            
            # ÉP ĐI LÙI (-curr.direction) để giữ chuẩn động học
            path_goal.append({'points': points, 'is_dubins': False, 'direction': -curr.direction})
            curr = curr.parent
            
        # Hợp nhất: Nửa đầu + Đoạn nối mượt + Nửa cuối
        return path_start + bridge_segment + path_goal
# ==========================================
# 1. RRT*
# ==========================================
class RRTStar(PaperPlannerKinematicBase):
    def plan_step(self, iterations=50):
        if self.solved_node: return self.extract_path(self.solved_node)
        for _ in range(iterations):
            rnd = self.sample_free()
            nearest = self.nearest(self.node_list_start, rnd)
            nx, ny, nyaw, px, py, pyaw = self.steer_kinematic(nearest, rnd)
            
            if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                new_n = KinematicNode(nx, ny, nyaw, nearest)
                new_n.path_x, new_n.path_y, new_n.path_yaw = px, py, pyaw
                new_n.cost = nearest.cost + self.calc_kinematic_energy(nearest, px, py, pyaw, new_n.direction)
                self.node_list_start.append(new_n)
                
                # Thuần RRT: Chạm mốc gần đích là hoàn thành
                if dist((nx, ny), self.goal_pos) <= self.connect_dist:
                    self.solved_node = new_n
                    return self.extract_path(self.solved_node)
        return None

# ==========================================
# 2. BI-RRT*
# ==========================================
class BiRRTStar(PaperPlannerKinematicBase):
    def plan_step(self, iterations=50):
        if self.solved_node: return self.solved_node
        for _ in range(iterations):
            for t_src, t_tgt in [(self.node_list_start, self.node_list_goal), (self.node_list_goal, self.node_list_start)]:
                rnd = self.sample_free()
                nearest = self.nearest(t_src, rnd)
                nx, ny, nyaw, px, py, pyaw = self.steer_kinematic(nearest, rnd)
                
                if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                    new_n = KinematicNode(nx, ny, nyaw, nearest)
                    new_n.path_x, new_n.path_y, new_n.path_yaw = px, py, pyaw
                    t_src.append(new_n)
                    
                    tgt_near = self.nearest(t_tgt, (nx, ny))
                    if dist((nx, ny), (tgt_near.x, tgt_near.y)) <= self.connect_dist:
                        
                        # CHỐT GÓC ĐỒNG BỘ:
                        yaw_diff = abs(normalize_angle(tgt_near.yaw - nyaw))
                        if yaw_diff < 0.5 or abs(yaw_diff - math.pi) < 0.5:
                            
                            # SINH NODE CẦU NỐI ĐỘNG HỌC (BRIDGE NODE):
                            fnx, fny, fnyaw, fpx, fpy, fpyaw = self.steer_kinematic(new_n, (tgt_near.x, tgt_near.y))
                            
                            if not check_path_collision(fpx, fpy, fpyaw, self.outer, self.known_holes, self.dyn_obs):
                                bridge_node = KinematicNode(fnx, fny, fnyaw, new_n)
                                bridge_node.path_x, bridge_node.path_y, bridge_node.path_yaw = fpx, fpy, fpyaw
                                
                                if t_src is self.node_list_start:
                                    self.solved_node = self.extract_bidirectional_path(bridge_node, tgt_near)
                                else:
                                    self.solved_node = self.extract_bidirectional_path(tgt_near, bridge_node)
                                return self.solved_node
        return None

# ==========================================
# 3. GB-RRT* 
# ==========================================
class GBRRTStar(RRTStar):
    def sample_free(self):
        if random.random() < 0.40: return (self.goal_pos[0], self.goal_pos[1])
        return super().sample_free()

# ==========================================
# 4. APF-RRT* 
# ==========================================
class APFRRTStar(RRTStar):
    def get_apf_target(self, nearest, rnd):
        # Tính khoảng cách tới đích hiện tại
        d_goal = math.hypot(self.goal_pos[0] - nearest.x, self.goal_pos[1] - nearest.y)
        
        f_x = 1.0 * (self.goal_pos[0] - nearest.x) + 0.5 * (rnd[0] - nearest.x)
        f_y = 1.0 * (self.goal_pos[1] - nearest.y) + 0.5 * (rnd[1] - nearest.y)
        
        for h in self.known_holes:
            cx, cy = sum(p[0] for p in h)/len(h), sum(p[1] for p in h)/len(h)
            d = math.hypot(nearest.x - cx, nearest.y - cy)
            if 0.1 < d <= 60.0:
                # Hệ số triệt tiêu lực đẩy: Càng gần đích, lực đẩy từ vật cản càng yếu đi
                scale_factor = (d_goal / 60.0) ** 2 if d_goal < 60.0 else 1.0
                
                rep = 2000.0 * (1.0/d - 1.0/60.0) / (d**2) * scale_factor
                f_x += rep * (nearest.x - cx) / d
                f_y += rep * (nearest.y - cy) / d
        return (nearest.x + f_x, nearest.y + f_y)

    def plan_step(self, iterations=50):
        if self.solved_node: return self.extract_path(self.solved_node)
        for _ in range(iterations):
            rnd = self.sample_free()
            nearest = self.nearest(self.node_list_start, rnd)
            apf_target = self.get_apf_target(nearest, rnd)
            nx, ny, nyaw, px, py, pyaw = self.steer_kinematic(nearest, apf_target)
            
            if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                new_n = KinematicNode(nx, ny, nyaw, nearest)
                new_n.path_x, new_n.path_y, new_n.path_yaw = px, py, pyaw
                self.node_list_start.append(new_n)
                
                # Đã tiến vào vùng ngưỡng connect_dist
                if dist((nx, ny), self.goal_pos) <= self.connect_dist:
                    
                    # ÉP ĐƯỜNG NỐI BEZIER CHUẨN VÀO TÂM ĐÍCH
                    final_n = KinematicNode(self.goal_pos[0], self.goal_pos[1], self.goal_yaw, new_n)
                    
                    d_gap = dist((nx, ny), self.goal_pos)
                    steps = max(30, int(d_gap / 0.5))
                    k = d_gap / 2.0
                    
                    p0_x, p0_y = nx, ny
                    p1_x = nx + k * math.cos(nyaw) * new_n.direction
                    p1_y = ny + k * math.sin(nyaw) * new_n.direction
                    
                    p3_x, p3_y = self.goal_pos[0], self.goal_pos[1]
                    p2_x = self.goal_pos[0] - k * math.cos(self.goal_yaw) * 1.0 
                    p2_y = self.goal_pos[1] - k * math.sin(self.goal_yaw) * 1.0
                    
                    bx, by, byaw = [], [], []
                    yaw_diff = normalize_angle(self.goal_yaw - nyaw)
                    
                    for i in range(steps):
                        t = i / (steps - 1)
                        mt = 1 - t
                        x = (mt**3)*p0_x + 3*(mt**2)*t*p1_x + 3*mt*(t**2)*p2_x + (t**3)*p3_x
                        y = (mt**3)*p0_y + 3*(mt**2)*t*p1_y + 3*mt*(t**2)*p2_y + (t**3)*p3_y
                        bx.append(x)
                        by.append(y)
                        byaw.append(normalize_angle(nyaw + t * yaw_diff))
                        
                    # Quét va chạm đoạn nối Bezier cuối
                    if not check_path_collision(bx, by, byaw, self.outer, self.known_holes, self.dyn_obs):
                        final_n.path_x, final_n.path_y, final_n.path_yaw = bx, by, byaw
                        self.solved_node = final_n
                    else:
                        self.solved_node = new_n 
                        
                    return self.extract_path(self.solved_node)
        return None

# ==========================================
# 5. BI-APF-RRT*
# ==========================================
class BiAPFRRTStar(BiRRTStar):
    def get_apf_target(self, nearest, rnd, target_coord):
        # Sửa lỗi GNRON (Triệt tiêu lực đẩy khi tới gần chướng ngại vật)
        d_tgt = math.hypot(target_coord[0] - nearest.x, target_coord[1] - nearest.y)
        
        f_x = 1.0 * (target_coord[0] - nearest.x) + 0.5 * (rnd[0] - nearest.x)
        f_y = 1.0 * (target_coord[1] - nearest.y) + 0.5 * (rnd[1] - nearest.y)
        for h in self.known_holes:
            cx, cy = sum(p[0] for p in h)/len(h), sum(p[1] for p in h)/len(h)
            d = math.hypot(nearest.x - cx, nearest.y - cy)
            if 0.1 < d <= 60.0:
                scale_factor = (d_tgt / 60.0) ** 2 if d_tgt < 60.0 else 1.0
                rep = 2000.0 * (1.0/d - 1.0/60.0) / (d**2) * scale_factor
                f_x += rep * (nearest.x - cx) / d
                f_y += rep * (nearest.y - cy) / d
        return (nearest.x + f_x, nearest.y + f_y)

    def plan_step(self, iterations=50):
        if self.solved_node: return self.solved_node
        for _ in range(iterations):
            for t_src, t_tgt in [(self.node_list_start, self.node_list_goal), (self.node_list_goal, self.node_list_start)]:
                rnd = self.sample_free()
                nearest = self.nearest(t_src, rnd)
                tgt_coord = (t_tgt[-1].x, t_tgt[-1].y)
                apf_target = self.get_apf_target(nearest, rnd, tgt_coord)
                
                nx, ny, nyaw, px, py, pyaw = self.steer_kinematic(nearest, apf_target)
                if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                    new_n = KinematicNode(nx, ny, nyaw, nearest)
                    new_n.path_x, new_n.path_y, new_n.path_yaw = px, py, pyaw
                    t_src.append(new_n)
                    
                    tgt_near = self.nearest(t_tgt, (nx, ny))
                    if dist((nx, ny), (tgt_near.x, tgt_near.y)) <= self.connect_dist:
                        
                        yaw_diff = abs(normalize_angle(tgt_near.yaw - nyaw))
                        if yaw_diff < 0.5 or abs(yaw_diff - math.pi) < 0.5:
                            
                            fnx, fny, fnyaw, fpx, fpy, fpyaw = self.steer_kinematic(new_n, (tgt_near.x, tgt_near.y))
                            
                            if not check_path_collision(fpx, fpy, fpyaw, self.outer, self.known_holes, self.dyn_obs):
                                bridge_node = KinematicNode(fnx, fny, fnyaw, new_n)
                                bridge_node.path_x, bridge_node.path_y, bridge_node.path_yaw = fpx, fpy, fpyaw
                                
                                if t_src is self.node_list_start:
                                    self.solved_node = self.extract_bidirectional_path(bridge_node, tgt_near)
                                else:
                                    self.solved_node = self.extract_bidirectional_path(tgt_near, bridge_node)
                                return self.solved_node
        return None
# ==========================================
# 6. MS-BI-RRT* (Multi Strategy Bidirectional)
# ==========================================
class MSBiRRTStar(PaperPlannerKinematicBase):
    def __init__(self, start_state, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs=None):
        super().__init__(start_state, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs)
        self.fail_count = 0
        self.fail_threshold = 15

    def get_obstacle_density(self, pt):
        count, min_dist = 0, float('inf')
        for h in self.known_holes:
            cx, cy = sum(p[0] for p in h)/len(h), sum(p[1] for p in h)/len(h)
            d = math.hypot(pt[0] - cx, pt[1] - cy)
            if d < min_dist: min_dist = d
            if d < 50.0: count += 1
        return count / 8.0, min_dist

    def multi_factor_cost(self, parent_node, path_x, path_y, path_yaw, direction, d_min, density):
        energy = self.calc_kinematic_energy(parent_node, path_x, path_y, path_yaw, direction)
        return parent_node.cost + energy + (2.0 * density) + (10.0 * math.exp(-d_min / 15.0))

    def plan_step(self, iterations=50):
        if self.solved_node: return self.solved_node
        for _ in range(iterations):
            for t_src, t_tgt in [(self.node_list_start, self.node_list_goal), (self.node_list_goal, self.node_list_start)]:
                p_bias = 1.0 if self.fail_count <= self.fail_threshold else self.fail_threshold / self.fail_count
                tgt_coord = (t_tgt[-1].x, t_tgt[-1].y)
                rnd = self.sample_free()
                nearest = self.nearest(t_src, rnd)
                
                density, d_min = self.get_obstacle_density((nearest.x, nearest.y))
                step_factor = max(0.3, math.exp(-0.6 * density)) 
                
                if random.random() <= p_bias:
                    nx, ny, nyaw, px, py, pyaw = self.steer_kinematic(nearest, tgt_coord, step_factor)
                else:
                    nx, ny, nyaw, px, py, pyaw = self.steer_kinematic(nearest, rnd, step_factor)
                    
                if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
                    self.fail_count = 0
                    new_n = KinematicNode(nx, ny, nyaw, nearest)
                    new_n.path_x, new_n.path_y, new_n.path_yaw = px, py, pyaw
                    new_n.cost = self.multi_factor_cost(nearest, px, py, pyaw, new_n.direction, d_min, density)
                    t_src.append(new_n)
                    
                    tgt_near = self.nearest(t_tgt, (nx, ny))
                    if dist((nx, ny), (tgt_near.x, tgt_near.y)) <= self.connect_dist:
                        
                        yaw_diff = abs(normalize_angle(tgt_near.yaw - nyaw))
                        if yaw_diff < 0.5 or abs(yaw_diff - math.pi) < 0.5:
                            
                            fnx, fny, fnyaw, fpx, fpy, fpyaw = self.steer_kinematic(new_n, (tgt_near.x, tgt_near.y))
                            
                            if not check_path_collision(fpx, fpy, fpyaw, self.outer, self.known_holes, self.dyn_obs):
                                bridge_node = KinematicNode(fnx, fny, fnyaw, new_n)
                                bridge_node.path_x, bridge_node.path_y, bridge_node.path_yaw = fpx, fpy, fpyaw
                                
                                if t_src is self.node_list_start:
                                    self.solved_node = self.extract_bidirectional_path(bridge_node, tgt_near)
                                else:
                                    self.solved_node = self.extract_bidirectional_path(tgt_near, bridge_node)
                                return self.solved_node
                else:
                    self.fail_count += 1
        return None