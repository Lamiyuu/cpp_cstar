import os
import random
import math
import numpy as np
from config import *
from core_math import point_in_polygon, get_car_corners, point_to_segment_dist

class DynamicObstacle:
    def __init__(self, x, y, radius, vx, vy):
        self.x, self.y = x, y
        self.radius = radius
        self.vx, self.vy = vx, vy

    def move(self, dt, bounds, outer_poly, holes, robot_state, robot_safe_radius, goal_pos=None, goal_radius=0.0):
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        
        min_x, max_x, min_y, max_y = bounds
        hit_bound = False
        if nx - self.radius < min_x or nx + self.radius > max_x:
            self.vx *= -1; hit_bound = True
        if ny - self.radius < min_y or ny + self.radius > max_y:
            self.vy *= -1; hit_bound = True
            
        if hit_bound:
            nx = self.x + self.vx * dt; ny = self.y + self.vy * dt

        hit_solid = False
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        check_pts = [(nx + self.radius * math.cos(a), ny + self.radius * math.sin(a)) for a in angles]
        check_pts.append((nx, ny))
        
        # 1. Va chạm Đa giác tĩnh
        for pt in check_pts:
            if outer_poly and not point_in_polygon(pt, outer_poly): 
                hit_solid = True; break
            for h in holes:
                if point_in_polygon(pt, h):
                    hit_solid = True; break
            if hit_solid: break
                    
        # 2. Va chạm với ĐÍCH ĐẾN (MỚI THÊM)
        if not hit_solid and goal_pos is not None:
            # Nếu khoảng cách từ tâm bóng tới tâm đích < tổng 2 bán kính -> Bật nảy
            if math.hypot(nx - goal_pos[0], ny - goal_pos[1]) < (self.radius + goal_radius):
                hit_solid = True

        # (Đã tắt va chạm bóng nảy vào xe theo yêu cầu từ trước)

        if hit_solid:
            self.vx *= -1; self.vy *= -1
            angle = math.atan2(self.vy, self.vx) + random.uniform(-0.5, 0.5)
            speed = math.hypot(self.vx, self.vy)
            self.vx = math.cos(angle) * speed
            self.vy = math.sin(angle) * speed
            nx = self.x; ny = self.y

        self.x = nx; self.y = ny

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
                    if len(curr)>2: holes.append(curr); curr = []
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

# THÊM BIẾN t_lookahead=1.5 VÀO ĐÂY
def check_collision_with_index(x, y, yaw, outer, holes, dyn_obs=None, t_lookahead=1.5):
    corners = get_car_corners(x, y, yaw)
    # Va chạm Tĩnh
    for p in corners:
        if outer and not point_in_polygon(p, outer): return True, -1
        for i, h in enumerate(holes):
            if point_in_polygon(p, h): return True, i
            
    # Va chạm Động
    if dyn_obs:
        safe_radius = math.hypot(CAR_L/2 + 1.0, CAR_WIDTH/2) + 1.0 
        for obs in dyn_obs:
            ax, ay = obs.x, obs.y
            bx, by = obs.x + obs.vx * t_lookahead, obs.y + obs.vy * t_lookahead
            # Dùng t_lookahead linh hoạt để nhận biết độ nguy hiểm
            if point_to_segment_dist(x, y, ax, ay, bx, by) < obs.radius + safe_radius:
                return True, -3 
    return False, -2

def check_path_collision(path_x, path_y, path_yaw, outer, holes, dyn_obs=None):
    step = 5 
    for i in range(0, len(path_x), step):
        # Mặc định quét đường tương lai luôn dùng 1.5 giây
        collided, _ = check_collision_with_index(path_x[i], path_y[i], path_yaw[i], outer, holes, dyn_obs, 1.5)
        if collided: return True
    return False