import os
import random
import math
import numpy as np
from config import *
from core_math import point_in_polygon, get_car_corners, point_to_segment_dist

# ==========================================
# THÊM MỚI: TOÁN HỌC CHỐNG "ĐI XUYÊN TƯỜNG" 
# ==========================================
def line_intersect(A, B, C, D):
    """Kiểm tra xem đoạn thẳng AB có cắt đoạn thẳng CD không"""
    def ccw(p1, p2, p3):
        return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def get_aabb(poly):
    """Tính toán hộp chữ nhật bao quanh đa giác"""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), max(xs), min(ys), max(ys)

def aabb_intersect(box1, box2):
    """Lọc nhanh xem 2 hộp chữ nhật có chạm nhau không"""
    return not (box1[1] < box2[0] or box1[0] > box2[1] or box1[3] < box2[2] or box1[2] > box2[3])

def poly_intersect(poly1, poly2):
    """Kiểm tra xem 2 đa giác có cắt cạnh hoặc bao trùm nhau không"""
    if not poly1 or not poly2: return False
    
    # --- BỘ LỌC NHANH (AABB) - GIÚP TĂNG TỐC CODE LÊN GẤP 10 LẦN ---
    if not aabb_intersect(get_aabb(poly1), get_aabb(poly2)):
        return False # Ở quá xa nhau -> An toàn ngay lập tức!
        
    # Nếu hộp ảo chạm nhau, mới dùng Toán học phức tạp để check chi tiết
    for i in range(len(poly1)):
        A = poly1[i]; B = poly1[(i+1)%len(poly1)]
        for j in range(len(poly2)):
            C = poly2[j]; D = poly2[(j+1)%len(poly2)]
            if line_intersect(A, B, C, D): return True
            
    if point_in_polygon(poly1[0], poly2): return True
    if point_in_polygon(poly2[0], poly1): return True
    return False

# ==========================================
# GIỮ NGUYÊN CODE CŨ CỦA BẠN (TỪ ĐÂY TRỞ XUỐNG)
# ==========================================
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
                    
        # 2. Va chạm với ĐÍCH ĐẾN
        if not hit_solid and goal_pos is not None:
            if math.hypot(nx - goal_pos[0], ny - goal_pos[1]) < (self.radius + goal_radius):
                hit_solid = True

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

# ==========================================
# ĐÃ CẬP NHẬT: HÀM CHECK VA CHẠM THÔNG MINH
# ==========================================
def check_collision_with_index(x, y, yaw, outer, holes, dyn_obs=None, t_lookahead=1.5):
    # Lấy tọa độ các góc xe (Nên giảm CAR_W, CAR_L trong config một chút)
    car_poly = get_car_corners(x, y, yaw)
    
    car_x = [pt[0] for pt in car_poly]
    car_y = [pt[1] for pt in car_poly]
    c_min_x, c_max_x = min(car_x), max(car_x)
    c_min_y, c_max_y = min(car_y), max(car_y)

    # 1. Kiểm tra lọt ra ngoài bản đồ
    for p in car_poly:
        if outer and not point_in_polygon(p, outer): 
            return True, -1
        
    # 2. Kiểm tra va chạm vạch kẻ (Holes)
    for i, h in enumerate(holes):
        # SỬA LỖI TẠI ĐÂY: Lấy khung bao của từng Hole (vạch kẻ)
        h_xs = [pt[0] for pt in h]
        h_ys = [pt[1] for pt in h] # Đã sửa từ p thành pt
        
        h_min_x, h_max_x = min(h_xs), max(h_xs)
        h_min_y, h_max_y = min(h_ys), max(h_ys)
        
        # Kiểm tra AABB (Nếu hộp bao không chạm nhau thì bỏ qua cho nhanh)
        if not (c_max_x < h_min_x or c_min_x > h_max_x or 
                c_max_y < h_min_y or c_min_y > h_max_y):
            
            # Nếu hộp bao chạm, kiểm tra chi tiết đa giác
            if poly_intersect(car_poly, h): 
                return True, i
            
    # 3. Kiểm tra va chạm động (Giữ nguyên logic của bạn)
    if dyn_obs:
        safe_radius = 10.0 # Tăng bán kính quét động lên một chút cho an toàn
        for obs in dyn_obs:
            if math.hypot(x - obs.x, y - obs.y) < safe_radius:
                ax, ay = obs.x, obs.y
                bx, by = obs.x + obs.vx * t_lookahead, obs.y + obs.vy * t_lookahead
                if point_to_segment_dist(x, y, ax, ay, bx, by) < obs.radius + 2.0:
                    return True, -3 
                    
    return False, -2

def check_path_collision(path_x, path_y, path_yaw, outer, holes, dyn_obs=None):
    # ĐÃ SỬA: step = 2 để quét đường Dubins kĩ hơn, không bị lọt vạch kẻ
    step = 2 
    for i in range(0, len(path_x), step):
        collided, _ = check_collision_with_index(path_x[i], path_y[i], path_yaw[i], outer, holes, dyn_obs, 1.5)
        if collided: return True
    return False