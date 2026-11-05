import pygame
import sys
import os
import glob
import math
import numpy as np
import random
import cv2  

from config import *

# TĂNG TẦM NHÌN RADAR TRỰC TIẾP TẠI ĐÂY MÀ KHÔNG CẦN SỬA CONFIG.PY
SENSOR_RADIUS = 50.0 

from core_math import get_car_corners, dist, point_to_segment_dist, point_in_polygon
from environment import get_valid_random_pos, DynamicObstacle, check_collision_with_index
from planners import KinematicRRT, KinematicMCPP, get_topological_path
from energy_tracker import EnergyTracker

# IMPORT CÁC THUẬT TOÁN TỪ BÀI BÁO
from rrt_planners import (
    RRTStar, BiRRTStar, APFRRTStar, BiAPFRRTStar, MSBiRRTStar
)

ALGO_MODES = [
    "MCPP_Original", "RRT_Original", "MS-BI-RRT*", "RRT*", 
    "BI-RRT*", "APF-RRT*", "BI-APF-RRT*"
]

# ==========================================
# XỬ LÝ ĐỌC ẢNH VÀ TRÍCH XUẤT CHƯỚNG NGẠI VẬT
# ==========================================
# ==========================================
# XỬ LÝ ĐỌC ẢNH VÀ TRÍCH XUẤT CHƯỚNG NGẠI VẬT (ĐÃ FIX LỖI CẮT ẢNH)
# ==========================================
def load_map_from_image(image_path, logical_size=500.0): 
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không tìm thấy file {image_path}")
        return [], [], None, 0, 0
        
    orig_h, orig_w = img.shape[:2]
    pixel_to_logic = logical_size / max(orig_w, orig_h)
    
    logic_w = orig_w * pixel_to_logic
    logic_h = orig_h * pixel_to_logic

    # 1. OUTER_POLY: Coi toàn bộ viền bức ảnh là giới hạn bản đồ
    outer_poly = [
        (0.0, 0.0), 
        (logic_w, 0.0), 
        (logic_w, logic_h), 
        (0.0, logic_h)
    ]

    # 2. TÌM VẬT CẢN: Lọc tất cả các màu KHÔNG PHẢI MÀU TRẮNG
    lower_white = np.array([240, 240, 240])
    upper_white = np.array([255, 255, 255])
    mask_white = cv2.inRange(img, lower_white, upper_white)
    
    # Đảo ngược mask: Vùng đen/xám của tòa nhà sẽ biến thành màu trắng để OpenCV tìm kiếm
    mask_obstacles = cv2.bitwise_not(mask_white)
    
    # Dùng RETR_EXTERNAL để lấy tất cả các cục chướng ngại vật rời rạc
    contours, hierarchy = cv2.findContours(mask_obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    real_holes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10: # Bỏ qua nhiễu ảnh quá bé
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Lật trục Y cho đúng với tọa độ của Pygame
            hole = [(float(pt[0][0]) * pixel_to_logic, logic_h - float(pt[0][1]) * pixel_to_logic) for pt in approx]
            if len(hole) >= 3:
                real_holes.append(hole)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bg_surface = pygame.surfarray.make_surface(np.transpose(img_rgb, (1, 0, 2)))
    
    print(f"Đã nạp map hoàn chỉnh. Kích thước vật lý: {logic_w:.1f}x{logic_h:.1f}m. Số vật cản: {len(real_holes)}")
    return outer_poly, real_holes, bg_surface, logic_w, logic_h


# ==========================================
# THUẬT TOÁN AMuGOPIA - SẮP XẾP ĐA MỤC TIÊU
# ==========================================
def angle_between_vectors(v1, v2):
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
        
    return ordered_path


# ==========================================
# CHƯƠNG TRÌNH CHÍNH
# ==========================================
def main():
    pygame.init()
    if not os.path.exists("Results"): os.makedirs("Results")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Stage-by-Stage Planning: Paused Dynamic Obstacles")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    map_images = sorted(glob.glob("*.jpg") + glob.glob("*.png"))
    
    current_map_idx = 0
    algo_idx = 0
    algo_mode = ALGO_MODES[algo_idx]
    
    outer_poly = []; real_holes = []
    known_hole_indices = set(); planner_holes_geom = []
    current_state = (0.0, 0.0, 0.0)
    planner = None
    
    planned_path = []; flat_planned_path = []; path_index = 0; path_history = []
    dyn_obstacles = [] 
    
    bg_surface = None
    logic_w = 0.0
    logic_h = 0.0
    scaled_bg = None
    
    multi_goals = [] 
    ordered_goals = []
    current_stage_idx = 0
    
    sim_report = None 
    need_path_check = False 
    
    click_step = 0 
    is_planning = False
    is_crashed = False 
    is_finished_all = False
    scale = 1.0

    # HÀM SPAWN SINH VIÊN (Chỉ đứng từ lề đường đi sang)
    def spawn_new_student(robot_state=None):
        zone_min_x, zone_max_x = 41.0, 479.0
        zone_min_y, zone_max_y = 25.0, 51.0
        STUDENT_RADIUS = 4.0
        STUDENT_SPEED = 5.0
        
        for _ in range(100):
            if random.choice([True, False]):
                sy = zone_min_y + 1.0 # Lề dưới
                angle = random.uniform(math.pi/2 - 0.2, math.pi/2 + 0.2)
            else:
                sy = zone_max_y - 1.0 # Lề trên
                angle = random.uniform(3*math.pi/2 - 0.2, 3*math.pi/2 + 0.2)
            
            sx = random.uniform(zone_min_x, zone_max_x)
            
            valid = True
            if outer_poly and not point_in_polygon((sx, sy), outer_poly): valid = False
            for h in real_holes:
                if point_in_polygon((sx, sy), h): valid = False
                
            # --- THÊM LOGIC SAFE ZONE ---
            # Nếu sinh viên chuẩn bị xuất hiện quá gần robot (bán kính 60px), tìm điểm khác
            if valid and robot_state is not None:
                if math.hypot(sx - robot_state[0], sy - robot_state[1]) < 60.0:
                    valid = False
            # ----------------------------
                    
            if valid:
                speed = random.uniform(STUDENT_SPEED * 0.7, STUDENT_SPEED * 1.2)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                return DynamicObstacle(sx, sy, STUDENT_RADIUS, vx, vy)
                
        return DynamicObstacle(200.0, 35.0, STUDENT_RADIUS, 0, STUDENT_SPEED)

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_state, known_hole_indices, planner_holes_geom
        nonlocal planner, planned_path, flat_planned_path, path_index, is_planning, scale, path_history, click_step
        nonlocal dyn_obstacles, is_crashed, multi_goals, ordered_goals, current_stage_idx, is_finished_all
        nonlocal bg_surface, logic_w, logic_h, sim_report, need_path_check
        nonlocal scaled_bg, top_left_x, top_left_y, offset_x, offset_y
        
        sim_report = EnergyTracker()
        need_path_check = False
        
        use_dummy = False
        if map_images:
            image_path = map_images[current_map_idx]
            if new_map: print(f"\nLoading Image Map: {image_path}")
            outer_poly, real_holes, bg_surface, logic_w, logic_h = load_map_from_image(image_path, logical_size=500.0)
            if not outer_poly: use_dummy = True
        else: use_dummy = True
        
        if use_dummy:
            outer_poly = [(0,0), (WINDOW_SIZE,0), (WINDOW_SIZE,WINDOW_SIZE), (0,WINDOW_SIZE)]
            real_holes = [[(300,300), (500,300), (500,500), (300,500)]]
            logic_w, logic_h = WINDOW_SIZE, WINDOW_SIZE
            bg_surface = None

        if logic_w > 0 and logic_h > 0:
            scale_x = (WINDOW_SIZE - 80) / logic_w
            scale_y = (WINDOW_SIZE - 80) / logic_h
            scale = min(scale_x, scale_y)
        else:
            scale = 1.0

        offset_x = (WINDOW_SIZE - (logic_w * scale)) / 2
        offset_y = (WINDOW_SIZE - (logic_h * scale)) / 2

        if bg_surface is not None:
            img_w_scr = int(logic_w * scale)
            img_h_scr = int(logic_h * scale)
            scaled_bg = pygame.transform.smoothscale(bg_surface, (img_w_scr, img_h_scr))
            top_left_x = int(offset_x)
            top_left_y = int(offset_y)
        else:
            scaled_bg = None
        
        click_step = 0; is_planning = False; planner = None; is_crashed = False; is_finished_all = False
        current_state = (0.0, 0.0, 0.0)
        
        multi_goals.clear(); ordered_goals.clear()
        current_stage_idx = 0
        known_hole_indices.clear(); planner_holes_geom.clear()
        planned_path.clear(); flat_planned_path.clear(); path_history.clear()
        path_index = 0
        
        if new_map: 
            dyn_obstacles.clear()
            if logic_w > 0 and logic_h > 0:
                NUM_STUDENTS = 0  
                for _ in range(NUM_STUDENTS):
                    dyn_obstacles.append(spawn_new_student())

    offset_x, offset_y = 40, 40
    top_left_x, top_left_y = 40, 40
    reset_sim(new_map=True)

    def to_scr(pos): 
        return int(pos[0] * scale + offset_x), int(WINDOW_SIZE - (pos[1] * scale + offset_y))
    def from_scr(sx, sy): 
        return (sx - offset_x) / scale, (WINDOW_SIZE - sy - offset_y) / scale
    
    def draw_car(state, color=CAR_COLOR):
        x, y, yaw = state
        corners = get_car_corners(x, y, yaw)
        scr_corners = [to_scr(p) for p in corners]
        pygame.draw.polygon(screen, color, scr_corners)
        pygame.draw.polygon(screen, BLACK, scr_corners, 1)
        fx = x + CAR_L * math.cos(yaw); fy = y + CAR_L * math.sin(yaw)
        pygame.draw.line(screen, BLACK, to_scr((x,y)), to_scr((fx,fy)), 2)

    def plan_next_stage(start_state, target_pos):
        bounds = [0, logic_w, 0, logic_h]
        goal_yaw = math.atan2(target_pos[1] - start_state[1], target_pos[0] - start_state[0])

        if algo_mode in ["MCPP_Original", "RRT_Original"]:
            def get_clearance_for_astar(pt):
                if outer_poly and not point_in_polygon(pt, outer_poly): return 0.0
                min_dist = 999.0
                for h in planner_holes_geom:
                    if point_in_polygon(pt, h): return 0.0
                    for i in range(len(h)):
                        A = h[i]; B = h[(i+1)%len(h)]
                        d = point_to_segment_dist(pt[0], pt[1], A[0], A[1], B[0], B[1])
                        if d < min_dist: min_dist = d
                return min_dist

            waypoints = get_topological_path(start_state[:2], target_pos, bounds, get_clearance_for_astar, grid_res=2.0)
            if len(waypoints) >= 2:
                goal_yaw = math.atan2(waypoints[-1][1] - waypoints[-2][1], waypoints[-1][0] - waypoints[-2][0])

            print(f"📍 A*: Xương sống chặng {current_stage_idx + 1} ({len(waypoints)} điểm) -> {algo_mode}")
            
            if algo_mode == "RRT_Original":
                return KinematicRRT(start_state, np.array(target_pos), goal_yaw, outer_poly, planner_holes_geom, bounds, None)
            else:
                return KinematicMCPP(start_state, np.array(target_pos), goal_yaw, outer_poly, planner_holes_geom, bounds, waypoints, None)
        else:
            if algo_mode == "RRT*": return RRTStar(start_state, target_pos, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "BI-RRT*": return BiRRTStar(start_state, target_pos, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "APF-RRT*": return APFRRTStar(start_state, target_pos, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "BI-APF-RRT*": return BiAPFRRTStar(start_state, target_pos, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "MS-BI-RRT*": return MSBiRRTStar(start_state, target_pos, goal_yaw, outer_poly, planner_holes_geom, bounds)

    running = True
    while running:
        clock.tick(FPS)
        dt_frame = 1.0 / FPS
        
        bounds = [0, logic_w, 0, logic_h]
        safe_car_radius = math.hypot(CAR_L/2 + 1.0, CAR_WIDTH/2)
        
        is_stage_finished = False
        path_planning_failed = False
        
        if click_step == 2 and not is_planning and not is_finished_all:
            if flat_planned_path and path_index >= len(flat_planned_path):
                current_target = ordered_goals[current_stage_idx]
                dist_to_target = math.hypot(current_state[0] - current_target[0], current_state[1] - current_target[1])
                if dist_to_target < 60: is_stage_finished = True
                else: path_planning_failed = True

        if click_step >= 1 and not is_stage_finished and not is_crashed:
            hit_now, _ = check_collision_with_index(current_state[0], current_state[1], current_state[2], outer_poly, real_holes, dyn_obstacles, t_lookahead=0.0)
            if hit_now:
                is_crashed = True
                print("💥 CRASH DETECTED!")

        # Xác định khi nào "thời gian" trong game tiếp tục chảy
        time_is_flowing = (click_step == 2 and not is_planning)

        # ========================================================
        # CẬP NHẬT VẬT CẢN ĐỘNG (Despawn & Respawn Sinh viên)
        # Sinh viên CHỈ di chuyển khi thuật toán đã tính toán xong
        # ========================================================
        if time_is_flowing:
            for obs in dyn_obstacles[:]: 
                old_vx, old_vy = obs.vx, obs.vy
                
                active_robot_state = current_state if not is_stage_finished else None
                active_goal_pos = ordered_goals[current_stage_idx] if current_stage_idx < len(ordered_goals) else None
                
                obs.move(dt_frame, bounds, outer_poly, real_holes, active_robot_state, safe_car_radius, active_goal_pos, GOAL_RADIUS)
                
                # KIỂM TRA ĐIỀU KIỆN BIẾN MẤT (DESPAWN)
                is_out_of_zone = obs.y < 24.0 or obs.y > 52.0
                is_bounced = (obs.vx * old_vx < 0) or (obs.vy * old_vy < 0)
                
                if is_out_of_zone or is_bounced:
                    dyn_obstacles.remove(obs)
                    # Truyền current_state vào để né robot khi đang chạy trên đường
                    dyn_obstacles.append(spawn_new_student(current_state))
            
        visible_dyn_obs = []
        if click_step >= 1 and not is_stage_finished and not is_crashed:
            rx, ry = current_state[0], current_state[1]
            for obs in dyn_obstacles:
                if math.hypot(obs.x - rx, obs.y - ry) <= (SENSOR_RADIUS + obs.radius):
                    visible_dyn_obs.append(obs)

        if click_step >= 1 and not is_stage_finished and not is_crashed:
            rx, ry = current_state[0], current_state[1]
            for i, h in enumerate(real_holes):
                if i not in known_hole_indices:
                    for j in range(len(h)):
                        A = h[j]; B = h[(j+1)%len(h)]
                        d = point_to_segment_dist(rx, ry, A[0], A[1], B[0], B[1])
                        if d <= SENSOR_RADIUS:
                            known_hole_indices.add(i)
                            planner_holes_geom.append(h)
                            need_path_check = True
                            break

        if click_step >= 2 and not is_planning and not is_crashed and not is_finished_all:
            if flat_planned_path and path_index < len(flat_planned_path) and need_path_check:
                path_is_blocked = False
                for i in range(path_index, len(flat_planned_path)):
                    pt = flat_planned_path[i]
                    yaw = pt[2] if len(pt) > 2 else current_state[2]
                    hit, _ = check_collision_with_index(pt[0], pt[1], yaw, outer_poly, planner_holes_geom, None)
                    if hit:
                        path_is_blocked = True
                        break
                
                if path_is_blocked:
                    if sim_report:
                        sim_report.add_obstacle()
                        sim_report.add_replan()
                    
                    is_planning = True
                    planned_path.clear()
                    flat_planned_path.clear()
                    path_index = 0
                    planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])
                    
                need_path_check = False 

        emergency_override = False
                
        # ========================================================
        # EVENT CHUỘT / BÀN PHÍM
        # ========================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    if map_images: current_map_idx=(current_map_idx+1)%len(map_images); reset_sim(True)
                elif event.key == pygame.K_r: reset_sim(False)
                elif event.key == pygame.K_TAB: 
                    algo_idx = (algo_idx + 1) % len(ALGO_MODES)
                    algo_mode = ALGO_MODES[algo_idx]
                    print(f"🔄 Chuyển sang thuật toán: {algo_mode}")
                    reset_sim(False)
                elif event.key == pygame.K_RETURN and click_step == 1 and len(multi_goals) > 0:
                    click_step = 2; 
                    start_coord = (current_state[0], current_state[1])
                    ordered_goals.clear()
                    ordered_goals.extend(amugopia_sabo_ordering(start_coord, multi_goals))
                    current_stage_idx = 0
                    is_planning = True
                    planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not is_crashed:
                    wx, wy = from_scr(event.pos[0], event.pos[1])
                    if click_step == 0:
                        current_state = (wx, wy, 0.0); click_step = 1
                    elif click_step == 1:
                        multi_goals.append((wx, wy))

        if not emergency_override and not is_crashed:
            if is_planning and click_step == 2:
                iters = 100
                path_segments = planner.plan_step(iterations=iters) if hasattr(planner, 'plan_step') and 'iterations' in planner.plan_step.__code__.co_varnames else planner.plan_step()
                
                if path_segments:
                    planned_path.clear()
                    planned_path.extend(path_segments)
                    
                    flat_planned_path.clear()
                    for seg in path_segments: 
                        is_dubins = seg.get('is_dubins', False)
                        direction = seg.get('direction', 1)
                        for p in seg['points']:
                            flat_planned_path.append((p[0], p[1], p[2], direction, is_dubins))
                            
                    is_planning = False
                    path_index = 0
                    print(f"✅ ĐÃ TÌM ĐƯỢC ĐƯỜNG MỚI CHO CHẶNG {current_stage_idx + 1}!")
                    
            elif not is_planning and flat_planned_path and path_index < len(flat_planned_path) and click_step == 2:
                dynamic_obstacle_incoming = False
                look_limit = min(path_index + LOOKAHEAD_STEPS * 2, len(flat_planned_path)) 
                
                for i in range(path_index, look_limit):
                    fs = flat_planned_path[i]
                    collided, hit_idx = check_collision_with_index(fs[0], fs[1], fs[2], outer_poly, real_holes, visible_dyn_obs, t_lookahead=1.5)
                    if collided and hit_idx == -3:
                        dynamic_obstacle_incoming = True
                        break 
                
                if not dynamic_obstacle_incoming:
                    fs = flat_planned_path[path_index]
                    current_state = (fs[0], fs[1], fs[2])
                    
                    if sim_report:
                        sim_report.add_waypoint(fs[0], fs[1], fs[2]) 
                        
                    path_history.append((current_state[0], current_state[1]))
                    path_index += 1

        if is_stage_finished and not is_finished_all:
            if current_stage_idx < len(ordered_goals) - 1:
                current_stage_idx += 1
                current_state = (flat_planned_path[-1][0], flat_planned_path[-1][1], flat_planned_path[-1][2])
                planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])
                planned_path.clear(); flat_planned_path.clear(); path_index = 0
                is_planning = True
            else:
                is_finished_all = True
                
                if sim_report:
                    current_map_name = os.path.basename(map_images[current_map_idx]) if map_images else "Dummy_Map"
                    safe_algo_name = algo_mode.replace("*", "Star")
                    sim_report.export_report(f"Results/Report_{safe_algo_name}_{current_map_name}.txt") 
                    sim_report.export_csv(map_name=f"{safe_algo_name}_{current_map_name}", filename="Results/Bao_Cao_Nang_Luong_Tong_Hop.csv")
                
                if map_images and current_map_idx < len(map_images) - 1:
                    pygame.display.flip()
                    pygame.time.delay(1500)
                    current_map_idx += 1
                    reset_sim(new_map=True)
                    
        elif path_planning_failed and not is_crashed:
            rev_dist = VELOCITY_MAX * dt_frame * 3.5
            nx = current_state[0] - rev_dist * math.cos(current_state[2])
            ny = current_state[1] - rev_dist * math.sin(current_state[2])
            
            w_hit, _ = check_collision_with_index(nx, ny, current_state[2], outer_poly, real_holes, None)
            if not w_hit:
                current_state = (nx, ny, current_state[2]) 
            
            is_planning = True
            planned_path.clear(); flat_planned_path.clear(); path_index = 0
            planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])

        # --- GIAO DIỆN UI ---
        ui_status = ""
        ui_color = BLUE
        if is_crashed: ui_status = "CRASHED! PRESS 'R' TO RESTART"; ui_color = BLACK
        elif emergency_override: ui_status = "DANGER! REVERSING!"; ui_color = RED
        elif click_step == 0: ui_status = "CLICK START POS"
        elif click_step == 1: ui_status = "CLICK GOALS -> PRESS ENTER"
        else:
            if is_finished_all: ui_status = "ALL MISSIONS COMPLETE"; ui_color = GREEN
            elif path_planning_failed: ui_status = "PLAN FAILED! REVERSING..."; ui_color = (255, 140, 0)
            elif is_planning: ui_status = f"PLANNING STAGE {current_stage_idx+1}/{len(ordered_goals)}..."; ui_color = RED
            elif not is_planning and flat_planned_path and path_index < len(flat_planned_path):
                if 'dynamic_obstacle_incoming' in locals() and dynamic_obstacle_incoming:
                    ui_status = "BRAKING..."; ui_color = (255, 140, 0)
                else: ui_status = f"MOVING TO GOAL {current_stage_idx+1}"; ui_color = GREEN

        screen.fill(WHITE)
        
        if scaled_bg is not None:
            screen.blit(scaled_bg, (top_left_x, top_left_y))
        else:
            if outer_poly: 
                pygame.draw.polygon(screen, BLACK, [to_scr(p) for p in outer_poly], 2)
        
        for i, h in enumerate(real_holes):
            if i in known_hole_indices:
                pygame.draw.polygon(screen, RED, [to_scr(p) for p in h], 2)

        if click_step >= 1 and not is_finished_all and not is_crashed:
            pygame.draw.circle(screen, (200, 230, 255), to_scr(current_state[:2]), int(SENSOR_RADIUS * scale), 1)

        for obs in dyn_obstacles:
            if click_step >= 1 and obs in visible_dyn_obs:
                pygame.draw.circle(screen, (255, 140, 0), to_scr((obs.x, obs.y)), int(obs.radius * scale))
                fut_x = obs.x + obs.vx * 1.5; fut_y = obs.y + obs.vy * 1.5
                pygame.draw.line(screen, (255, 200, 100), to_scr((obs.x, obs.y)), to_scr((fut_x, fut_y)), 2)
            else:
                pygame.draw.circle(screen, (200, 200, 200), to_scr((obs.x, obs.y)), int(obs.radius * scale))

        if click_step >= 2 and planner and hasattr(planner, 'waypoints'):
            wp_scr = [to_scr(wp[:2]) for wp in planner.waypoints]
            if len(wp_scr) > 1:
                pygame.draw.lines(screen, (255, 165, 0), False, wp_scr, 2)
                for i, pt in enumerate(wp_scr):
                    current_idx = getattr(planner, 'current_wp_idx', -1)
                    if i == current_idx:
                        pygame.draw.circle(screen, (255, 0, 0), pt, 7)
                        pygame.draw.circle(screen, (255, 255, 255), pt, 3)
                    else:
                        pygame.draw.circle(screen, (255, 140, 0), pt, 4)

        if is_planning and click_step == 2 and not is_crashed and planner:
            trees_to_draw = []
            if hasattr(planner, 'node_list'):
                trees_to_draw.append((planner.node_list, (200, 200, 255)))
            if hasattr(planner, 'node_list_start'):
                trees_to_draw.append((planner.node_list_start, (170, 210, 255)))
            if hasattr(planner, 'node_list_goal'):
                trees_to_draw.append((planner.node_list_goal, (255, 180, 210)))
            
            for tree, col in trees_to_draw:
                for node in tree:
                    parent = getattr(node, 'parent', None) or getattr(node, 'parent_node', None)
                    if parent:
                        if hasattr(node, 'path_x') and len(node.path_x) > 1:
                            pts = [to_scr((px, py)) for px, py in zip(node.path_x, node.path_y)]
                            pygame.draw.lines(screen, col, False, pts, 1)
                        else:
                            nx, ny = getattr(node, 'state', (node.x, node.y))[:2] if hasattr(node, 'state') else (node.x, node.y)
                            px, py = getattr(parent, 'state', (parent.x, parent.y))[:2] if hasattr(parent, 'state') else (parent.x, parent.y)
                            pygame.draw.line(screen, col, to_scr((px, py)), to_scr((nx, ny)), 1)

        if not is_planning and planned_path and click_step == 2:
            for seg in planned_path:
                points = seg['points']
                if len(points) > 1:
                    pts_scr = [to_scr((p[0], p[1])) for p in points]
                    if seg.get('is_dubins', False): col = DUBINS_COLOR; w = 4
                    elif seg.get('direction', 1) == -1: col = REVERSE_COLOR; w = 2
                    else: col = GREEN; w = 2
                    pygame.draw.lines(screen, col, False, pts_scr, w)

        if len(path_history) > 1:
            pygame.draw.lines(screen, BLACK, False, [to_scr(p) for p in path_history], 1)

        if click_step >= 1:
            for i, goal in enumerate(multi_goals):
                g_scr = to_scr(goal)
                try:
                    order_in_route = ordered_goals.index(goal)
                    is_visited = order_in_route < current_stage_idx
                except ValueError:
                    is_visited = False
                
                color = (150, 150, 150) if is_visited else BLUE
                pygame.draw.circle(screen, color, g_scr, int(GOAL_RADIUS * scale))
                text = font.render(str(i+1), True, (255, 255, 255))
                screen.blit(text, (g_scr[0]-5, g_scr[1]-8))

        if click_step >= 1: 
            draw_car(current_state, BLACK if is_crashed else CAR_COLOR)

        screen.blit(font.render(f"Mode: {algo_mode} | [TAB] Switch | [N] Next Map | [R] Reset", True, BLUE), (10, 10))
        screen.blit(font.render(f"Status: {ui_status}", True, ui_color), (10, 30))
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()