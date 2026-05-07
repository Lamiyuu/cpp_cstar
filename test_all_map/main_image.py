import pygame
import sys
import cv2
import numpy as np
import math
import random

from config import *
from core_math import get_car_corners, point_in_polygon, point_to_segment_dist
from planners import KinematicRRT, KinematicMCPP, get_topological_path
from environment import check_collision_with_index

MATH_WIDTH_TARGET = 100.0 

# ==========================================
# 1. HÀM ĐỌC ẢNH VÀ TRÍCH XUẤT ĐA GIÁC 
# ==========================================
def load_and_scale_image(image_path, max_width=1000):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ LỖI: Không đọc được ảnh '{image_path}'.")
        sys.exit()
        
    if img.shape[1] > max_width:
        scale_img = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * scale_img)))
        
    h_px, w_px = img.shape
    math_scale = MATH_WIDTH_TARGET / float(w_px)
    
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    kernel_dilate = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    kernel_close = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    cv2.rectangle(thresh, (0, 0), (w_px - 1, h_px - 1), 0, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    real_holes_math = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        if area > 30 or length > 50: 
            epsilon = 0.005 * length 
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            poly_m = []
            for pt in approx:
                mx = float(pt[0][0]) * math_scale
                my = float(h_px - pt[0][1]) * math_scale 
                poly_m.append((mx, my))
            if len(poly_m) >= 3:
                real_holes_math.append(poly_m)
                
    w_m = w_px * math_scale
    h_m = h_px * math_scale
    outer_poly_math = [(0.0, 0.0), (w_m, 0.0), (w_m, h_m), (0.0, h_m)]
    
    bg_img = cv2.imread(image_path)
    if bg_img.shape[1] > max_width:
        bg_img = cv2.resize(bg_img, (w_px, h_px))
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB) 
    bg_surface = pygame.image.frombuffer(bg_img.flatten(), (w_px, h_px), 'RGB')
    return outer_poly_math, real_holes_math, w_px, h_px, bg_surface, math_scale

# ==========================================
# 2. THUẬT TOÁN AMuGOPIA - SẮP XẾP ĐA MỤC TIÊU
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
        
    # Không ép quay về vạch xuất phát nữa (Open-loop)
    return ordered_path

def cast_ray(x, y, yaw, outer_poly, holes, max_dist=20.0):
    step = 0.1 
    dist = 0.0
    while dist < max_dist:
        nx = x + math.cos(yaw) * dist
        ny = y + math.sin(yaw) * dist
        hit = False
        if outer_poly and not point_in_polygon((nx, ny), outer_poly): hit = True
        for h in holes:
            if point_in_polygon((nx, ny), h): hit = True; break
        if hit: return dist
        dist += step
    return max_dist

def get_best_yaw_for_parking(wx, wy, outer_poly, holes):
    angles = [0.0, math.pi/2, math.pi, 3*math.pi/2] 
    dists = [cast_ray(wx, wy, a, outer_poly, holes) for a in angles]
    return angles[dists.index(max(dists))] 

# ==========================================
# 3. VÒNG LẶP CHÍNH 
# ==========================================
def main():
    pygame.init()
    IMAGE_FILE = "parking_lot_2.jpg"
    print(f"Khởi động Hệ thống Điều hướng Đa Chặng (Stage-by-Stage MCPP)...")
    
    outer_poly, real_holes, w_px, h_px, bg_surface, math_scale = load_and_scale_image(IMAGE_FILE)
    bounds = [0, w_px * math_scale, 0, h_px * math_scale]

    screen = pygame.display.set_mode((w_px, h_px))
    pygame.display.set_caption("Stage-by-Stage MCPP")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    def to_pygame(math_x, math_y):
        return int(math_x / math_scale), int(h_px - (math_y / math_scale))

    def from_pygame(scr_x, scr_y):
        return float(scr_x) * math_scale, float(h_px - scr_y) * math_scale

    def draw_car(state, color=CAR_COLOR):
        corners = get_car_corners(state[0], state[1], state[2])
        scr_corners = [to_pygame(p[0], p[1]) for p in corners]
        pygame.draw.polygon(screen, color, scr_corners)
        pygame.draw.polygon(screen, BLACK, scr_corners, 1)
        fx = state[0] + CAR_L * math.cos(state[2])
        fy = state[1] + CAR_L * math.sin(state[2])
        pygame.draw.line(screen, BLACK, to_pygame(state[0], state[1]), to_pygame(fx, fy), 2)

    # Bộ Cache cho A*
    all_segments = []
    for h in real_holes:
        for i in range(len(h)):
            all_segments.append((h[i], h[(i+1)%len(h)]))

    def get_clearance_fast(pt):
        if outer_poly and not point_in_polygon(pt, outer_poly): return 0.0
        min_dist = 999.0
        for A, B in all_segments:
            if abs(pt[0] - A[0]) > 10.0 and abs(pt[0] - B[0]) > 10.0 and abs(pt[1] - A[1]) > 10.0 and abs(pt[1] - B[1]) > 10.0: continue
            d = point_to_segment_dist(pt[0], pt[1], A[0], A[1], B[0], B[1])
            if d < min_dist: min_dist = d
        if min_dist < 2.0:
            for h in real_holes:
                if point_in_polygon(pt, h): return 0.0
        return min_dist

    # ========================================================
    # HÀM LẬP KẾ HOẠCH CHO 1 CHẶNG ĐƠN LẺ (SINGLE STAGE)
    # ========================================================
    def plan_next_stage(start_state, target_pos):
        goal_yaw = get_best_yaw_for_parking(target_pos[0], target_pos[1], outer_poly, real_holes)
        
        # Thay vì cắm đầu theo hướng xe đang đỗ (start_state[2])
        open_yaw = get_best_yaw_for_parking(start_state[0], start_state[1], outer_poly, real_holes)
        escape_x = start_state[0] + math.cos(open_yaw) * 8.0
        escape_y = start_state[1] + math.sin(open_yaw) * 8.0
        escape_pt = (escape_x, escape_y)

        out_x, out_y = math.cos(goal_yaw), math.sin(goal_yaw)
        perp_x1, perp_y1 = -out_y, out_x
        perp_x2, perp_y2 = out_y, -out_x

        dx = escape_pt[0] - target_pos[0]
        dy = escape_pt[1] - target_pos[1]
        if (dx * perp_x1 + dy * perp_y1) > 0: perp_x, perp_y = perp_x1, perp_y1
        else: perp_x, perp_y = perp_x2, perp_y2

        # Lùi chuồng (Dubins)
        front_x = target_pos[0] + out_x * 8.0
        front_y = target_pos[1] + out_y * 8.0
        front_pt = (front_x, front_y)

        pull_x = front_x + perp_x * 12.0
        pull_y = front_y + perp_y * 12.0
        pull_ahead_pt = (pull_x, pull_y)

        curve_x = front_x + perp_x * 6.0 + out_x * 2.0
        curve_y = front_y + perp_y * 6.0 + out_y * 2.0
        curve_pt = (curve_x, curve_y)

        waypoints = get_topological_path(escape_pt, pull_ahead_pt, bounds, get_clearance_fast, grid_res=2.0)
        
        waypoints.insert(0, escape_pt)
        waypoints.append(curve_pt)     
        waypoints.append(front_pt)     
        waypoints.append(target_pos)
        
        print(f"📍 A*: Xong xương sống chặng {current_stage_idx + 1} ({len(waypoints)} points). Đang chạy MCPP...")
        return KinematicMCPP(start_state, np.array(target_pos), goal_yaw, outer_poly, real_holes, bounds, waypoints, None)

    click_step = 0
    is_planning = False
    is_crashed = False
    is_finished_all = False
    
    current_state = (0, 0, 0)
    planner = None
    planned_path = []; flat_planned_path = []; path_index = 0
    
    multi_goals = [] 
    ordered_goals = []
    current_stage_idx = 0
    
    def reset_sim():
        nonlocal click_step, is_planning, is_crashed, is_finished_all, planner, planned_path, flat_planned_path, path_index, multi_goals, ordered_goals, current_stage_idx
        click_step = 0; is_planning = False; is_crashed = False; is_finished_all = False
        planner = None; planned_path = []; flat_planned_path = []; path_index = 0
        multi_goals = []; ordered_goals = []; current_stage_idx = 0

    running = True
    while running:
        clock.tick(FPS)
        
        # Nhận diện khi xe CHẠY XONG MỘT CHẶNG (Stage Completed)
        is_stage_finished = (click_step == 2 and not is_planning and flat_planned_path and path_index >= len(flat_planned_path))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: reset_sim()
                
                # BẤM ENTER -> CHỐT TẤT CẢ VÀ BẮT ĐẦU CHẶNG ĐẦU TIÊN
                elif event.key == pygame.K_RETURN and click_step == 1 and len(multi_goals) > 0:
                    click_step = 2; 
                    print(f"🚀 TÍNH TOÁN TASK PLANNER (AMuGOPIA)...")
                    start_coord = (current_state[0], current_state[1])
                    ordered_goals = amugopia_sabo_ordering(start_coord, multi_goals)
                    
                    current_stage_idx = 0
                    is_planning = True
                    print(f"📋 Nhiệm vụ tổng: Phải qua {len(ordered_goals)} điểm giao hàng.")
                    
                    # Bắt đầu lên lộ trình cho Chặng 1
                    planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])
                        
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not is_crashed:
                    wx, wy = from_pygame(event.pos[0], event.pos[1])
                    if click_step == 0:
                        start_yaw = get_best_yaw_for_parking(wx, wy, outer_poly, real_holes)
                        current_state = (wx, wy, start_yaw)
                        click_step = 1
                        print("📍 Đã đặt Start. Thả Đa mục tiêu và nhấn ENTER để xe chạy từng chặng.")
                    elif click_step == 1:
                        multi_goals.append((wx, wy))
                        print(f"🚩 Đã thả Goal số {len(multi_goals)}.")

        if click_step >= 1 and not is_stage_finished and not is_crashed:
            hit_now, _ = check_collision_with_index(current_state[0], current_state[1], current_state[2], outer_poly, real_holes, None, t_lookahead=0.0)
            if hit_now:
                is_crashed = True
                print("💥 TAI NẠN!")

        # MCPP CHẠY QUY HOẠCH CHO CHẶNG HIỆN TẠI
        if is_planning and click_step == 2 and not is_crashed:
            path_segments = planner.plan_step(iterations=50) 
            if path_segments:
                planned_path = path_segments
                flat_planned_path = []
                for seg in path_segments: flat_planned_path.extend(seg['points'])
                is_planning = False
                path_index = 0
                print(f"✅ ĐÃ TÌM ĐƯỢC LỘ TRÌNH CHO CHẶNG {current_stage_idx + 1}. Xe bắt đầu di chuyển...")

        # XE LĂN BÁNH TRÊN CHẶNG HIỆN TẠI
        if not is_planning and flat_planned_path and path_index < len(flat_planned_path) and click_step == 2 and not is_crashed:
            current_state = flat_planned_path[path_index]
            path_index += 1

        # LOGIC CHUYỂN CHẶNG (TRÁI TIM CỦA HỆ THỐNG MỚI)
        if is_stage_finished and not is_finished_all:
            if current_stage_idx < len(ordered_goals) - 1:
                print(f"🏁 ĐÃ ĐẾN ĐÍCH {current_stage_idx + 1}! Chuẩn bị đi chặng {current_stage_idx + 2}...")
                current_stage_idx += 1
                current_state = flat_planned_path[-1] # Lấy chính xác tọa độ/góc vừa đỗ xong
                
                # Gọi lại hàm quy hoạch từ vị trí đỗ hiện tại tới Goal tiếp theo
                planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])
                planned_path = []
                flat_planned_path = []
                path_index = 0
                is_planning = True
            else:
                print("🏆 ĐÃ GIAO HÀNG XONG TOÀN BỘ!")
                is_finished_all = True

        # VẼ ĐỒ HỌA
        screen.blit(bg_surface, (0, 0)) 
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_pygame(p[0], p[1]) for p in outer_poly], 1)

        # Chỉ vẽ đường A* Cam của chặng hiện tại
        if click_step >= 2 and planner and hasattr(planner, 'waypoints'):
            wp_scr = [to_pygame(wp[0], wp[1]) for wp in planner.waypoints]
            if len(wp_scr) > 1:
                pygame.draw.lines(screen, (255, 165, 0), False, wp_scr, 2)
                for i, pt in enumerate(wp_scr):
                    current_idx = getattr(planner, 'current_wp_idx', -1)
                    if i == current_idx:
                        pygame.draw.circle(screen, (255, 0, 0), pt, 7)
                        pygame.draw.circle(screen, (255, 255, 255), pt, 3)
                    else:
                        pygame.draw.circle(screen, (255, 140, 0), pt, 4)

        if not is_planning and planned_path and click_step == 2:
            for seg in planned_path:
                points = seg['points']
                if len(points) > 1:
                    pts_scr = [to_pygame(p[0], p[1]) for p in points]
                    col = DUBINS_COLOR if seg['is_dubins'] else (REVERSE_COLOR if seg['direction'] == -1 else GREEN)
                    pygame.draw.lines(screen, col, False, pts_scr, 2 if not seg['is_dubins'] else 4)

        # Vẽ Đa mục tiêu (Xám mờ nếu đã qua, Xanh nếu chưa tới)
        if click_step >= 1:
            for i, goal in enumerate(multi_goals):
                g_scr = to_pygame(goal[0], goal[1])
                # Trích xuất xem mục tiêu này đang ở thứ tự bao nhiêu trong danh sách ordered_goals
                try:
                    order_in_route = ordered_goals.index(goal)
                    is_visited = order_in_route < current_stage_idx
                except ValueError:
                    is_visited = False
                
                color = (150, 150, 150) if is_visited else BLUE
                pygame.draw.circle(screen, color, g_scr, int(GOAL_RADIUS / math_scale))
                text = font.render(str(i+1), True, (255, 255, 255))
                screen.blit(text, (g_scr[0]-5, g_scr[1]-8))
            
        if click_step >= 1: 
            draw_car(current_state, BLACK if is_crashed else CAR_COLOR)

        ui_status = "IDLE"
        ui_color = BLUE
        if is_crashed: ui_status = "CRASHED! PRESS 'R'"; ui_color = BLACK
        elif click_step == 0: ui_status = "CLICK START"
        elif click_step == 1: ui_status = "CLICK GOALS -> ENTER"
        else:
            if is_finished_all: ui_status = "ALL MISSIONS COMPLETE"; ui_color = GREEN
            elif is_planning: ui_status = f"PLANNING STAGE {current_stage_idx+1}/{len(ordered_goals)}..."; ui_color = RED
            elif path_index < len(flat_planned_path): ui_status = f"MOVING TO GOAL {current_stage_idx+1}"; ui_color = GREEN

        screen.blit(font.render(f"Mode: Stage-by-Stage MCPP | [R] Reset", True, BLUE), (10, 10))
        screen.blit(font.render(f"Status: {ui_status}", True, ui_color), (10, 30))
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()