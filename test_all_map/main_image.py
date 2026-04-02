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

MATH_WIDTH_TARGET = 100.0 # ÉP CHIỀU DÀI BẢN ĐỒ VỀ ĐÚNG 100 ĐƠN VỊ

# ==========================================
# 1. HÀM ĐỌC ẢNH VÀ TRÍCH XUẤT ĐA GIÁC (BẢO TOÀN VẠCH KẺ MẢNH)
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
    
    # [SỬA LỖI 1]: LÀM DÀY VẠCH KẺ MẢNH
    # Dùng hàm Dilate để "bơm" các vạch đỗ xe mỏng manh lên, tránh bị đứt gãy
    kernel_dilate = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    # [SỬA LỖI 2]: TRÁM KHỐI RỖ (CARO)
    # Vẫn giữ hàm Close để nối liền các khối đen ở góc bản đồ
    kernel_close = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # XÓA BỎ LỆNH MORPH_OPEN Ở BẢN CŨ ĐỂ KHÔNG LÀM BỐC HƠI VẠCH KẺ
    
    # Đóng gói viền để nhận diện khối đen tràn ảnh
    cv2.rectangle(thresh, (0, 0), (w_px - 1, h_px - 1), 0, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    real_holes_math = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        
        # [SỬA LỖI 3]: CỨU VẠCH KẺ DỰA VÀO CHIỀU DÀI
        # Nếu diện tích nhỏ nhưng chiều dài > 50 pixel thì chắc chắn nó là vạch đỗ xe!
        if area > 30 or length > 50: 
            # Giảm độ làm tròn từ 0.01 xuống 0.005 để các vạch kẻ giữ được độ thẳng góc
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
# 2. CẢM BIẾN TÌM HƯỚNG CỬA CHUỒNG 
# ==========================================
def cast_ray(x, y, yaw, outer_poly, holes, max_dist=20.0):
    step = 0.1 # Bước dò 0.1 đơn vị
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
# 3. VÒNG LẶP CHÍNH CỦA MÔ PHỎNG
# ==========================================
def main():
    pygame.init()
    
    IMAGE_FILE = "parking_lot_2.jpg"
    print(f"Đang thiết lập sa bàn quy chuẩn 100 từ ảnh '{IMAGE_FILE}' ...")
    
    outer_poly, real_holes, w_px, h_px, bg_surface, math_scale = load_and_scale_image(IMAGE_FILE)
    bounds = [0, w_px * math_scale, 0, h_px * math_scale]

    screen = pygame.display.set_mode((w_px, h_px))
    pygame.display.set_caption(f"Kinematic Car - Base 100 Map: {IMAGE_FILE}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    def to_pygame(math_x, math_y):
        scr_x = int(math_x / math_scale)
        scr_y = int(h_px - (math_y / math_scale))
        return scr_x, scr_y

    def from_pygame(scr_x, scr_y):
        math_x = float(scr_x) * math_scale
        math_y = float(h_px - scr_y) * math_scale
        return math_x, math_y

    def draw_car(state, color=CAR_COLOR):
        corners = get_car_corners(state[0], state[1], state[2])
        scr_corners = [to_pygame(p[0], p[1]) for p in corners]
        pygame.draw.polygon(screen, color, scr_corners)
        pygame.draw.polygon(screen, BLACK, scr_corners, 1)
        fx = state[0] + CAR_L * math.cos(state[2])
        fy = state[1] + CAR_L * math.sin(state[2])
        pygame.draw.line(screen, BLACK, to_pygame(state[0], state[1]), to_pygame(fx, fy), 2)

    algo_mode = "RRT" 
    click_step = 0
    is_planning = False
    is_crashed = False
    current_state = (0, 0, 0)
    goal_pos = np.array([0, 0]); goal_yaw = 0.0
    planner = None
    planned_path = []; flat_planned_path = []; path_index = 0

    def reset_sim():
        nonlocal click_step, is_planning, is_crashed, planner, planned_path, flat_planned_path, path_index
        click_step = 0; is_planning = False; is_crashed = False
        planner = None; planned_path = []; flat_planned_path = []; path_index = 0

    running = True
    while running:
        clock.tick(FPS)
        is_finished = (click_step == 2 and not is_planning and flat_planned_path and path_index >= len(flat_planned_path))

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: reset_sim()
                elif event.key == pygame.K_TAB:
                    algo_mode = "MCPP" if algo_mode == "RRT" else "RRT"
                    reset_sim()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if click_step < 2 and not is_crashed:
                    wx, wy = from_pygame(event.pos[0], event.pos[1])
                    
                    if click_step == 0:
                        start_yaw = get_best_yaw_for_parking(wx, wy, outer_poly, real_holes)
                        current_state = (wx, wy, start_yaw)
                        click_step = 1
                        
                    elif click_step == 1:
                        goal_yaw = get_best_yaw_for_parking(wx, wy, outer_poly, real_holes)
                        goal_pos = np.array([wx, wy])
                        
                        click_step = 2; is_planning = True
                        print(f"🚀 Bắt đầu tìm đường bằng {algo_mode}...")
                        
                        if algo_mode == "RRT":
                            planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, real_holes, bounds, None)
                        else:
                            # TẦNG 1: Dùng hàm A* để lấy danh sách waypoints
                            def get_clearance(pt):
                                if outer_poly and not point_in_polygon(pt, outer_poly): 
                                    return 0.0
                                for h in real_holes:
                                    if point_in_polygon(pt, h): 
                                        return 0.0
                                
                                min_dist = 999.0
                                for h in real_holes:
                                    for i in range(len(h)):
                                        A = h[i]; B = h[(i+1)%len(h)]
                                        d = point_to_segment_dist(pt[0], pt[1], A[0], A[1], B[0], B[1])
                                        if d < min_dist:
                                            min_dist = d
                                return min_dist
                            
                            # ====================================================
                            # 1. ĐIỂM THOÁT CHUỒNG (Rút ngắn còn 8.0 cho an toàn)
                            # ====================================================
                            escape_dist = 8.0 
                            escape_x = current_state[0] + math.cos(current_state[2]) * escape_dist
                            escape_y = current_state[1] + math.sin(current_state[2]) * escape_dist
                            escape_pt = (escape_x, escape_y)

                            # ====================================================
                            # 2. XÁC ĐỊNH HƯỚNG HÀNH LANG TỐI ƯU
                            # ====================================================
                            out_x, out_y = math.cos(goal_yaw), math.sin(goal_yaw)
                            perp_x1, perp_y1 = -out_y, out_x
                            perp_x2, perp_y2 = out_y, -out_x

                            # Chọn hướng dọc hành lang hướng về phía xe đang đứng
                            dx = escape_pt[0] - goal_pos[0]
                            dy = escape_pt[1] - goal_pos[1]
                            if (dx * perp_x1 + dy * perp_y1) > 0:
                                perp_x, perp_y = perp_x1, perp_y1
                            else:
                                perp_x, perp_y = perp_x2, perp_y2

                            # ====================================================
                            # 3. QUỸ ĐẠO LÙI (Kéo các điểm nằm trọn giữa hành lang)
                            # ====================================================
                            # Điểm mặt chuồng (Cách cửa 8.0)
                            front_x = goal_pos[0] + out_x * 8.0
                            front_y = goal_pos[1] + out_y * 8.0
                            front_pt = (front_x, front_y)

                            # Điểm vòng lên (Từ mặt chuồng tiến dọc hành lang 12.0)
                            pull_x = front_x + perp_x * 12.0
                            pull_y = front_y + perp_y * 12.0
                            pull_ahead_pt = (pull_x, pull_y)

                            # Điểm bo cua lùi
                            curve_x = front_x + perp_x * 6.0 + out_x * 2.0
                            curve_y = front_y + perp_y * 6.0 + out_y * 2.0
                            curve_pt = (curve_x, curve_y)

                            # 4. CHẠY A* TỚI ĐIỂM VÒNG LÊN 
                            waypoints = get_topological_path(escape_pt, pull_ahead_pt, bounds, get_clearance, grid_res=2.0)
                            
                            # 5. LẮP RÁP LỘ TRÌNH
                            waypoints.insert(0, escape_pt)
                            waypoints.append(curve_pt)     
                            waypoints.append(front_pt)     
                            waypoints.append(goal_pos)
                            print(f"🗺️ Tầng 1: Đã tìm thấy xương sống dẫn đường gồm {len(waypoints)} điểm.")
                            
                            # TẦNG 2: MCPP có dẫn hướng (Truyền waypoints vào)
                            planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, real_holes, bounds, waypoints, None)

        if click_step >= 1 and not is_finished and not is_crashed:
            hit_now, _ = check_collision_with_index(current_state[0], current_state[1], current_state[2], outer_poly, real_holes, None, t_lookahead=0.0)
            if hit_now:
                is_crashed = True
                print("💥 TAI NẠN! Click nhầm vào tường.")

        if is_planning and click_step == 2 and not is_crashed:
            steps_per_frame = 100 if algo_mode == "RRT" else 1
            for _ in range(steps_per_frame):
                path_segments = planner.plan_step()
                if path_segments:
                    planned_path = path_segments
                    flat_planned_path = []
                    for seg in path_segments: flat_planned_path.extend(seg['points'])
                    is_planning = False
                    path_index = 0
                    print("✅ ĐÃ TÌM THẤY LỘ TRÌNH!")
                    break

        if not is_planning and flat_planned_path and path_index < len(flat_planned_path) and click_step == 2 and not is_crashed:
            current_state = flat_planned_path[path_index]
            path_index += 1

        # VẼ ĐỒ HỌA
        screen.blit(bg_surface, (0, 0)) 
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_pygame(p[0], p[1]) for p in outer_poly], 1)

        # ==========================================================
        # VẼ CÁC WAYPOINT TỪ TẦNG 1 (Xương sống dẫn đường)
        # ==========================================================
        if click_step >= 2 and algo_mode == "MCPP" and planner and hasattr(planner, 'waypoints'):
            wp_scr = [to_pygame(wp[0], wp[1]) for wp in planner.waypoints]
            if len(wp_scr) > 1:
                # Vẽ đường thẳng nối các Waypoint (Màu Cam)
                pygame.draw.lines(screen, (255, 165, 0), False, wp_scr, 2)
                
                # Vẽ từng điểm Waypoint
                for i, pt in enumerate(wp_scr):
                    # Lấy index của waypoint hiện tại mà xe đang nhắm tới
                    current_idx = getattr(planner, 'current_wp_idx', -1)
                    
                    if i == current_idx:
                        # Điểm Mồi Mục Tiêu Hiện Tại: Vẽ To và có viền Đỏ để dễ nhìn
                        pygame.draw.circle(screen, (255, 0, 0), pt, 7)
                        pygame.draw.circle(screen, (255, 255, 255), pt, 3)
                    else:
                        # Các Điểm Mồi Bình Thường: Vẽ nhỏ hơn, màu Cam
                        pygame.draw.circle(screen, (255, 140, 0), pt, 4)
        # ==========================================================

        if is_planning and click_step == 2 and not is_crashed:
            for node in planner.node_list:
                parent = getattr(node, 'parent', None) or getattr(node, 'parent_node', None)
                if parent:
                    pts = [to_pygame(px, py) for px, py in zip(node.path_x, node.path_y)]
                    if len(pts)>1: pygame.draw.lines(screen, (200, 200, 255), False, pts, 1)

        if not is_planning and planned_path and click_step == 2:
            for seg in planned_path:
                points = seg['points']
                if len(points) > 1:
                    pts_scr = [to_pygame(p[0], p[1]) for p in points]
                    col = DUBINS_COLOR if seg['is_dubins'] else (REVERSE_COLOR if seg['direction'] == -1 else GREEN)
                    w = 4 if seg['is_dubins'] else 2
                    pygame.draw.lines(screen, col, False, pts_scr, w)

        if click_step >= 2:
            g_scr = to_pygame(goal_pos[0], goal_pos[1])
            pygame.draw.circle(screen, BLUE, g_scr, int(GOAL_RADIUS / math_scale))
            arrow_end = (goal_pos[0] + 4.0*math.cos(goal_yaw), goal_pos[1] + 4.0*math.sin(goal_yaw))
            pygame.draw.line(screen, BLUE, g_scr, to_pygame(arrow_end[0], arrow_end[1]), 3)
            
        if click_step >= 1: 
            car_draw_color = BLACK if is_crashed else CAR_COLOR
            draw_car(current_state, car_draw_color)

        ui_status = ""
        ui_color = BLUE
        if is_crashed: ui_status = "CRASHED! PRESS 'R'"; ui_color = BLACK
        elif click_step == 0: ui_status = "CLICK CHON DIEM DAU"
        elif click_step == 1: ui_status = "CLICK CHON DIEM DICH"
        else:
            if not is_planning and flat_planned_path and path_index < len(flat_planned_path):
                ui_status = "MOVING"; ui_color = GREEN
            elif is_finished: ui_status = "FINISHED"; ui_color = GREEN
            elif is_planning: ui_status = "PLANNING..."; ui_color = RED

        screen.blit(font.render(f"Mode: {algo_mode} | [TAB] Switch | [R] Reset", True, BLUE), (10, 10))
        screen.blit(font.render(f"Status: {ui_status}", True, ui_color), (10, 30))
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()