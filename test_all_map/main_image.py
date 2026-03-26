import pygame
import sys
import cv2
import numpy as np
import math
import random

from config import *
from core_math import get_car_corners, point_in_polygon
from planners import KinematicRRT, KinematicMCPP
from environment import check_collision_with_index

MATH_WIDTH_TARGET = 100.0 # ÉP CHIỀU DÀI BẢN ĐỒ VỀ ĐÚNG 100 ĐƠN VỊ

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
    
    # TỰ ĐỘNG TÍNH SCALE: Sao cho w_px * math_scale = 100.0
    math_scale = MATH_WIDTH_TARGET / float(w_px)
    
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # THÊM: Làm mượt vạch kẻ để tránh răng cưa gây va chạm ảo
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    real_holes_math = []
    for cnt in contours:
        # TĂNG NGƯỠNG: Chỉ lấy những vạch kẻ thực sự (diện tích > 100 pixel)
        if cv2.contourArea(cnt) > 100: 
            # Làm mịn đa giác để giảm số cạnh (giúp check va chạm nhanh hơn)
            epsilon = 0.01 * cv2.arcLength(cnt, True) 
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            poly_m = []
            for pt in approx:
                mx = float(pt[0][0]) * math_scale
                my = float(h_px - pt[0][1]) * math_scale # Giữ nguyên logic lật Y của bạn
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
    step = 0.1 # Bước dò 0.1 đơn vị (Tránh đi xuyên tường)
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
    
    IMAGE_FILE = "parking_lot_1.jpg"
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
                            # ĐỒNG BỘ: Đã thay thế {} bằng bounds để MCPP nhận đúng kích thước map
                            planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, real_holes, bounds, None)

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