import pygame
import sys
import os
import glob
import math
import numpy as np
import random

from config import *
from core_math import get_car_corners, dist
from environment import load_data, get_valid_random_pos, DynamicObstacle, check_collision_with_index
from planners import KinematicRRT, KinematicMCPP

def main():
    pygame.init()
    if not os.path.exists("Results"): os.makedirs("Results")
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Kinematic Planner + True Radar Sensor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC12_*")))
    
    current_map_idx = 0
    algo_mode = "RRT"
    outer_poly = []; real_holes = []
    known_hole_indices = set(); planner_holes_geom = []
    current_state = (0, 0, 0)
    goal_pos = np.array([0, 0]); goal_yaw = 0.0
    planner = None
    
    planned_path = []; flat_planned_path = []; path_index = 0; path_history = []
    global_grid_penalties = {}
    dyn_obstacles = [] 
    
    click_step = 0 
    is_planning = False
    is_crashed = False # TRẠNG THÁI MỚI: Kiểm tra xe có bị đâm không
    scale = 1.0

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_state, goal_pos, goal_yaw, known_hole_indices, planner_holes_geom
        nonlocal planner, planned_path, flat_planned_path, path_index, is_planning, scale, path_history, click_step
        nonlocal global_grid_penalties, dyn_obstacles, is_crashed

        use_dummy = False
        if map_folders:
            folder = map_folders[current_map_idx]
            if new_map: print(f"Loading: {folder}")
            outer_poly, real_holes = load_data(folder)
            if not outer_poly: use_dummy = True
        else: use_dummy = True
        
        if use_dummy:
             outer_poly = [(0,0), (WINDOW_SIZE,0), (WINDOW_SIZE,WINDOW_SIZE), (0,WINDOW_SIZE)]
             real_holes = [[(300,300), (500,300), (500,500), (300,500)]]

        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        mx = max(max(xs), max(ys))
        scale = (WINDOW_SIZE - 80) / mx
        
        click_step = 0; is_planning = False; planner = None; is_crashed = False
        current_state = (0.0, 0.0, 0.0)
        goal_pos = np.array([0.0, 0.0]); goal_yaw = 0.0
        
        if new_map: 
            known_hole_indices = set(); planner_holes_geom = []
            global_grid_penalties = {}; dyn_obstacles.clear()
            
            bounds = [0, mx, 0, mx]
            for _ in range(NUM_DYN_OBS):
                rp = get_valid_random_pos(outer_poly, real_holes, bounds)
                angle = random.uniform(0, 2*math.pi)
                speed = random.uniform(DYN_OBS_SPEED/2, DYN_OBS_SPEED)
                dyn_obstacles.append(DynamicObstacle(rp[0], rp[1], DYN_OBS_RADIUS, math.cos(angle)*speed, math.sin(angle)*speed))
            
        planned_path = []; flat_planned_path = []; path_index = 0; path_history = []

    reset_sim(new_map=True)

    def to_scr(pos): return int(pos[0]*scale)+40, int(WINDOW_SIZE - (pos[1]*scale)-40)
    def from_scr(sx, sy): return (sx - 40) / scale, (WINDOW_SIZE - sy - 40) / scale
    
    def draw_car(state, color=CAR_COLOR):
        x, y, yaw = state
        corners = get_car_corners(x, y, yaw)
        scr_corners = [to_scr(p) for p in corners]
        pygame.draw.polygon(screen, color, scr_corners)
        pygame.draw.polygon(screen, BLACK, scr_corners, 1)
        fx = x + CAR_L * math.cos(yaw); fy = y + CAR_L * math.sin(yaw)
        pygame.draw.line(screen, BLACK, to_scr((x,y)), to_scr((fx,fy)), 2)

    running = True
    while running:
        clock.tick(FPS)
        dt_frame = 1.0 / FPS
        
        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        bounds = [0, max(max(xs), max(ys)), 0, max(max(xs), max(ys))] if outer_poly else [0, 700, 0, 700]
        safe_car_radius = math.hypot(CAR_L/2 + 1.0, CAR_WIDTH/2)
        
        # --- XÁC ĐỊNH TRẠNG THÁI VỀ ĐÍCH ---
        is_finished = (click_step == 2 and not is_planning and flat_planned_path and path_index >= len(flat_planned_path))

        # --- XÁC ĐỊNH TAI NẠN THỰC TẾ (CRASH) ---
        if click_step >= 1 and not is_finished and not is_crashed:
            # t_lookahead = 0.0 nghĩa là kiểm tra đụng chạm vật lý ngay tại thời khắc hiện tại
            hit_now, _ = check_collision_with_index(current_state[0], current_state[1], current_state[2], outer_poly, real_holes, dyn_obstacles, t_lookahead=0.0)
            if hit_now:
                is_crashed = True
                print("💥 CRASH DETECTED! Shutting down planner.")

        # 1. Các quả bóng luôn di chuyển vật lý trong thế giới
        for obs in dyn_obstacles:
            active_robot_state = current_state if (click_step >= 1 and not is_finished) else None
            active_goal_pos = goal_pos if click_step >= 2 else None
            obs.move(dt_frame, bounds, outer_poly, real_holes, active_robot_state, safe_car_radius, active_goal_pos, GOAL_RADIUS)
            
        # ========================================================
        # 2. BỘ LỌC CẢM BIẾN (Chỉ nhận diện vật trong Radar)
        # ========================================================
        visible_dyn_obs = []
        if click_step >= 1 and not is_finished and not is_crashed:
            rx, ry = current_state[0], current_state[1]
            for obs in dyn_obstacles:
                if math.hypot(obs.x - rx, obs.y - ry) <= (SENSOR_RADIUS + obs.radius):
                    visible_dyn_obs.append(obs)

        # ========================================================
        # 3. LÙI KHẨN CẤP (Bỏ qua nếu đã đâm hoặc đã về đích)
        # ========================================================
        emergency_override = False
        if click_step >= 1 and not is_finished and not is_crashed: 
            hit, h_idx = check_collision_with_index(current_state[0], current_state[1], current_state[2], outer_poly, real_holes, visible_dyn_obs, t_lookahead=0.8)
            if hit and h_idx == -3:
                emergency_override = True

        if emergency_override:
            rev_dist = VELOCITY_MAX * dt_frame * 1.5
            nx = current_state[0] - rev_dist * math.cos(current_state[2])
            ny = current_state[1] - rev_dist * math.sin(current_state[2])

            w_hit, _ = check_collision_with_index(nx, ny, current_state[2], outer_poly, real_holes, None)
            if not w_hit:
                current_state = (nx, ny, current_state[2]) 
                if len(path_history) == 0 or dist(path_history[-1], (nx, ny)) > 0.5:
                    path_history.append((nx, ny))

            if click_step == 2:
                is_planning = True
                planned_path = []; flat_planned_path = []
                if algo_mode == "RRT":
                    planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, bounds, None)
                else:
                    planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, global_grid_penalties, None)

        # --- EVENT CHUỘT/BÀN PHÍM ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    if map_folders: current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim(True)
                elif event.key == pygame.K_r: reset_sim(False)
                elif event.key == pygame.K_TAB: algo_mode = "MCPP" if algo_mode == "RRT" else "RRT"; reset_sim(False)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if click_step < 2 and not is_crashed:
                    wx, wy = from_scr(event.pos[0], event.pos[1])
                    if click_step == 0:
                        current_state = (wx, wy, 0.0); click_step = 1
                    elif click_step == 1:
                        goal_pos = np.array([wx, wy]); goal_yaw = 0.0 
                        click_step = 2; is_planning = True
                        if algo_mode == "RRT":
                            planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, bounds, None)
                        else:
                            planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, global_grid_penalties, None)

        # ========================================================
        # 4. TÌM ĐƯỜNG & ĐẠP PHANH (Khóa nếu đã đâm hoặc đang lùi)
        # ========================================================
        if not emergency_override and not is_crashed:
            if is_planning and click_step == 2:
                for _ in range(20): 
                    path_segments = planner.plan_step()
                    if path_segments:
                        planned_path = path_segments; flat_planned_path = []
                        for seg in path_segments: flat_planned_path.extend(seg['points'])
                        is_planning = False; path_index = 0
                        print("Path Found!")
                        break
                
                limit = RRT_MAX_ITER if algo_mode == "RRT" else MCPP_ITER * 10
                if len(planner.node_list) > limit:
                    print("Retry planning...")
                    reset_sim(False) 

            elif flat_planned_path and path_index < len(flat_planned_path) and click_step == 2:
                collision_detected_static = False
                dynamic_obstacle_incoming = False
                look_limit = min(path_index + LOOKAHEAD_STEPS * 2, len(flat_planned_path)) 
                
                for i in range(path_index, look_limit):
                    fs = flat_planned_path[i]
                    collided, hit_idx = check_collision_with_index(fs[0], fs[1], fs[2], outer_poly, real_holes, visible_dyn_obs, t_lookahead=1.5)
                    
                    if collided:
                        if hit_idx >= 0:
                            collision_detected_static = True
                            if hit_idx not in known_hole_indices:
                                known_hole_indices.add(hit_idx)
                                planner_holes_geom.append(real_holes[hit_idx])
                                for pt in real_holes[hit_idx]:
                                    gid = (int(pt[0] // BIG_GRID_SIZE), int(pt[1] // BIG_GRID_SIZE))
                                    global_grid_penalties[gid] = global_grid_penalties.get(gid, 0) + 5000.0 
                            break 
                        elif hit_idx == -3:
                            dynamic_obstacle_incoming = True
                            break 
                
                if collision_detected_static:
                    is_planning = True; planned_path = []; flat_planned_path = []
                    if algo_mode == "RRT":
                        planner = KinematicRRT(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, bounds, None)
                    else:
                        planner = KinematicMCPP(current_state, goal_pos, goal_yaw, outer_poly, planner_holes_geom, global_grid_penalties, None)
                
                elif dynamic_obstacle_incoming:
                    pass 
                else:
                    if path_index < len(flat_planned_path):
                        current_state = flat_planned_path[path_index]
                        path_history.append((current_state[0], current_state[1]))
                        path_index += 1
                    else:
                        current_state = flat_planned_path[-1]

        # --- LOGIC CHỮ HIỂN THỊ UI ---
        ui_status = ""
        ui_color = BLUE
        if is_crashed: ui_status = "CRASHED! PRESS 'R' TO RESTART"; ui_color = BLACK
        elif emergency_override: ui_status = "DANGER! REVERSING!"; ui_color = RED
        elif click_step == 0: ui_status = "CLICK CHON DIEM DAU"
        elif click_step == 1: ui_status = "CLICK CHON DIEM DICH"
        else:
            if not is_planning and flat_planned_path and path_index < len(flat_planned_path):
                if 'dynamic_obstacle_incoming' in locals() and dynamic_obstacle_incoming:
                    ui_status = "BRAKING..."; ui_color = (255, 140, 0)
                else: ui_status = "MOVING"; ui_color = GREEN
            elif is_finished: ui_status = "FINISHED"; ui_color = GREEN
            elif is_planning: ui_status = "PLANNING..."; ui_color = RED

        # ========================================================
        # 5. DRAW (VẼ ĐỒ HỌA MÔ PHỎNG FOG OF WAR)
        # ========================================================
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        
        for i, h in enumerate(real_holes):
            col = RED if i in known_hole_indices else GHOST_GRAY
            pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

        if click_step >= 1 and not is_finished and not is_crashed:
            pygame.draw.circle(screen, (200, 230, 255), to_scr(current_state[:2]), int(SENSOR_RADIUS * scale), 1)

        for obs in dyn_obstacles:
            if click_step >= 1 and obs in visible_dyn_obs:
                pygame.draw.circle(screen, DYN_COLOR, to_scr((obs.x, obs.y)), int(obs.radius * scale))
                fut_x = obs.x + obs.vx * 1.5; fut_y = obs.y + obs.vy * 1.5
                pygame.draw.line(screen, (255, 180, 180), to_scr((obs.x, obs.y)), to_scr((fut_x, fut_y)), 3)
            else:
                pygame.draw.circle(screen, (220, 220, 220), to_scr((obs.x, obs.y)), int(obs.radius * scale))

        if is_planning and click_step == 2 and not is_crashed:
            for node in planner.node_list:
                parent = getattr(node, 'parent', None) or getattr(node, 'parent_node', None)
                if parent:
                    pts = [to_scr((px, py)) for px, py in zip(node.path_x, node.path_y)]
                    if len(pts)>1: pygame.draw.lines(screen, (200, 200, 255), False, pts, 1)

        if not is_planning and planned_path and click_step == 2:
            for seg in planned_path:
                points = seg['points']
                if len(points) > 1:
                    pts_scr = [to_scr((p[0], p[1])) for p in points]
                    if seg['is_dubins']: col = DUBINS_COLOR; w = 4
                    elif seg['direction'] == -1: col = REVERSE_COLOR; w = 2
                    else: col = GREEN; w = 2
                    pygame.draw.lines(screen, col, False, pts_scr, w)

        if len(path_history) > 1:
            pygame.draw.lines(screen, BLACK, False, [to_scr(p) for p in path_history], 1)

        if click_step >= 2:
            g_scr = to_scr(goal_pos)
            pygame.draw.circle(screen, BLUE, g_scr, int(GOAL_RADIUS * scale))
            arrow_end = (goal_pos[0] + 4.0*math.cos(goal_yaw), goal_pos[1] + 4.0*math.sin(goal_yaw))
            pygame.draw.line(screen, BLUE, g_scr, to_scr(arrow_end), 3)

        # Đổi màu xe thành Đen nếu bị đâm
        if click_step >= 1: 
            car_draw_color = BLACK if is_crashed else CAR_COLOR
            draw_car(current_state, car_draw_color)

        screen.blit(font.render(f"Mode: {algo_mode} | [TAB] Switch | [N] Next | [R] Reset", True, BLUE), (10, 10))
        screen.blit(font.render(f"Status: {ui_status}", True, ui_color), (10, 30))
        screen.blit(font.render("GREEN: Fwd | ORANGE: Rev | PURPLE: Dubins", True, BLACK), (10, 50))
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()