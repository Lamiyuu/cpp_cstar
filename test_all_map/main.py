import pygame
import sys
import os
import glob
import math
import json
import numpy as np
import random

from config import *
from core_math import get_car_corners, dist, point_to_segment_dist, point_in_polygon
from environment import load_data, get_valid_random_pos, DynamicObstacle, check_collision_with_index
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

SCENARIO_FILE = "scenarios_ac300.json"

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
    """
    Hỗ trợ goals dạng (x, y) hoặc (x, y, yaw).
    Khi sắp xếp chỉ dùng (x, y) nhưng giữ nguyên góc đỗ yaw đi kèm từng goal.
    """
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
    pygame.display.set_caption("Stage-by-Stage Planning: Original vs Paper Algorithms")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 15)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC12_*")))
    
    current_map_idx = 0
    algo_idx = 0
    algo_mode = ALGO_MODES[algo_idx]
    
    outer_poly = []; real_holes = []
    known_hole_indices = set(); planner_holes_geom = []
    current_state = (0.0, 0.0, 0.0)
    planner = None
    
    planned_path = []; flat_planned_path = []; path_index = 0; path_history = []
    dyn_obstacles = [] 
    
    # Biến cho Stage-by-stage: mỗi goal có dạng (x, y, yaw)
    multi_goals = [] 
    ordered_goals = []
    current_stage_idx = 0
    
    sim_report = None 
    drag_start_world = None
    
    click_step = 0 
    is_planning = False
    is_crashed = False 
    is_finished_all = False
    scale = 1.0

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

    def get_current_map_key():
        if map_folders:
            return os.path.basename(map_folders[current_map_idx])
        return "Dummy_Map"

    def load_scenarios():
        if os.path.exists(SCENARIO_FILE):
            try:
                with open(SCENARIO_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_current_scenario():
        if click_step < 1 or not multi_goals:
            print("⚠️ Chưa có đủ Start và Goals để lưu kịch bản!")
            return
        map_key = get_current_map_key()
        data = load_scenarios()
        data[map_key] = {
            "start": [round(current_state[0], 2), round(current_state[1], 2), round(current_state[2], 4)],
            "goals": [[round(g[0], 2), round(g[1], 2), round(g[2] if len(g) >= 3 and g[2] is not None else 0.0, 4)] for g in multi_goals]
        }
        with open(SCENARIO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"💾 Đã lưu kịch bản cố định cho [{map_key}] vào {SCENARIO_FILE}!")

    def plan_next_stage(start_state, target_goal):
        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        bounds = [0, max(max(xs), max(ys)), 0, max(max(xs), max(ys))] if outer_poly else [0, 700, 0, 700]
        
        target_xy = (target_goal[0], target_goal[1])
        has_custom_yaw = (len(target_goal) >= 3 and target_goal[2] is not None)

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

        # LUÔN TÍNH XƯƠNG SỐNG A* ĐỂ DẪN HƯỚNG CHO MỌI THUẬT TOÁN
        waypoints = get_topological_path(start_state[:2], target_xy, bounds, get_clearance_for_astar, grid_res=2.0)
        
        if has_custom_yaw:
            goal_yaw = target_goal[2]
        elif len(waypoints) >= 2:
            goal_yaw = math.atan2(waypoints[-1][1] - waypoints[-2][1], waypoints[-1][0] - waypoints[-2][0])
        else:
            goal_yaw = math.atan2(target_xy[1] - start_state[1], target_xy[0] - start_state[0])

        print(f"📍 A*: Xương sống chặng {current_stage_idx + 1} ({len(waypoints)} điểm) | Góc đỗ: {math.degrees(goal_yaw):.1f}° -> {algo_mode}")
        
        if algo_mode == "MCPP_Original":
            return KinematicMCPP(start_state, np.array(target_xy), goal_yaw, outer_poly, planner_holes_geom, bounds, waypoints, None)
        elif algo_mode == "RRT_Original":
            return KinematicRRT(start_state, np.array(target_xy), goal_yaw, outer_poly, planner_holes_geom, bounds, None, waypoints=waypoints)
        else:
            if algo_mode == "RRT*": planner_obj = RRTStar(start_state, target_xy, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "BI-RRT*": planner_obj = BiRRTStar(start_state, target_xy, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "APF-RRT*": planner_obj = APFRRTStar(start_state, target_xy, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "BI-APF-RRT*": planner_obj = BiAPFRRTStar(start_state, target_xy, goal_yaw, outer_poly, planner_holes_geom, bounds)
            elif algo_mode == "MS-BI-RRT*": planner_obj = MSBiRRTStar(start_state, target_xy, goal_yaw, outer_poly, planner_holes_geom, bounds)
            planner_obj.waypoints = waypoints
            return planner_obj

    def reset_sim(new_map=False):
        nonlocal outer_poly, real_holes, current_state, known_hole_indices, planner_holes_geom
        nonlocal planner, planned_path, flat_planned_path, path_index, is_planning, scale, path_history, click_step
        nonlocal dyn_obstacles, is_crashed, multi_goals, ordered_goals, current_stage_idx, is_finished_all
        nonlocal sim_report, drag_start_world
        
        sim_report = EnergyTracker()
        drag_start_world = None
        
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
        
        click_step = 0; is_planning = False; planner = None; is_crashed = False; is_finished_all = False
        current_state = (0.0, 0.0, 0.0)
        
        multi_goals.clear(); ordered_goals.clear()
        current_stage_idx = 0
        known_hole_indices.clear(); planner_holes_geom.clear()
        planned_path.clear(); flat_planned_path.clear(); path_history.clear()
        path_index = 0
        
        if new_map: 
            dyn_obstacles.clear()

        # --- TỰ ĐỘNG NẠP KỊCH BẢN TỪ SCENARIOS_AC300.JSON NẾU ĐÃ CÓ ---
        map_key = get_current_map_key()
        scenarios = load_scenarios()
        if map_key in scenarios:
            sc = scenarios[map_key]
            current_state = tuple(sc["start"])
            multi_goals.extend([tuple(g) for g in sc["goals"]])
            
            # Quét Rada lần đầu tại điểm xuất phát trước khi lập kế hoạch chặng 1
            rx, ry = current_state[0], current_state[1]
            for i, h in enumerate(real_holes):
                for j in range(len(h)):
                    A = h[j]; B = h[(j+1)%len(h)]
                    if point_to_segment_dist(rx, ry, A[0], A[1], B[0], B[1]) <= SENSOR_RADIUS:
                        known_hole_indices.add(i)
                        planner_holes_geom.append(h)
                        break

            start_coord = (current_state[0], current_state[1])
            ordered_goals.extend(amugopia_sabo_ordering(start_coord, multi_goals))
            current_stage_idx = 0
            click_step = 2
            is_planning = True
            print(f"🎯 Đã tự động nạp kịch bản [{map_key}]: Start=({current_state[0]:.1f}, {current_state[1]:.1f}, {math.degrees(current_state[2]):.1f}°), Số Goal={len(multi_goals)}")
            planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])

    reset_sim(new_map=True)

    running = True
    while running:
        clock.tick(FPS)
        dt_frame = 1.0 / FPS
        
        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        bounds = [0, max(max(xs), max(ys)), 0, max(max(xs), max(ys))] if outer_poly else [0, 700, 0, 700]
        safe_car_radius = math.hypot(CAR_L/2 + 1.0, CAR_WIDTH/2)
        
        is_stage_finished = False
        path_planning_failed = False
        
        # --- KIỂM TRA TRẠNG THÁI VỀ ĐÍCH CỦA CHẶNG ---
        if click_step == 2 and not is_planning and not is_finished_all:
            if flat_planned_path and path_index >= len(flat_planned_path):
                current_target = ordered_goals[current_stage_idx]
                dist_to_target = math.hypot(current_state[0] - current_target[0], current_state[1] - current_target[1])
                
                if dist_to_target < 60:
                    is_stage_finished = True
                else:
                    path_planning_failed = True

        # --- KIỂM TRA VA CHẠM THỰC TẾ ---
        if click_step >= 1 and not is_stage_finished and not is_crashed:
            hit_now, _ = check_collision_with_index(current_state[0], current_state[1], current_state[2], outer_poly, real_holes, dyn_obstacles, t_lookahead=0.0)
            if hit_now:
                is_crashed = True
                print("💥 CRASH DETECTED!")

        # --- CẬP NHẬT VẬT CẢN ĐỘNG ---
        for obs in dyn_obstacles:
            active_robot_state = current_state if (click_step >= 1 and not is_stage_finished) else None
            active_goal_pos = ordered_goals[current_stage_idx][:2] if (click_step >= 2 and current_stage_idx < len(ordered_goals)) else None
            obs.move(dt_frame, bounds, outer_poly, real_holes, active_robot_state, safe_car_radius, active_goal_pos, GOAL_RADIUS)
            
        visible_dyn_obs = []
        if click_step >= 1 and not is_stage_finished and not is_crashed:
            rx, ry = current_state[0], current_state[1]
            for obs in dyn_obstacles:
                if math.hypot(obs.x - rx, obs.y - ry) <= (SENSOR_RADIUS + obs.radius):
                    visible_dyn_obs.append(obs)

        # ========================================================
        # 1. RADA QUÉT VẬT CẢN MỚI (FOG OF WAR)
        # ========================================================
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
                            print(f"📡 Rada phát hiện vật cản mới #{i}!")
                            break

        # ========================================================
        # 2. KIỂM TRA LỘ TRÌNH VÀ RE-PLAN TẠI CHỖ
        # ========================================================
        if click_step >= 2 and not is_planning and not is_crashed and not is_finished_all:
            if flat_planned_path and path_index < len(flat_planned_path):
                path_is_blocked = False
                
                for i in range(path_index, len(flat_planned_path)):
                    pt = flat_planned_path[i]
                    yaw = pt[2] if len(pt) > 2 else current_state[2]
                    
                    hit, _ = check_collision_with_index(pt[0], pt[1], yaw, outer_poly, planner_holes_geom, None)
                    if hit:
                        path_is_blocked = True
                        break
                
                if path_is_blocked:
                    print("⚠️ Phát hiện vật cản mới chặn đường! Dừng xe và Re-planning...")
                    if sim_report:
                        sim_report.add_obstacle()
                        sim_report.add_replan()
                    
                    is_planning = True
                    planned_path.clear()
                    flat_planned_path.clear()
                    path_index = 0
                    planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])

        # --- NÉ VẬT CẢN ĐỘNG KHẨN CẤP ---
        emergency_override = False
        if click_step >= 1 and not is_stage_finished and not is_crashed: 
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
                planned_path.clear(); flat_planned_path.clear(); path_index = 0
                planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])

        # ========================================================
        # EVENT CHUỘT / BÀN PHÍM (KÉO CHUỘT CHỌN GÓC ĐỖ & LƯU KỊCH BẢN)
        # ========================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    if map_folders: current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim(True)
                elif event.key == pygame.K_r: 
                    reset_sim(False)
                elif event.key == pygame.K_s:
                    save_current_scenario()
                elif event.key == pygame.K_c:
                    map_key = get_current_map_key()
                    data = load_scenarios()
                    if map_key in data:
                        del data[map_key]
                        with open(SCENARIO_FILE, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)
                        print(f"🗑️ Đã xóa kịch bản cố định của [{map_key}]. Kéo chuột để tạo mới!")
                    reset_sim(False)
                elif event.key == pygame.K_TAB: 
                    algo_idx = (algo_idx + 1) % len(ALGO_MODES)
                    algo_mode = ALGO_MODES[algo_idx]
                    print(f"🔄 Chuyển sang thuật toán: {algo_mode}")
                    reset_sim(False)
                elif event.key == pygame.K_RETURN and click_step == 1 and len(multi_goals) > 0:
                    save_current_scenario()
                    click_step = 2
                    print(f"🚀 TÍNH TOÁN NHIỆM VỤ ĐA MỤC TIÊU (AMuGOPIA)...")
                    start_coord = (current_state[0], current_state[1])
                    ordered_goals.clear()
                    ordered_goals.extend(amugopia_sabo_ordering(start_coord, multi_goals))
                    current_stage_idx = 0
                    is_planning = True
                    print(f"📋 Tổng số trạm cần đi: {len(ordered_goals)}")
                    planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not is_crashed and click_step in [0, 1]:
                    drag_start_world = from_scr(event.pos[0], event.pos[1])

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if not is_crashed and drag_start_world is not None:
                    release_world = from_scr(event.pos[0], event.pos[1])
                    dx = release_world[0] - drag_start_world[0]
                    dy = release_world[1] - drag_start_world[1]
                    chosen_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1.5 else 0.0
                    
                    if click_step == 0:
                        current_state = (drag_start_world[0], drag_start_world[1], chosen_yaw)
                        click_step = 1
                        print(f"📍 Đặt Start: ({current_state[0]:.1f}, {current_state[1]:.1f}), Góc đầu xe: {math.degrees(chosen_yaw):.1f}°. Kéo thả tiếp các Goal rồi nhấn ENTER.")
                    elif click_step == 1:
                        multi_goals.append((drag_start_world[0], drag_start_world[1], chosen_yaw))
                        print(f"🚩 Đã thêm Goal #{len(multi_goals)}: ({drag_start_world[0]:.1f}, {drag_start_world[1]:.1f}), Góc đỗ: {math.degrees(chosen_yaw):.1f}°")
                    
                    drag_start_world = None

        # ========================================================
        # TÌM ĐƯỜNG VÀ DI CHUYỂN XE
        # ========================================================
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

        # ========================================================
        # LOGIC CHUYỂN CHẶNG KHI ĐẾN ĐÍCH
        # ========================================================
        if is_stage_finished and not is_finished_all:
            if current_stage_idx < len(ordered_goals) - 1:
                print(f"🏁 ĐÃ ĐẾN ĐÍCH {current_stage_idx + 1}! Chuẩn bị đi chặng {current_stage_idx + 2}...")
                current_stage_idx += 1
                current_state = (flat_planned_path[-1][0], flat_planned_path[-1][1], flat_planned_path[-1][2])
                
                planner = plan_next_stage(current_state, ordered_goals[current_stage_idx])
                planned_path.clear(); flat_planned_path.clear(); path_index = 0
                is_planning = True
            else:
                print(f"🏆 ĐÃ HOÀN THÀNH TOÀN BỘ CÁC ĐÍCH!")
                is_finished_all = True
                
                if sim_report:
                    current_map_name = get_current_map_key()
                    safe_algo_name = algo_mode.replace("*", "Star")
                    sim_report.export_report(f"Results/Report_{safe_algo_name}_{current_map_name}.txt") 
                    sim_report.export_csv(map_name=f"{safe_algo_name}_{current_map_name}", filename="Results/Bao_Cao_Nang_Luong_Tong_Hop.csv")
                
                if map_folders and current_map_idx < len(map_folders) - 1:
                    print("⏩ Chuyển Map tiếp theo sau 1.5s...")
                    pygame.display.flip()
                    pygame.time.delay(1500)
                    current_map_idx += 1
                    reset_sim(new_map=True)
                    
        # --- CƠ CHẾ GỠ KẸT ---
        elif path_planning_failed and not is_crashed:
            print("⚠️ Bị kẹt đường! Tự lùi xe để tìm lại...")
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
        if is_crashed: ui_status = "CRASHED! PRESS 'R' TO RESTART | 'C' CLEAR SCENARIO"; ui_color = BLACK
        elif emergency_override: ui_status = "DANGER! REVERSING!"; ui_color = RED
        elif click_step == 0: ui_status = "CLICK & DRAG START POS (POSITION + YAW)"
        elif click_step == 1: ui_status = "CLICK & DRAG GOALS -> ENTER (AUTO-SAVE)"
        else:
            if is_finished_all: ui_status = "ALL MISSIONS COMPLETE"; ui_color = GREEN
            elif path_planning_failed: ui_status = "PLAN FAILED! REVERSING..."; ui_color = (255, 140, 0)
            elif is_planning: ui_status = f"PLANNING STAGE {current_stage_idx+1}/{len(ordered_goals)}..."; ui_color = RED
            elif not is_planning and flat_planned_path and path_index < len(flat_planned_path):
                if 'dynamic_obstacle_incoming' in locals() and dynamic_obstacle_incoming:
                    ui_status = "BRAKING..."; ui_color = (255, 140, 0)
                else: ui_status = f"MOVING TO GOAL {current_stage_idx+1}"; ui_color = GREEN

        # ========================================================
        # VẼ GIAO DIỆN ĐỒ HỌA
        # ========================================================
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        
        for i, h in enumerate(real_holes):
            col = RED if i in known_hole_indices else GHOST_GRAY
            pygame.draw.polygon(screen, col, [to_scr(p) for p in h])

        if click_step >= 1 and not is_finished_all and not is_crashed:
            pygame.draw.circle(screen, (200, 230, 255), to_scr(current_state[:2]), int(SENSOR_RADIUS * scale), 1)

        for obs in dyn_obstacles:
            if click_step >= 1 and obs in visible_dyn_obs:
                pygame.draw.circle(screen, DYN_COLOR, to_scr((obs.x, obs.y)), int(obs.radius * scale))
                fut_x = obs.x + obs.vx * 1.5; fut_y = obs.y + obs.vy * 1.5
                pygame.draw.line(screen, (255, 180, 180), to_scr((obs.x, obs.y)), to_scr((fut_x, fut_y)), 3)
            else:
                pygame.draw.circle(screen, (220, 220, 220), to_scr((obs.x, obs.y)), int(obs.radius * scale))

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

        # Vẽ cây RRT khi đang tính toán
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

        # Vẽ đường đi hoàn chỉnh
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

        # Vẽ mũi tên xem trước khi đang nhấn giữ và kéo chuột chọn góc đỗ
        if drag_start_world is not None:
            mx_scr, my_scr = pygame.mouse.get_pos()
            pygame.draw.circle(screen, RED, to_scr(drag_start_world), 5)
            pygame.draw.line(screen, RED, to_scr(drag_start_world), (mx_scr, my_scr), 2)

        # Vẽ các Goal kèm kim chỉ hướng góc đỗ xe
        if click_step >= 1:
            for i, goal in enumerate(multi_goals):
                g_scr = to_scr((goal[0], goal[1]))
                try:
                    order_in_route = ordered_goals.index(goal)
                    is_visited = order_in_route < current_stage_idx
                except ValueError:
                    is_visited = False
                
                color = (150, 150, 150) if is_visited else BLUE
                pygame.draw.circle(screen, color, g_scr, int(GOAL_RADIUS * scale))
                
                if len(goal) >= 3 and goal[2] is not None:
                    arrow_len = 6.0
                    gx_end = goal[0] + arrow_len * math.cos(goal[2])
                    gy_end = goal[1] + arrow_len * math.sin(goal[2])
                    pygame.draw.line(screen, RED if not is_visited else (120, 120, 120), g_scr, to_scr((gx_end, gy_end)), 2)

                text = font.render(str(i+1), True, (255, 255, 255))
                screen.blit(text, (g_scr[0]-5, g_scr[1]-8))

        if click_step >= 1: 
            draw_car(current_state, BLACK if is_crashed else CAR_COLOR)

        screen.blit(font.render(f"Mode: {algo_mode} | [TAB] Switch | [N] Next | [R] Reset | [S] Save | [C] Clear", True, BLUE), (10, 10))
        screen.blit(font.render(f"Status: {ui_status}", True, ui_color), (10, 30))
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()