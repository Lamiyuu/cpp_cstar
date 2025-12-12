import pygame
import numpy as np
import math
import random
import sys
import os
import glob

# --- CẤU HÌNH ---
WINDOW_SIZE = 900
FPS = 60
DATASET_DIR = "AC300"

# TỐC ĐỘ MỌC CÂY (Quan trọng)
# Số nhánh cây mọc ra trong 1 khung hình (Tăng lên để chạy nhanh hơn)
GROWTH_SPEED = 1

# SỐ LƯỢNG ĐÍCH
NUM_GOALS = 5

# MÀU SẮC
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)      # Hố
GRAY = (220, 220, 220)   # Cây RRT
GREEN = (0, 180, 0)      # Start
BLUE = (0, 0, 255)       # Goal chưa đến
CYAN = (0, 255, 255)     # Goal đã đến
PURPLE = (148, 0, 211)   # Đường đi tìm thấy
YELLOW = (255, 200, 0)   # Nhánh mới nhất (Highlight)

# THÔNG SỐ RRT
RRT_STEP_LEN = 3.0
RRT_GOAL_SAMPLE_RATE = 0.1

# --- HÀM HÌNH HỌC & LOAD DATA ---
def dist(a, b): return np.linalg.norm(a - b)

def point_in_polygon(point, polygon):
    x, y = point; n = len(polygon); inside = False; p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def load_data(folder):
    outer = []; holes = []
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
                    if len(curr)>2: holes.append(curr)
                    curr = []
                else:
                    try: curr.append((float(parts[0]), float(parts[1])))
                    except: pass
        if len(curr)>2: holes.append(curr)
    return outer, holes

def get_valid_random_pos(outer_poly, holes):
    if not outer_poly: return np.array([50.0, 50.0])
    xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
    min_x, max_x = min(xs), max(xs); min_y, max_y = min(ys), max(ys)
    for _ in range(200):
        rx = random.uniform(min_x, max_x); ry = random.uniform(min_y, max_y)
        cand = np.array([rx, ry])
        valid = True
        if not point_in_polygon(cand, outer_poly): valid = False
        if valid:
            for h in holes:
                if point_in_polygon(cand, h): valid = False; break
        if valid: return cand
    return np.array([50.0, 50.0])

# --- INCREMENTAL RRT PLANNER (CHẠY TỪNG BƯỚC) ---
class IncrementalRRT:
    class Node:
        def __init__(self, x, y):
            self.x = x; self.y = y; self.parent = None
            
    def __init__(self, start, goals, outer, holes, bounds):
        self.start = self.Node(start[0], start[1])
        self.goals = [self.Node(g[0], g[1]) for g in goals]
        self.outer = outer
        self.holes = holes
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        
        self.node_list = [self.start]
        self.found_paths = {} # {goal_idx: [path_points]}
        self.goals_reached = [False] * len(self.goals)
        self.last_added_node = None # Để vẽ highlight

    def is_collision(self, x, y):
        pt = (x, y)
        if not point_in_polygon(pt, self.outer): return True
        for h in self.holes:
            if point_in_polygon(pt, h): return True
        return False

    def step(self):
        """Thực hiện MỘT bước mở rộng cây (1 iteration)"""
        # Nếu đã tìm hết đích thì dừng để tiết kiệm CPU
        if all(self.goals_reached):
            return

        # 1. Sample
        unreached = [i for i, v in enumerate(self.goals_reached) if not v]
        if unreached and random.random() < RRT_GOAL_SAMPLE_RATE:
            rnd = self.goals[random.choice(unreached)]
        else:
            rnd = self.Node(random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))

        # 2. Nearest
        dists = [(node.x-rnd.x)**2 + (node.y-rnd.y)**2 for node in self.node_list]
        nearest = self.node_list[dists.index(min(dists))]

        # 3. Steer
        theta = math.atan2(rnd.y - nearest.y, rnd.x - nearest.x)
        new_node = self.Node(
            nearest.x + RRT_STEP_LEN * math.cos(theta),
            nearest.y + RRT_STEP_LEN * math.sin(theta)
        )
        new_node.parent = nearest

        # 4. Collision Check
        if not self.is_collision(new_node.x, new_node.y):
            self.node_list.append(new_node)
            self.last_added_node = new_node
            
            # 5. Check Goals
            for i, goal in enumerate(self.goals):
                if not self.goals_reached[i]:
                    dx = new_node.x - goal.x
                    dy = new_node.y - goal.y
                    if math.sqrt(dx*dx + dy*dy) <= RRT_STEP_LEN:
                        # Kết nối vào đích
                        final_node = self.Node(goal.x, goal.y)
                        final_node.parent = new_node
                        self.node_list.append(final_node)
                        
                        # Lưu đường đi
                        path = []
                        curr = final_node
                        while curr:
                            path.append(np.array([curr.x, curr.y]))
                            curr = curr.parent
                        self.found_paths[i] = path
                        self.goals_reached[i] = True

# --- MAIN LOOP ---
def main():
    global GROWTH_SPEED
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Visualizing RRT Growth")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    font_big = pygame.font.SysFont("Consolas", 24)

    # Load Maps
    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC10_*")))
    if not map_folders: return

    # State
    current_map_idx = 0
    planner = None
    outer_poly = []; holes = []
    scale = 1.0
    start_pos = None; goal_positions = []

    def reset_sim():
        nonlocal planner, outer_poly, holes, scale, start_pos, goal_positions
        folder = map_folders[current_map_idx]
        print(f"Loading: {folder}")
        outer_poly, holes = load_data(folder)
        
        if outer_poly:
            xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
            mx = max(max(xs), max(ys))
            scale = (WINDOW_SIZE - 50) / mx
            
            # Tạo Start & Goals
            min_x, min_y = min(xs), min(ys)
            start_pos = np.array([min_x+2.0, min_y+2.0])
            goal_positions = []
            for _ in range(NUM_GOALS):
                goal_positions.append(get_valid_random_pos(outer_poly, holes))
            
            # Khởi tạo Planner
            bounds = [0, mx, 0, mx]
            planner = IncrementalRRT(start_pos, goal_positions, outer_poly, holes, bounds)
        else:
            scale = 1.0

    reset_sim()

    def to_scr(pos):
        return int(pos[0]*scale)+25, int(WINDOW_SIZE - pos[1]*scale)-25

    running = True
    while running:
        # 1. Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    current_map_idx = (current_map_idx + 1) % len(map_folders)
                    reset_sim()
                elif event.key == pygame.K_p: 
                    current_map_idx = (current_map_idx - 1) % len(map_folders)
                    reset_sim()
                elif event.key == pygame.K_r: reset_sim()
                elif event.key == pygame.K_UP: GROWTH_SPEED += 5
                elif event.key == pygame.K_DOWN: GROWTH_SPEED = max(1, GROWTH_SPEED - 5)

        # 2. Logic: Mọc cây (Chạy nhiều bước mỗi frame)
        if planner:
            for _ in range(GROWTH_SPEED):
                planner.step()

        # 3. Rendering
        screen.fill(WHITE)
        
        # Vẽ Map
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        for h in holes: pygame.draw.polygon(screen, RED, [to_scr(p) for p in h])

        if planner:
            # Vẽ Cây RRT (Các nhánh xám)
            # (Vẽ từng line sẽ chậm nếu cây quá lớn -> Optimization: vẽ ra Surface riêng nếu cần)
            # Để đơn giản và trực quan, ta vẽ trực tiếp các node mới nhất
            for node in planner.node_list:
                if node.parent:
                    start_pt = to_scr((node.parent.x, node.parent.y))
                    end_pt = to_scr((node.x, node.y))
                    pygame.draw.line(screen, GRAY, start_pt, end_pt, 1)
            
            # Highlight nhánh mới nhất (Màu vàng)
            if planner.last_added_node and planner.last_added_node.parent:
                n = planner.last_added_node
                pygame.draw.line(screen, YELLOW, to_scr((n.parent.x, n.parent.y)), to_scr((n.x, n.y)), 3)

            # Vẽ các đường đi đã tìm thấy (Màu Tím)
            for idx, path in planner.found_paths.items():
                if len(path) > 1:
                    scr_path = [to_scr(p) for p in path]
                    pygame.draw.lines(screen, PURPLE, False, scr_path, 3)

        # Vẽ Start & Goals
        pygame.draw.circle(screen, GREEN, to_scr(start_pos), 6)
        for i, g in enumerate(goal_positions):
            reached = False
            if planner and planner.goals_reached[i]: reached = True
            color = CYAN if reached else BLUE
            pygame.draw.circle(screen, color, to_scr(g), 8)
            # Vẽ số
            # screen.blit(font.render(str(i+1), True, WHITE), to_scr(g))

        # GUI Info
        map_name = os.path.basename(map_folders[current_map_idx])
        screen.blit(font_big.render(f"Map: {map_name}", True, BLACK), (10, 10))
        
        reached_count = sum(planner.goals_reached) if planner else 0
        screen.blit(font.render(f"Goals Found: {reached_count}/{NUM_GOALS}", True, BLUE), (10, 40))
        screen.blit(font.render(f"Tree Nodes: {len(planner.node_list) if planner else 0}", True, BLACK), (10, 60))
        screen.blit(font.render(f"Growth Speed: {GROWTH_SPEED} (UP/DOWN to change)", True, RED), (10, 80))
        screen.blit(font.render("[N]: Next Map | [R]: Reset", True, (100,100,100)), (10, WINDOW_SIZE-30))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()