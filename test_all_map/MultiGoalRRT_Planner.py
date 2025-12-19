import pygame
import numpy as np
import math
import random
import sys
import os
import glob
import itertools

# --- CẤU HÌNH ---
WINDOW_SIZE = 900
FPS = 60
DATASET_DIR = "AC300"
GROWTH_SPEED = 50
NUM_GOALS = 5
SENSOR_RADIUS = 60.0

# --- BẢNG MÀU ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

OBSTACLE_HIDDEN = (245, 245, 245)
OBSTACLE_LIT = (255, 50, 50)
OBSTACLE_OUTLINE = (220, 220, 220)

TREE_PALETTE = [
    (0, 150, 0), (0, 0, 200), (120, 0, 180), 
    (200, 100, 0), (0, 120, 120), (100, 50, 10), (200, 10, 100)
]
GRAPH_EDGE_COLOR = (200, 200, 200) # Làm mờ đường RRT gốc đi
FINAL_PATH_COLOR = (255, 0, 0)     # Đỏ: Đường RRT thô (Zigzag)
SMOOTH_PATH_COLOR = (50, 255, 50)  # XANH LÁ: Đường đã tối ưu (Thẳng tắp)

RRT_STEP_LEN = 5.0
RRT_GOAL_SAMPLE_RATE = 0.05

# --- HÀM HÌNH HỌC ---
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

def check_line_collision(p1, p2, outer, holes, step=5.0):
    """Kiểm tra va chạm đoạn thẳng"""
    distance = np.linalg.norm(p2 - p1)
    if distance < 1e-3: return False
    num_steps = int(distance / step) + 1
    for i in range(num_steps + 1):
        t = i / num_steps
        pt = p1 + (p2 - p1) * t
        if outer and not point_in_polygon(pt, outer): return True
        for h in holes:
            if point_in_polygon(pt, h): return True
    return False

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

# --- MAIN CLASS ---
class MultiTreeExplorer:
    class Node:
        def __init__(self, x, y):
            self.x = x; self.y = y; self.parent = None
            
    def __init__(self, start, goals, outer, holes, bounds):
        self.outer = outer
        self.holes = holes
        self.discovered_holes = [False] * len(holes)
        
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.trees = {} 
        self.trees[0] = [self.Node(start[0], start[1])]
        self.start_pos = start
        self.goals = goals
        self.goals_status = [False] * len(goals)
        
        for i, g in enumerate(goals):
            self.trees[i + 1] = [self.Node(g[0], g[1])]
            
        self.found_paths = {} 
        self.last_added_node = None
        
        # TSP Variables
        self.tsp_path_calculated = False
        self.tsp_sequence = []     
        self.raw_tour_points = []    # Đường RRT thô (Zigzag)
        self.smooth_tour_points = [] # Đường đã làm mượt (Optimized)

    def check_collision_and_sense(self, x, y):
        pt = (x, y)
        if not point_in_polygon(pt, self.outer): return True
        collision = False
        for i, h in enumerate(self.holes):
            if point_in_polygon(pt, h):
                self.discovered_holes[i] = True 
                collision = True
            if not self.discovered_holes[i]:
                for vert in h:
                    if (vert[0]-x)**2 + (vert[1]-y)**2 < SENSOR_RADIUS**2:
                        self.discovered_holes[i] = True
                        break
        return collision

    def grow_tree(self, tree_id):
        tree_nodes = self.trees[tree_id]
        rnd = None
        if random.random() < RRT_GOAL_SAMPLE_RATE:
            if tree_id == 0: 
                unreached = [i+1 for i, v in enumerate(self.goals_status) if not v]
                if unreached:
                    target_tree = self.trees[random.choice(unreached)]
                    target_node = random.choice(target_tree)
                    rnd = self.Node(target_node.x, target_node.y)
            else:
                target_node = random.choice(self.trees[0])
                rnd = self.Node(target_node.x, target_node.y)

        if rnd is None:
            rnd = self.Node(random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))

        dists = [(node.x-rnd.x)**2 + (node.y-rnd.y)**2 for node in tree_nodes]
        nearest = tree_nodes[dists.index(min(dists))]

        theta = math.atan2(rnd.y - nearest.y, rnd.x - nearest.x)
        new_x = nearest.x + RRT_STEP_LEN * math.cos(theta)
        new_y = nearest.y + RRT_STEP_LEN * math.sin(theta)

        if not self.check_collision_and_sense(new_x, new_y):
            new_node = self.Node(new_x, new_y)
            new_node.parent = nearest
            tree_nodes.append(new_node)
            self.last_added_node = new_node
            return new_node
        return None

    def try_connect(self, new_node, current_tree_id):
        target_tree_ids = []
        if current_tree_id == 0:
            target_tree_ids = [i+1 for i, v in enumerate(self.goals_status) if not v]
        else:
            if not self.goals_status[current_tree_id - 1]:
                target_tree_ids = [0]
        
        for tid in target_tree_ids:
            target_nodes = self.trees[tid]
            dists = [(node.x-new_node.x)**2 + (node.y-new_node.y)**2 for node in target_nodes]
            min_d = min(dists)
            
            if min_d <= (RRT_STEP_LEN * RRT_STEP_LEN):
                nearest_target = target_nodes[dists.index(min_d)]
                goal_idx = (tid - 1) if tid != 0 else (current_tree_id - 1)
                
                path = []
                start_node_ref = new_node if current_tree_id == 0 else nearest_target
                goal_node_ref = nearest_target if current_tree_id == 0 else new_node
                
                curr = start_node_ref
                while curr:
                    path.append(np.array([curr.x, curr.y]))
                    curr = curr.parent
                path = path[::-1] 
                
                curr = goal_node_ref
                while curr:
                    path.append(np.array([curr.x, curr.y]))
                    curr = curr.parent
                
                self.found_paths[goal_idx] = path
                self.goals_status[goal_idx] = True

    def step(self):
        if all(self.goals_status):
            if not self.tsp_path_calculated:
                self.solve_tsp_and_smooth()
            return
            
        active_trees = [0] + [i+1 for i, v in enumerate(self.goals_status) if not v]
        tree_to_grow = random.choice(active_trees)
        new_node = self.grow_tree(tree_to_grow)
        if new_node:
            self.try_connect(new_node, tree_to_grow)

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length

    def get_rrt_path_between(self, idx_a, idx_b):
        path_a = [self.start_pos] if idx_a == -1 else self.found_paths[idx_a]
        path_b = [self.start_pos] if idx_b == -1 else self.found_paths[idx_b]
        
        common_len = 0
        min_len = min(len(path_a), len(path_b))
        for k in range(min_len):
            if np.linalg.norm(path_a[k] - path_b[k]) < 1e-3:
                common_len = k + 1
            else: break
        
        segment_1 = path_a[common_len-1:][::-1]
        segment_2 = path_b[common_len:]
        return list(segment_1) + list(segment_2)

    # --- HÀM MỚI: PATH SMOOTHING (LÀM MƯỢT) ---
    def smooth_path(self, path):
        """
        Cắt tỉa các đỉnh thừa.
        Nếu điểm A nhìn thấy điểm C thì bỏ qua B.
        """
        if len(path) < 3: return path
        
        new_path = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Nhìn xa nhất có thể từ điểm hiện tại
            # Tìm điểm xa nhất (gần cuối list) mà current_idx có thể nối thẳng tới
            next_idx = current_idx + 1
            best_idx = next_idx
            
            # Duyệt ngược từ cuối về
            for i in range(len(path)-1, current_idx, -1):
                # Kiểm tra va chạm đoạn thẳng
                if not check_line_collision(path[current_idx], path[i], self.outer, self.holes):
                    best_idx = i
                    break
            
            new_path.append(path[best_idx])
            current_idx = best_idx
            
        return new_path

    def solve_tsp_and_smooth(self):
        goal_indices = list(range(len(self.goals)))
        all_indices = [-1] + goal_indices
        best_paths = {}; costs = {}
        
        # 1. Tính toán mọi cặp đường đi
        for i in range(len(all_indices)):
            for j in range(i + 1, len(all_indices)):
                u, v = all_indices[i], all_indices[j]
                
                # Lấy đường RRT thô
                rrt_path = self.get_rrt_path_between(u, v)
                
                # Cố gắng làm mượt ngay lập tức đoạn này
                # (Để TSP có trọng số chính xác hơn là trọng số zigzag)
                optimized_segment = self.smooth_path(rrt_path)
                
                seg_len = self.calculate_path_length(optimized_segment)
                
                best_paths[(u, v)] = optimized_segment
                best_paths[(v, u)] = optimized_segment[::-1]
                costs[(u, v)] = seg_len
                costs[(v, u)] = seg_len

        # 2. Giải TSP
        min_total_dist = float('inf')
        best_perm = None
        
        for perm in itertools.permutations(goal_indices):
            current_dist = costs[(-1, perm[0])]
            for i in range(len(perm) - 1):
                current_dist += costs[(perm[i], perm[i+1])]
            if current_dist < min_total_dist:
                min_total_dist = current_dist
                best_perm = perm

        self.tsp_sequence = best_perm
        
        # 3. Ghép đường đi thô (để so sánh) & đường đi mượt (để hiển thị đẹp)
        full_tour_smooth = []
        
        # Start -> First
        path = best_paths[(-1, best_perm[0])]
        full_tour_smooth.extend(path)
        
        for i in range(len(best_perm) - 1):
            u, v = best_perm[i], best_perm[i+1]
            path = best_paths[(u, v)]
            full_tour_smooth.extend(path[1:]) # Skip điểm đầu trùng
            
        self.smooth_tour_points = full_tour_smooth
        
        # Lấy bản RRT gốc (để vẽ màu đỏ làm nền)
        raw_tour = []
        p0 = self.get_rrt_path_between(-1, best_perm[0])
        raw_tour.extend(p0)
        for i in range(len(best_perm) - 1):
            p_next = self.get_rrt_path_between(best_perm[i], best_perm[i+1])
            raw_tour.extend(p_next[1:])
        self.raw_tour_points = raw_tour

        self.tsp_path_calculated = True

# --- MAIN ---
def main():
    global GROWTH_SPEED
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("RRT Zigzag vs Smoothed Path")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 14)
    font_big = pygame.font.SysFont("Arial", 20, bold=True)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC10_*")))
    dummy_outer = [(0,0), (WINDOW_SIZE,0), (WINDOW_SIZE,WINDOW_SIZE), (0,WINDOW_SIZE)]
    # Map có khe hẹp để test khả năng làm mượt
    dummy_holes = [
       [(200, 300), (400, 300), (400, 500), (200, 500)], # Khối vuông lớn
       [(500, 100), (550, 100), (550, 600), (500, 600)], # Tường dài
    ]

    planner = None
    outer_poly = []; holes = []
    scale = 1.0; start_pos = None; goal_positions = []
    current_map_idx = 0

    def reset_sim():
        nonlocal planner, outer_poly, holes, scale, start_pos, goal_positions
        use_dummy = False
        if map_folders and current_map_idx < len(map_folders):
            try:
                outer_poly, holes = load_data(map_folders[current_map_idx])
                if not outer_poly: use_dummy = True
            except: use_dummy = True
        else: use_dummy = True

        if use_dummy: outer_poly = dummy_outer; holes = dummy_holes
            
        xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
        mx = max(max(xs), max(ys))
        scale = (WINDOW_SIZE - 50) / mx
        min_x, min_y = min(xs), min(ys)
        start_pos = np.array([min_x+20.0, min_y+20.0])
        goal_positions = []
        for _ in range(NUM_GOALS):
            goal_positions.append(get_valid_random_pos(outer_poly, holes))
        bounds = [0, mx, 0, mx]
        planner = MultiTreeExplorer(start_pos, goal_positions, outer_poly, holes, bounds)

    reset_sim()
    def to_scr(pos): return int(pos[0]*scale)+25, int(WINDOW_SIZE - pos[1]*scale)-25

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n: 
                    if map_folders: current_map_idx = (current_map_idx + 1) % len(map_folders); reset_sim()
                elif event.key == pygame.K_p: 
                    if map_folders: current_map_idx = (current_map_idx - 1) % len(map_folders); reset_sim()
                elif event.key == pygame.K_r: reset_sim()

        if planner:
            for _ in range(GROWTH_SPEED): planner.step()

        screen.fill(WHITE)

        # 1. Vẽ Map & Sensing
        for i, h in enumerate(holes):
            scr_h = [to_scr(p) for p in h]
            if planner and planner.discovered_holes[i]:
                pygame.draw.polygon(screen, OBSTACLE_LIT, scr_h)
            else:
                pygame.draw.polygon(screen, OBSTACLE_HIDDEN, scr_h)
                pygame.draw.polygon(screen, OBSTACLE_OUTLINE, scr_h, 1)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)

        if planner:
            # 2. Vẽ Cây RRT (Mờ)
            for tid, nodes in planner.trees.items():
                color = TREE_PALETTE[tid % len(TREE_PALETTE)]
                for node in nodes:
                    if node.parent:
                        pygame.draw.line(screen, color, to_scr((node.parent.x, node.parent.y)), to_scr((node.x, node.y)), 1)

            # 3. Vẽ Graph Edges (Kết nối thô giữa các cây)
            if not planner.tsp_path_calculated:
                for idx, path in planner.found_paths.items():
                    if len(path) > 1:
                        scr_path = [to_scr(p) for p in path]
                        pygame.draw.lines(screen, GRAPH_EDGE_COLOR, False, scr_path, 2)

            # 4. FINAL RESULT
            if planner.tsp_path_calculated:
                # A. Đường đỏ: RRT gốc (Zigzag)
                if len(planner.raw_tour_points) > 1:
                    scr_raw = [to_scr(p) for p in planner.raw_tour_points]
                    pygame.draw.lines(screen, FINAL_PATH_COLOR, False, scr_raw, 2)
                
                # B. Đường Xanh Lá: Đã tối ưu (Thẳng tắp)
                if len(planner.smooth_tour_points) > 1:
                    scr_smooth = [to_scr(p) for p in planner.smooth_tour_points]
                    # Vẽ viền đen cho nổi
                    pygame.draw.lines(screen, BLACK, False, scr_smooth, 8)
                    # Vẽ lõi xanh
                    pygame.draw.lines(screen, SMOOTH_PATH_COLOR, False, scr_smooth, 4)
                    
                    # Vẽ các điểm chốt (Vertices) của đường mới
                    for p in scr_smooth:
                        pygame.draw.circle(screen, BLACK, p, 3)

                # Order Numbers
                for order, goal_idx in enumerate(planner.tsp_sequence):
                    pos = goal_positions[goal_idx]
                    txt_surf = font_big.render(str(order + 1), True, BLACK)
                    txt_rect = txt_surf.get_rect(center=to_scr(pos))
                    pygame.draw.rect(screen, WHITE, txt_rect.inflate(4,4))
                    screen.blit(txt_surf, txt_rect)

        # Start/Goals
        pygame.draw.circle(screen, TREE_PALETTE[0], to_scr(start_pos), 8)
        for i, g in enumerate(goal_positions):
            color = TREE_PALETTE[(i+1) % len(TREE_PALETTE)]
            pygame.draw.circle(screen, color, to_scr(g), 8)

        status = "Building..."
        if planner and planner.tsp_path_calculated: status = "RED: Raw RRT | GREEN: Optimized Path"
        screen.blit(font.render(status, True, BLACK), (10, 10))
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()