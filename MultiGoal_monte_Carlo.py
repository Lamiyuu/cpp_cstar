import pygame
import numpy as np
import math
import sys
import os
import glob
import time
import random
import csv
import datetime

# --- CẤU HÌNH ---
WINDOW_SIZE = 900
FPS = 60
DATASET_DIR = "AC300"
RESULT_DIR = "Results_Optimized"

NUM_GOALS = 5  
AUTO_NEXT = False 

# MÀU SẮC
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)          
GHOST_GRAY = (240, 240, 240) 
GREEN = (0, 180, 0)          
BLUE = (0, 0, 255)           
CYAN = (0, 200, 200)         
YELLOW = (255, 200, 0)
MAGENTA = (255, 0, 255)      # MÀU ĐƯỜNG TỐI ƯU

# THÔNG SỐ
MCPP_EPSILON = 5.0
MCPP_C = 1.414
MCPP_ITER = 200         
MCPP_DEPTH = 15

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

# --- THUẬT TOÁN TỐI ƯU HÓA (SEGMENTED PRUNING) ---
def check_line_collision(p1, p2, outer, holes):
    d = dist(p1, p2)
    steps = int(d / 1.0) + 1 
    for i in range(steps + 1):
        t = i / steps
        pt = p1 + (p2 - p1) * t
        if not point_in_polygon(pt, outer): return True
        for h in holes:
            if point_in_polygon(pt, h): return True
    return False

def prune_segment(path_segment, outer, holes):
    """Cắt tỉa 1 đoạn đường (Start -> Goal X)"""
    if len(path_segment) < 3: return path_segment
    
    optimized = [path_segment[0]]
    current_idx = 0
    
    while current_idx < len(path_segment) - 1:
        next_idx = current_idx + 1
        # Tìm điểm xa nhất trong đoạn này mà có thể nối thẳng được
        for i in range(len(path_segment) - 1, current_idx + 1, -1):
            if not check_line_collision(path_segment[current_idx], path_segment[i], outer, holes):
                next_idx = i
                break
        
        optimized.append(path_segment[next_idx])
        current_idx = next_idx
    return optimized

def calculate_total_length(segments):
    total = 0
    for seg in segments:
        for i in range(len(seg)-1):
            total += dist(seg[i], seg[i+1])
    return total

# --- MCPP PLANNER ---
class MCPP_Planner:
    class VNode:
        def __init__(self, state): self.state = state; self.N=0; self.children={}
    class QNode:
        def __init__(self, parent, action): self.parent=parent; self.action=action; self.n=0; self.Q=0.0; self.child_v=None; self.cum_r=0.0

    def __init__(self, start, goal, outer, known_holes):
        self.root = self.VNode(start); self.goal = goal; self.outer = outer; self.known_holes = known_holes

    def is_valid(self, pos):
        if not point_in_polygon(pos, self.outer): return False
        for h in self.known_holes:
            if point_in_polygon(pos, h): return False
        return True

    def get_action_ucb(self, v):
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            s = q.Q + MCPP_C * math.sqrt(math.log(max(1, v.N))/q.n)
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        ang = random.uniform(0, 2*math.pi); r = random.uniform(1.0, MCPP_EPSILON)
        nxt = v.state + np.array([r*math.cos(ang), r*math.sin(ang)])
        if not self.is_valid(nxt): return None
        act = tuple(nxt)
        if act not in v.children: v.children[act] = self.QNode(v, act)
        return act

    def sim_v(self, v, d):
        if d==0 or dist(v.state, self.goal)<2.0: return -dist(v.state, self.goal)
        if len(v.children)<8:
            act = self.expand(v)
            if act: return self.sim_q(v.children[act], d)
        if not v.children: return -dist(v.state, self.goal)
        _, q = self.get_action_ucb(v)
        return self.sim_q(q, d)

    def sim_q(self, q, d):
        if not q.child_v: q.child_v = self.VNode(np.array(q.action)); return -dist(q.child_v.state, self.goal)
        r = -dist(q.parent.state, np.array(q.action)) + self.sim_v(q.child_v, d-1)
        q.n+=1; q.cum_r+=r; q.Q=q.cum_r/q.n; q.parent.N+=1
        return r

    def search(self):
        for _ in range(MCPP_ITER): self.sim_v(self.root, MCPP_DEPTH)
        if not self.root.children: return None
        best = max(self.root.children.items(), key=lambda i:i[1].Q)[0]
        return np.array(best)

# --- MAIN LOOP ---
def main():
    global SCALE 
    pygame.init()
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    csv_file = os.path.join(RESULT_DIR, f"SegmentedOptim_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerow(["Map", "Original Len", "Optimized Len", "Improvement (%)"])

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("MCPP: Multi-Goal Segmented Optimization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)
    font_big = pygame.font.SysFont("Consolas", 24)

    map_folders = sorted(glob.glob(os.path.join(DATASET_DIR, "AC10_*")))
    if not map_folders: return

    # --- STATE ---
    current_map_idx = 0
    outer_poly = []; holes = []
    scale = 1.0
    
    current_pos = np.array([0.,0.])
    unvisited_goals = []
    current_goal = None
    
    known_holes = []
    
    # QUẢN LÝ ĐƯỜNG ĐI THEO CHẶNG (SEGMENTS)
    # raw_segments = [ [p1, p2... goal1], [goal1, p3... goal2], ... ]
    raw_segments = [] 
    current_segment = [] 
    
    # Kết quả tối ưu
    optimized_segments = [] 
    
    planner_vis = None
    finished = False
    
    # Hàm tìm goal gần nhất (Smart Greedy)
    def check_intersection(p1, p2, poly):
        steps=5
        for i in range(1, steps):
            t=i/steps; pt=p1+(p2-p1)*t
            if point_in_polygon(pt, poly): return True
        return False

    def get_smart_goal(pos, goals, known):
        if not goals: return None
        bst=None; mn=float('inf')
        for g in goals:
            d = dist(pos, g); pen=0
            for h in known:
                if check_intersection(pos, g, h): pen+=d*3.0; break
            score=d+pen
            if score<mn: mn=score; bst=g
        return bst

    def reset_sim():
        nonlocal outer_poly, holes, scale, current_pos, unvisited_goals, current_goal
        nonlocal known_holes, raw_segments, current_segment, finished, planner_vis, optimized_segments
        
        folder = map_folders[current_map_idx]
        print(f"Loading: {folder}")
        outer_poly, holes = load_data(folder)
        
        if outer_poly:
            xs = [p[0] for p in outer_poly]; ys = [p[1] for p in outer_poly]
            mx = max(max(xs), max(ys)); scale = (WINDOW_SIZE - 80) / mx
            
            min_x, min_y = min(xs), min(ys)
            current_pos = np.array([min_x+2.0, min_y+2.0])
            
            unvisited_goals = []
            for _ in range(NUM_GOALS):
                unvisited_goals.append(get_valid_random_pos(outer_poly, holes))
            current_goal = get_smart_goal(current_pos, unvisited_goals, [])
        else: scale = 1.0
        
        known_holes = []
        raw_segments = []
        current_segment = [current_pos] # Start point của segment đầu tiên
        optimized_segments = []
        
        planner_vis = None
        finished = False

    reset_sim()

    def to_scr(pos):
        return int(pos[0]*scale)+40, int(WINDOW_SIZE - (pos[1]*scale))-40

    def draw_mcpp_tree(v, p_pos=None):
        if not v: return
        c = to_scr(v.state)
        if p_pos: pygame.draw.line(screen, GREEN, p_pos, c, 1)
        if v.N > 2:
            for _, q in v.children.items():
                if q.child_v: draw_mcpp_tree(q.child_v, c)

    running = True
    while running:
        clock.tick(FPS)
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

        # LOGIC
        if not finished and outer_poly and current_goal is not None:
            if dist(current_pos, current_goal) < 3.0:
                print(f"Reached Goal at {current_goal}")
                
                # --- CHỐT CHẶNG HIỆN TẠI ---
                # Thêm điểm đích vào cuối segment để đảm bảo kín
                current_segment.append(current_goal)
                raw_segments.append(current_segment)
                
                # Bắt đầu chặng mới từ vị trí hiện tại
                current_segment = [current_goal]
                
                # Remove reached goal
                for i, g in enumerate(unvisited_goals):
                    if np.array_equal(g, current_goal): unvisited_goals.pop(i); break
                
                if unvisited_goals:
                    current_goal = get_smart_goal(current_pos, unvisited_goals, known_holes)
                else:
                    # FINISH -> RUN OPTIMIZATION PER SEGMENT
                    finished = True
                    current_goal = None
                    
                    print("Optimizing Segments...", end="")
                    optimized_segments = []
                    # Tối ưu từng chặng một, giữ nguyên điểm nối (Goal)
                    for seg in raw_segments:
                        opt_seg = prune_segment(seg, outer_poly, holes) # Dùng holes thật để tối ưu
                        optimized_segments.append(opt_seg)
                    print(" Done!")
                    
                    # Save results
                    len_old = calculate_total_length(raw_segments)
                    len_new = calculate_total_length(optimized_segments)
                    improv = (len_old - len_new) / len_old * 100 if len_old > 0 else 0
                    
                    with open(csv_file, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            os.path.basename(map_folders[current_map_idx]),
                            f"{len_old:.2f}", f"{len_new:.2f}", f"{improv:.2f}"
                        ])
                    
                    if AUTO_NEXT: 
                        current_map_idx=(current_map_idx+1)%len(map_folders); reset_sim()
            
            # Planning
            if current_goal is not None:
                planner = MCPP_Planner(current_pos, current_goal, outer_poly, known_holes)
                next_pos = planner.search()
                planner_vis = planner 
                
                if next_pos is not None:
                    col = False; hit = None
                    if not point_in_polygon(next_pos, outer_poly): col = True
                    if not col:
                        for h in holes:
                            if point_in_polygon(next_pos, h): col=True; hit=h; break
                    
                    if col:
                        if hit and hit not in known_holes: known_holes.append(hit)
                        if len(current_segment)>1:
                            back = current_segment[-2]-current_pos
                            if np.linalg.norm(back)>0: current_pos += back/np.linalg.norm(back)*1.5
                    else:
                        current_pos = next_pos
                        current_segment.append(current_pos) # Lưu vào chặng hiện tại

        # RENDERING
        screen.fill(WHITE)
        if outer_poly: pygame.draw.polygon(screen, (50,50,50), [to_scr(p) for p in outer_poly], 2)
        for h in holes:
            pygame.draw.polygon(screen, RED if h in known_holes else GHOST_GRAY, [to_scr(p) for p in h])

        if planner_vis and not finished: draw_mcpp_tree(planner_vis.root)

        for g in unvisited_goals:
            col = BLUE if np.array_equal(g, current_goal) else CYAN
            pygame.draw.circle(screen, col, to_scr(g), 8)
        
        if current_goal is not None:
            pygame.draw.line(screen, YELLOW, to_scr(current_pos), to_scr(current_goal), 2)

        # DRAW PATHS (THEO SEGMENT)
        # 1. Đường thô (Đen)
        # Vẽ các đoạn đã xong
        for seg in raw_segments:
            if len(seg)>1: pygame.draw.lines(screen, (150, 150, 150) if finished else BLACK, False, [to_scr(p) for p in seg], 2)
        # Vẽ đoạn đang đi
        if len(current_segment)>1:
            pygame.draw.lines(screen, BLACK, False, [to_scr(p) for p in current_segment], 3)
        
        # 2. ĐƯỜNG TỐI ƯU (Tím Đậm - Đảm bảo đi qua các Goal)
        if finished and optimized_segments:
            for seg in optimized_segments:
                if len(seg)>1:
                    pygame.draw.lines(screen, MAGENTA, False, [to_scr(p) for p in seg], 4)
                    # Vẽ điểm chốt (Start/Goal của chặng)
                    pygame.draw.circle(screen, MAGENTA, to_scr(seg[0]), 5)
                    pygame.draw.circle(screen, MAGENTA, to_scr(seg[-1]), 5)

        pygame.draw.circle(screen, BLACK, to_scr(current_pos), 6)

        # UI
        mname = os.path.basename(map_folders[current_map_idx])
        screen.blit(font_big.render(f"Map: {mname}", True, BLACK), (10, 10))
        screen.blit(font.render(f"Goals Left: {len(unvisited_goals)}/{NUM_GOALS}", True, BLUE), (10, 40))
        
        if finished:
            l_old = calculate_total_length(raw_segments)
            l_new = calculate_total_length(optimized_segments)
            imp = (l_old - l_new) / l_old * 100 if l_old > 0 else 0
            screen.blit(font.render(f"Original Dist: {l_old:.2f} m", True, BLACK), (10, 70))
            screen.blit(font.render(f"Optimized Dist: {l_new:.2f} m", True, MAGENTA), (10, 90))
            screen.blit(font.render(f"Improvement: {imp:.1f}%", True, GREEN), (10, 110))
            screen.blit(font.render("FINISHED - PRESS 'N' FOR NEXT", True, RED), (10, 140))
        else:
            screen.blit(font.render("RUNNING...", True, GREEN), (10, 70))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()