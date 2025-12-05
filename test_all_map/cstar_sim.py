import numpy as np
import networkx as nx
from scipy.spatial import distance
from collections import deque
import random
import pygame
import sys
import os
from PIL import Image, ImageDraw

# --- CẤU HÌNH THAM SỐ ---
MAP_SIZE = 100          # Scale 100x100 chuẩn AC300
SAMPLE_RES_W = 2        # W=2
SENSOR_RANGE = 10       # Tầm nhìn 10
OBSTACLE_COLOR = 1
FREE_COLOR = 0
UNKNOWN_COLOR = -1

# Pygame
CELL_SIZE = 8           # Giảm size để vẽ vừa màn hình map 100
SCREEN_WIDTH = MAP_SIZE * CELL_SIZE
SCREEN_HEIGHT = MAP_SIZE * CELL_SIZE + 60
FPS = 60

# Màu sắc
COLOR_BG = (220, 220, 220)
COLOR_FREE = (255, 255, 255)
COLOR_OBS = (20, 20, 20)
COLOR_NODE_OPEN = (0, 200, 0)
COLOR_NODE_CLOSED = (200, 0, 0)
COLOR_NODE_GOAL = (255, 200, 0)
COLOR_EDGE = (50, 50, 255)
COLOR_ROBOT = (0, 0, 255)
COLOR_PRUNED_NODE = (128, 0, 128) # Tím
COLOR_PRUNED_EDGE = (180, 180, 180)

# --- CLASS MÔI TRƯỜNG ---
class Environment:
    def __init__(self, size, map_folder=None):
        self.size = size
        self.grid = np.full((size, size), UNKNOWN_COLOR)
        self.real_map = np.zeros((size, size))
        
        if map_folder and os.path.exists(map_folder):
            self._load_ac300_map(map_folder)
        else:
            self._add_default_obstacles()

    def _read_coords(self, filepath):
        coords = []
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try: coords.append((float(parts[0]), float(parts[1])))
                        except: pass
        return coords

    def _load_ac300_map(self, folder):
        outer = self._read_coords(os.path.join(folder, "outer_polygon"))
        holes = self._read_coords(os.path.join(folder, "holes"))
        img = Image.new('L', (self.size, self.size), 0)
        draw = ImageDraw.Draw(img)
        sx = self.size / 100.0
        sy = self.size / 100.0
        if holes:
            poly = [(x*sx, self.size - y*sy) for x, y in holes]
            if len(poly) > 2: draw.polygon(poly, fill=1, outline=1)
        raw_data = np.array(img)
        for x in range(self.size):
            for y in range(self.size):
                val = raw_data[self.size - 1 - y, x]
                if val > 0: self.real_map[x, y] = OBSTACLE_COLOR
                else: self.real_map[x, y] = FREE_COLOR

    def _add_default_obstacles(self):
        self.real_map[15:35, 15:18] = 1
        self.real_map[15:35, 25:28] = 1
        self.real_map[15:18, 15:28] = 1
        self.real_map[40:48, 40:48] = 1 
        self.real_map[5:10, 30:40] = 1

    def sense(self, x, y, radius):
        x_min, x_max = max(0, int(x - radius)), min(self.size, int(x + radius))
        y_min, y_max = max(0, int(y - radius)), min(self.size, int(y + radius))
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if self.grid[i, j] == UNKNOWN_COLOR:
                    self.grid[i, j] = self.real_map[i, j]

    def is_valid(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.real_map[int(x), int(y)] != OBSTACLE_COLOR
        return False
    
    def get_safe_start(self):
        if self.real_map[5, 5] == FREE_COLOR: return (5, 5)
        frees = np.argwhere(self.real_map == FREE_COLOR)
        return (frees[0][0], frees[0][1]) if len(frees) > 0 else (0,0)

# --- CLASS RCG ---
class RCG:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {}
        self.pruned_nodes = []
        self.pruned_edges = []
        self.node_counter = 0
        self.current_node_id = None

    def add_node(self, x, y, status='Open'):
        for nid, data in self.nodes.items():
            if distance.euclidean((x, y), data['pos']) < 0.5: return nid
        nid = self.node_counter
        self.nodes[nid] = {'pos': (x, y), 'status': status}
        self.graph.add_node(nid, pos=(x, y))
        self.node_counter += 1
        return nid

    def expand(self, environment, robot_pos):
        x_min = max(0, int(robot_pos[0] - SENSOR_RANGE))
        x_max = min(MAP_SIZE, int(robot_pos[0] + SENSOR_RANGE))
        y_min = max(0, int(robot_pos[1] - SENSOR_RANGE))
        y_max = min(MAP_SIZE, int(robot_pos[1] + SENSOR_RANGE))
        
        new_nodes = []
        for x in range(x_min, x_max):
            if x % SAMPLE_RES_W == 0:
                for y in range(y_min, y_max):
                    if environment.grid[x, y] == FREE_COLOR:
                        nid = self.add_node(x, y)
                        if nid not in new_nodes: new_nodes.append(nid)
        
        all_ids = list(self.nodes.keys())
        # Tăng bán kính để bắt chéo (Tránh kẹt)
        radius = np.sqrt(2) * SAMPLE_RES_W + 0.5 
        
        for nid in new_nodes:
            p1 = self.nodes[nid]['pos']
            for other in all_ids:
                if nid == other: continue
                p2 = self.nodes[other]['pos']
                dist = distance.euclidean(p1, p2)
                if dist <= radius:
                    mx, my = int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)
                    if environment.is_valid(mx, my):
                        if not self.graph.has_edge(nid, other):
                            self.graph.add_edge(nid, other, weight=dist)

    def prune_rcg(self, environment):
        """Chiến lược Cắt tỉa Thẳng hàng (Collinear Pruning)"""
        all_nodes = list(self.nodes.keys())
        
        for nid in all_nodes:
            if nid not in self.nodes or nid == self.current_node_id:
                continue
            
            # [FIX] Gọi hàm _is_essential (Giờ nó đã là method ngang cấp)
            if self._is_essential(nid, environment):
                continue

            try: neighbors = list(self.graph.neighbors(nid))
            except: continue
            
            # Điều kiện xóa: Nút bậc 2 (Nút trung gian)
            if len(neighbors) == 2:
                n1, n2 = neighbors[0], neighbors[1]
                if n1 not in self.nodes or n2 not in self.nodes: continue
                
                p1 = self.nodes[n1]['pos']
                p  = self.nodes[nid]['pos']
                p2 = self.nodes[n2]['pos']
                
                # Kiểm tra thẳng hàng (Cross Product ~ 0)
                cross_product = (p2[1] - p1[1]) * (p[0] - p1[0]) - (p2[0] - p1[0]) * (p[1] - p1[1])
                
                if abs(cross_product) < 0.1:
                    # Kiểm tra an toàn đường nối tắt
                    mid_x = int((p1[0]+p2[0])/2)
                    mid_y = int((p1[1]+p2[1])/2)
                    
                    if environment.is_valid(mid_x, mid_y):
                        # [VISUALIZATION]
                        self.pruned_nodes.append(p)
                        self.pruned_edges.append((p1, p))
                        self.pruned_edges.append((p, p2))

                        # Hợp nhất cạnh
                        w1 = self.graph[n1][nid]['weight']
                        w2 = self.graph[nid][n2]['weight']
                        self.graph.add_edge(n1, n2, weight=w1+w2)
                        
                        # Xóa nút giữa
                        self.graph.remove_node(nid)
                        del self.nodes[nid]

    # [FIX] Đã đưa hàm này ra ngoài prune_rcg (Thụt lề đúng)
    def _is_essential(self, nid, environment):
        return self._is_adjacent_to_unknown(nid, environment)

    def _is_adjacent_to_unknown(self, nid, environment):
        p = self.nodes[nid]['pos']; px, py = int(p[0]), int(p[1])
        for dx, dy in [(-1,0),(1,0),(0,1),(0,-1)]:
            nx, ny = px+dx, py+dy
            if 0<=nx<MAP_SIZE and 0<=ny<MAP_SIZE:
                if environment.grid[nx, ny] == UNKNOWN_COLOR: return True
        return False

    def select_zigzag_goal(self, current_id):
        if current_id is None: return None
        neighbors = list(self.graph.neighbors(current_id))
        open_neighbors = [n for n in neighbors if self.nodes[n]['status']=='Open']
        if not open_neighbors: return None
        
        c_pos = self.nodes[current_id]['pos']
        candidates = {'Left':[], 'Up':[], 'Down':[], 'Right':[]}
        EPS = 0.1
        
        for nid in open_neighbors:
            n_pos = self.nodes[nid]['pos']
            dx = n_pos[0] - c_pos[0]
            dy = n_pos[1] - c_pos[1]
            
            if dx < -EPS: candidates['Left'].append(nid)
            elif dx > EPS: candidates['Right'].append(nid)
            else:
                if dy > EPS: candidates['Up'].append(nid)
                elif dy < -EPS: candidates['Down'].append(nid)
        
        priority = ['Left', 'Up', 'Down', 'Right']
        for d in priority:
            if candidates[d]: return random.choice(candidates[d])
        return None

    def detect_holes(self, environment, current_id, next_goal_id):
        neighbors = list(self.graph.neighbors(current_id))
        candidates = [n for n in neighbors if self.nodes[n]['status']=='Open' and n!=next_goal_id]
        
        visited_global = set()
        holes = []
        MAX_HOLE_SIZE = 40 
        
        for start in candidates:
            if start in visited_global: continue
            
            cluster = []
            queue = deque([start])
            visited_global.add(start)
            visited_local = {start}
            is_hole = True
            
            while queue:
                curr = queue.popleft()
                cluster.append(curr)
                
                if len(cluster) > MAX_HOLE_SIZE: is_hole = False
                
                p = self.nodes[curr]['pos']
                for dx, dy in [(-1,0),(1,0),(0,1),(0,-1)]:
                    nx, ny = int(p[0])+dx, int(p[1])+dy
                    if 0<=nx<MAP_SIZE and 0<=ny<MAP_SIZE:
                        if environment.grid[nx, ny] == UNKNOWN_COLOR: is_hole = False
                
                for nb in self.graph.neighbors(curr):
                    if nb == next_goal_id: continue
                    if self.nodes[nb]['status'] == 'Closed': continue
                    
                    if nb not in visited_local:
                        visited_local.add(nb)
                        visited_global.add(nb)
                        queue.append(nb)
                        
            if is_hole and 0 < len(cluster) <= MAX_HOLE_SIZE:
                holes.append(cluster)
        return holes

    def solve_tsp(self, cluster_nodes, current_id, next_goal_id=None):
        if not cluster_nodes: return []
        
        n_start = current_id
        n_end = n_start 
        if next_goal_id is not None and next_goal_id not in cluster_nodes:
             n_end = next_goal_id

        targets = [n for n in cluster_nodes if n != n_start]
        if n_end != n_start and n_end not in targets:
            targets.append(n_end)
            
        ordered_stops = [n_start]
        curr = n_start
        
        while targets:
            best_next = None
            min_dist = float('inf')
            
            for t in targets:
                try:
                    d = nx.shortest_path_length(self.graph, curr, t, weight='weight')
                    if d < min_dist:
                        min_dist = d
                        best_next = t
                except nx.NetworkXNoPath:
                    continue
            
            if best_next is not None:
                ordered_stops.append(best_next)
                targets.remove(best_next)
                curr = best_next
            else:
                # Nếu kẹt (cô lập), thử dùng Euclid để nhảy
                if targets:
                    fb = min(targets, key=lambda n: distance.euclidean(self.nodes[curr]['pos'], self.nodes[n]['pos']))
                    ordered_stops.append(fb)
                    targets.remove(fb)
                    curr = fb
                else: break
                
        full_path = []
        for i in range(len(ordered_stops) - 1):
            u = ordered_stops[i]
            v = ordered_stops[i+1]
            try:
                segment = nx.shortest_path(self.graph, u, v, weight='weight')
                if i == 0: full_path.extend(segment)
                else: full_path.extend(segment[1:])
            except nx.NetworkXNoPath: pass
            
        return full_path

    def get_nearest_retreat_node(self, current_id):
        try:
            lengths = nx.single_source_dijkstra_path_length(self.graph, current_id, weight='weight', cutoff=100)
            best_node = None; min_dist = float('inf')
            for nid, dist in lengths.items():
                if self.nodes[nid]['status'] == 'Open':
                    is_retreat = False
                    for nb in self.graph.neighbors(nid):
                        if self.nodes[nb]['status'] == 'Closed': is_retreat = True; break
                    if is_retreat and dist < min_dist: min_dist = dist; best_node = nid
            return best_node
        except: return None

# --- MAIN ---
def to_screen(x, y):
    return int(x * CELL_SIZE), int((MAP_SIZE - y) * CELL_SIZE)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("C* Algorithm (Fixed Indentation)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    
    pruned_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

    # [CẤU HÌNH MAP]
    TARGET_MAP = "AC300/AC10_0000"
    env = Environment(MAP_SIZE, map_folder=TARGET_MAP)
    rcg = RCG()
    
    start_pos = env.get_safe_start()
    curr_node = rcg.add_node(*start_pos, status='Closed')
    rcg.current_node_id = curr_node
    
    path_queue = [] 
    step = 0
    mode = "Start"
    running = True
    paused = False
    show_pruned = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_v: show_pruned = not show_pruned

        if not paused:
            c_pos = rcg.nodes[rcg.current_node_id]['pos']
            env.sense(c_pos[0], c_pos[1], SENSOR_RANGE)
            
            rcg.expand(env, c_pos)
            
            # Chỉ cắt tỉa khi đi ZigZag
            if not path_queue:
                rcg.prune_rcg(env)

            # Update Visual
            for p1, p2 in rcg.pruned_edges:
                pygame.draw.line(pruned_surface, COLOR_PRUNED_EDGE, to_screen(*p1), to_screen(*p2), 1)
            for p in rcg.pruned_nodes:
                pygame.draw.circle(pruned_surface, COLOR_PRUNED_NODE, to_screen(*p), 2)
            rcg.pruned_nodes.clear(); rcg.pruned_edges.clear()

            next_node = None
            mode = "Zig-Zag"

            if path_queue:
                mode = "Following Path"
                cand = path_queue.pop(0)
                if cand in rcg.nodes: next_node = cand
                else: path_queue.clear()
            else:
                tentative_next = rcg.select_zigzag_goal(rcg.current_node_id)
                holes = rcg.detect_holes(env, rcg.current_node_id, tentative_next)
                
                if holes:
                    mode = "Hole -> TSP"
                    tsp_path = rcg.solve_tsp(holes[0], rcg.current_node_id, tentative_next)
                    if tsp_path:
                        path_queue.extend(tsp_path)
                        if path_queue: 
                            cand = path_queue.pop(0)
                            if cand in rcg.nodes: next_node = cand
                else:
                    next_node = tentative_next
                    if next_node is None:
                        mode = "Dead-end Escape"
                        retreat = rcg.get_nearest_retreat_node(rcg.current_node_id)
                        if retreat:
                            try:
                                p = nx.shortest_path(rcg.graph, rcg.current_node_id, retreat, weight='weight')
                                if len(p) > 1: path_queue.extend(p[1:]); next_node = path_queue.pop(0)
                            except: mode = "No Path"; paused = True
                        else: mode = "COMPLETE"; paused = True

            if next_node is not None and next_node in rcg.nodes:
                rcg.nodes[next_node]['status'] = 'Closed'
                rcg.current_node_id = next_node
                step += 1

        # Draw
        screen.fill(COLOR_BG)
        
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                if env.grid[x, y] != UNKNOWN_COLOR:
                    col = COLOR_FREE if env.grid[x, y] == FREE_COLOR else COLOR_OBS
                    pygame.draw.rect(screen, col, (*to_screen(x, y+1), CELL_SIZE, CELL_SIZE))
        
        if show_pruned:
            screen.blit(pruned_surface, (0,0))

        for u, v in rcg.graph.edges():
            if u in rcg.nodes and v in rcg.nodes:
                p1, p2 = rcg.nodes[u]['pos'], rcg.nodes[v]['pos']
                pygame.draw.line(screen, COLOR_EDGE, to_screen(*p1), to_screen(*p2), 2)
        
        for nid, data in rcg.nodes.items():
            col = COLOR_NODE_CLOSED if data['status']=='Closed' else COLOR_NODE_OPEN
            if nid == next_node: col = COLOR_NODE_GOAL
            pygame.draw.circle(screen, col, to_screen(*data['pos']), 3)
        
        if rcg.current_node_id in rcg.nodes:
            curr_p = rcg.nodes[rcg.current_node_id]['pos']
            pygame.draw.circle(screen, COLOR_ROBOT, to_screen(*curr_p), 6)

        pygame.draw.rect(screen, (30, 30, 30), (0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 50))
        op_cnt = len([n for n in rcg.nodes if rcg.nodes[n]['status']=='Open'])
        info = f"Step: {step} | {mode} | Active: {len(rcg.nodes)} | 'V': Toggle Pruned"
        screen.blit(font.render(info, True, (255, 255, 255)), (10, SCREEN_HEIGHT - 35))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()