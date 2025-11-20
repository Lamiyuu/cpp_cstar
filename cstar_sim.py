import numpy as np
import networkx as nx
from scipy.spatial import distance
from collections import deque
import random
import pygame
import sys

# [cite_start]--- CẤU HÌNH THAM SỐ [cite: 341, 709] ---
MAP_SIZE = 50          # Kích thước bản đồ 50x50m
SAMPLE_RES_W = 1       # Khoảng cách giữa các đường Lap (w)
SENSOR_RANGE = 15       # Bán kính cảm biến
OBSTACLE_COLOR = 1
FREE_COLOR = 0
UNKNOWN_COLOR = -1

# Cấu hình Hiển thị Pygame
CELL_SIZE = 14         # Tăng kích thước ô để dễ nhìn
SCREEN_WIDTH = MAP_SIZE * CELL_SIZE
SCREEN_HEIGHT = MAP_SIZE * CELL_SIZE + 60 
FPS = 60               

# --- BẢNG MÀU (TƯƠNG PHẢN CAO) ---
COLOR_BG = (220, 220, 220)      # Nền Xám sáng
COLOR_OBS = (20, 20, 20)        # Vật cản Đen
COLOR_FREE = (255, 255, 255)    # Vùng đã khám phá (Trắng)

# Màu Đồ thị Chính (Active)
COLOR_NODE_OPEN = (0, 200, 0)   # Xanh lá (Chưa đi)
COLOR_NODE_CLOSED = (200, 0, 0) # Đỏ (Đã đi)
COLOR_NODE_GOAL = (255, 200, 0) # Vàng (Đích)
COLOR_EDGE = (50, 50, 255)      # Xanh dương đậm (Cạnh nối)
COLOR_ROBOT = (0, 0, 255)       # Robot

# Màu Đồ thị Bị Cắt Tỉa (Pruned - Bóng ma)
COLOR_PRUNED_NODE = (148, 0, 211)   # Màu Tím Đậm (Dark Violet) - Dễ nhìn thấy "xác" nút
COLOR_PRUNED_EDGE = (200, 200, 200) # Màu Xám nhạt cho cạnh cũ

# --- LOGIC THUẬT TOÁN ---

class Environment:
    def __init__(self, size):
        self.size = size
        self.grid = np.full((size, size), UNKNOWN_COLOR)
        self.real_map = np.zeros((size, size)) 
        self._add_obstacles()

    def _add_obstacles(self):
        # Vật cản chữ U (Tạo Lỗ hổng)
        self.real_map[15:35, 15:18] = 1   
        self.real_map[15:35, 25:28] = 1   
        self.real_map[15:18, 15:28] = 1   
        # Vật cản chặn đường (Tạo Dead-end)
        self.real_map[40:48, 40:48] = 1 
        self.real_map[5:10, 30:40] = 1

    def sense(self, x, y, radius):
        """Mô phỏng cảm biến Lidar cập nhật bản đồ"""
        # Quét vùng vuông xung quanh robot để đơn giản hóa raycasting
        x_min, x_max = max(0, int(x - radius)), min(self.size, int(x + radius))
        y_min, y_max = max(0, int(y - radius)), min(self.size, int(y + radius))
        
        # Cập nhật vùng Unknown thành Free hoặc Obstacle dựa trên real_map
        scanned_area = []
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if self.grid[i, j] == UNKNOWN_COLOR:
                    self.grid[i, j] = self.real_map[i, j]
                    scanned_area.append((i, j))
        return scanned_area

    def is_valid(self, x, y):
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.real_map[int(x), int(y)] != OBSTACLE_COLOR
        return False

class RCG:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {} 
        # Lưu trữ lịch sử cắt tỉa để vẽ
        self.pruned_nodes = [] 
        self.pruned_edges = [] 
        
        self.node_counter = 0
        self.current_node_id = None

    def add_node(self, x, y, status='Open'):
        # Deduplication
        for nid, data in self.nodes.items():
            if distance.euclidean((x, y), data['pos']) < 0.5:
                return nid
        nid = self.node_counter
        self.nodes[nid] = {'pos': (x, y), 'status': status}
        self.graph.add_node(nid, pos=(x, y))
        self.node_counter += 1
        return nid

    def expand(self, environment, robot_pos):
        """
        Chiến lược Mở rộng RCG (Chuẩn Paper)
        Bán kính kết nối: sqrt(2) * w 
        """
        # 1. Xác định vùng quét
        x_scan_min = max(0, int(robot_pos[0] - SENSOR_RANGE))
        x_scan_max = min(MAP_SIZE, int(robot_pos[0] + SENSOR_RANGE))
        y_scan_min = max(0, int(robot_pos[1] - SENSOR_RANGE))
        y_scan_max = min(MAP_SIZE, int(robot_pos[1] + SENSOR_RANGE))
        
        new_nodes = []
        
        # 2. Tạo Node (Lấy mẫu)
        for x in range(x_scan_min, x_scan_max):
            # Chỉ lấy mẫu trên các đường Lap (cách nhau w)
            if x % SAMPLE_RES_W == 0: 
                for y in range(y_scan_min, y_scan_max):
                    if environment.grid[x, y] == FREE_COLOR:
                        nid = self.add_node(x, y)
                        if nid not in new_nodes:
                            new_nodes.append(nid)

        # 3. Kết nối Node (Expansion)
        all_ids = list(self.nodes.keys())
        
        # --- ĐỊNH NGHĨA BÁN KÍNH CHUẨN PAPER ---
        # Radius = sqrt(2) * w
        paper_radius = np.sqrt(2) * SAMPLE_RES_W
        
        for nid in new_nodes:
            pos1 = self.nodes[nid]['pos']
            for other_id in all_ids:
                if nid == other_id: continue
                pos2 = self.nodes[other_id]['pos']
                dist = distance.euclidean(pos1, pos2)
                
                # So sánh khoảng cách với bán kính chuẩn
                # Thêm 1e-9 là dung sai cực nhỏ cho lỗi số thực (floating point error)
                if dist <= paper_radius + 1e-9:
                    mid_x = int((pos1[0]+pos2[0])/2)
                    mid_y = int((pos1[1]+pos2[1])/2)
                    
                    if environment.is_valid(mid_x, mid_y):
                        if not self.graph.has_edge(nid, other_id):
                            self.graph.add_edge(nid, other_id, weight=dist)

    def prune_rcg(self, environment):
        """
        Chiến lược Cắt tỉa chuẩn Paper (Section III-B-3-b)
        Dựa trên Định nghĩa III.8 và Hình 7.
        """
        # Copy danh sách để duyệt an toàn
        all_nodes = list(self.nodes.keys())
        
        for nid in all_nodes:
            # Bỏ qua nếu nút đã xóa hoặc là nút robot đang đứng
            if nid not in self.nodes or nid == self.current_node_id:
                continue
            
            # Kiểm tra xem nút có phải là 'Essential' (Thiết yếu) không?
            # Nếu CÓ -> Giữ lại
            if self._is_essential(nid, environment):
                continue 

            # --- NẾU LÀ NÚT KHÔNG THIẾT YẾU (INESSENTIAL) ---
            # Thực hiện hành động cắt tỉa theo mô tả [cite: 428-429]:
            # "edges connecting it to the two adjacent nodes on its lap are merged"
            
            neighbors = list(self.graph.neighbors(nid))
            pos = self.nodes[nid]['pos']
            
            # Tìm 2 hàng xóm nằm trên CÙNG LAP (thẳng hàng dọc)
            vertical_neighbors = []
            for nb in neighbors:
                nb_pos = self.nodes[nb]['pos']
                # Kiểm tra cùng tọa độ X (sai số nhỏ)
                if abs(nb_pos[0] - pos[0]) < 0.1:
                    vertical_neighbors.append(nb)
            
            # Chỉ xóa được nếu nó nằm kẹp giữa 2 nút trên cùng 1 Lap
            if len(vertical_neighbors) == 2:
                n1, n2 = vertical_neighbors[0], vertical_neighbors[1]
                
                # [VISUALIZATION] Lưu vết để vẽ
                self.pruned_nodes.append(pos)
                self.pruned_edges.append((self.nodes[n1]['pos'], pos))
                self.pruned_edges.append((pos, self.nodes[n2]['pos']))

                # Hợp nhất cạnh (Merge Edge): Cộng dồn trọng số
                w1 = self.graph[n1][nid]['weight']
                w2 = self.graph[nid][n2]['weight']
                new_weight = w1 + w2
                
                # Tạo cạnh mới nối n1-n2
                self.graph.add_edge(n1, n2, weight=new_weight)
                
                # Xóa nút và các cạnh cũ
                self.graph.remove_node(nid)
                del self.nodes[nid]

    def _is_essential(self, nid, environment):
        """
        Kiểm tra Định nghĩa III.8 (Essential Node) 
        """
        # Điều kiện 1: Adjacent to Unknown area [cite: 384]
        if self._is_adjacent_to_unknown(nid, environment):
            return True
            
        # Điều kiện 2: End node of a lap [cite: 385]
        if self._is_end_of_lap(nid):
            return True
            
        # Điều kiện 3: Connected to an end node of an ADJACENT lap [cite: 386]
        # "not an end-node but connected to an end node, say nx, of an adjacent lap"
        neighbors = list(self.graph.neighbors(nid))
        pos = self.nodes[nid]['pos']
        
        for nb in neighbors:
            nb_pos = self.nodes[nb]['pos']
            
            # Kiểm tra xem nb có nằm ở Lap liền kề không? (Khác X khoảng ~W)
            is_adjacent_lap = abs(abs(nb_pos[0] - pos[0]) - SAMPLE_RES_W) < 0.5
            
            if is_adjacent_lap:
                # Kiểm tra xem hàng xóm đó có phải là End Node của lap nó không?
                if self._is_end_of_lap(nb):
                    # (Để đơn giản hóa Condition 3a/3b phức tạp về vật cản, 
                    # ta giữ lại mọi kết nối tới End Node của lap bên cạnh để đảm bảo liên thông)
                    return True
                    
        return False
    def _is_adjacent_to_unknown(self, nid, environment):
        """Helper cho Điều kiện 1"""
        pos = self.nodes[nid]['pos']
        px, py = int(pos[0]), int(pos[1])
        for dx, dy in [(-1,0), (1,0), (0,1), (0,-1)]:
            nx, ny = px+dx, py+dy
            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                if environment.grid[nx, ny] == UNKNOWN_COLOR:
                    return True
        return False
    def _is_end_of_lap(self, nid):
        """Helper cho Điều kiện 2: Kiểm tra xem nút có phải đầu mút của Lap không"""
        # Nút là đầu mút nếu trên cùng Lap (cùng X), nó chỉ có 1 hàng xóm (hoặc 0)
        neighbors = list(self.graph.neighbors(nid))
        pos = self.nodes[nid]['pos']
        
        vertical_neighbors_count = 0
        for nb in neighbors:
            nb_pos = self.nodes[nb]['pos']
            # Cùng X
            if abs(nb_pos[0] - pos[0]) < 0.1:
                vertical_neighbors_count += 1
                
        # Nếu chỉ có 1 hàng xóm dọc (hoặc 0 nếu mới tạo), nó là đầu mút
        return vertical_neighbors_count < 2
    def select_zigzag_goal(self, current_id):
        """
        Chiến lược chọn hướng đi Zig-zag theo Mục III-B-4-a.
        Input: current_id (ID nút hiện tại)
        Output: ID nút mục tiêu tiếp theo hoặc None (nếu Dead-end)
        """
        if current_id is None: return None
        
        # 1. Tìm các hàng xóm 'Open' (Chưa đi)
        # "To obtain n_{i+1} an Open node is searched from the neighbors of n_i" 
        neighbors = list(self.graph.neighbors(current_id))
        open_neighbors = [n for n in neighbors if self.nodes[n]['status'] == 'Open']
        
        # Nếu không có nút Open nào -> Dead-end [cite: 509]
        if not open_neighbors: 
            return None 
            
        curr_pos = self.nodes[current_id]['pos']
        
        # 2. Phân loại hướng (Left, Up, Down, Right)
        # "These directions are defined with respect to a fixed coordinate frame 
        # whose vertical axis is parallel to the laps." 
        # Giả định: Laps chạy dọc theo trục Y -> Dy thay đổi là Up/Down, Dx thay đổi là Left/Right.
        candidates = {'Left': [], 'Up': [], 'Down': [], 'Right': []}
        
        # Ngưỡng sai số (Epsilon) để so sánh số thực
        EPS = 0.1 
        
        for nid in open_neighbors:
            n_pos = self.nodes[nid]['pos']
            dx = n_pos[0] - curr_pos[0]
            dy = n_pos[1] - curr_pos[1]
            
            if dx < -EPS:          # Lệch âm X -> Bên Trái
                candidates['Left'].append(nid)
            elif dx > EPS:         # Lệch dương X -> Bên Phải
                candidates['Right'].append(nid)
            else:                  # Cùng X (dx ~ 0) -> Trên cùng 1 Lap
                if dy > EPS:       # Lệch dương Y -> Phía Trên
                    candidates['Up'].append(nid)
                elif dy < -EPS:    # Lệch âm Y -> Phía Dưới
                    candidates['Down'].append(nid)
        
        # 3. Chọn theo thứ tự ưu tiên
        # "selected based on the priority order as follows: Left -> Up -> Down -> Right" 
        priority_order = ['Left', 'Up', 'Down', 'Right']
        
        for direction in priority_order:
            node_list = candidates[direction]
            if node_list:
                # 4. Xử lý trùng lặp bằng Random
                # "In case there are more than one Open neighbors... then a random pick is done." 
                return random.choice(node_list)
        
        return None

    def detect_holes(self, environment, current_id, next_goal_id):
        """
        Phát hiện lỗ hổng.
        [FIX] Thêm giới hạn kích thước: Chỉ xử lý lỗ hổng NHỎ (Local).
        Nếu vùng quá lớn -> Đó là vùng mở rộng, hãy để Zig-zag xử lý.
        """
        neighbors = list(self.graph.neighbors(current_id))
        candidates = [n for n in neighbors if self.nodes[n]['status'] == 'Open' and n != next_goal_id]
        
        visited_global = set()
        holes = []

        # Chỉ coi là lỗ hổng nếu số lượng nút < ngưỡng này
        MAX_HOLE_SIZE = 15 

        for start_node in candidates:
            if start_node in visited_global: continue
            
            cluster = []
            queue = deque([start_node])
            visited_local = {start_node}
            visited_global.add(start_node)
            is_hole = True 
            
            while queue:
                curr = queue.popleft()
                cluster.append(curr)
                
                # Nếu cụm quá lớn -> Không phải lỗ hổng cục bộ -> Bỏ qua ngay
                if len(cluster) > MAX_HOLE_SIZE:
                    is_hole = False
                    # Vẫn tiếp tục duyệt để đánh dấu visited_global nhưng không add vào holes
                
                cx, cy = int(self.nodes[curr]['pos'][0]), int(self.nodes[curr]['pos'][1])
                
                # Check biên giới Unknown
                for dx, dy in [(-1,0), (1,0), (0,1), (0,-1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
                        if environment.grid[nx, ny] == UNKNOWN_COLOR:
                            is_hole = False
                
                for neighbor in self.graph.neighbors(curr):
                    if neighbor == next_goal_id: continue
                    if self.nodes[neighbor]['status'] == 'Closed': continue
                    
                    if neighbor not in visited_local:
                        visited_local.add(neighbor)
                        visited_global.add(neighbor)
                        queue.append(neighbor)
            
            # Chỉ thêm vào danh sách nếu là lỗ hổng thực sự (nhỏ và cô lập)
            if is_hole and 0 < len(cluster) <= MAX_HOLE_SIZE:
                holes.append(cluster)
                
        return holes

    def solve_tsp(self, cluster_nodes, current_id, next_goal_id=None):
        """
        Giải TSP dùng Nearest Neighbor trên khoảng cách Đồ thị.
        [QUAN TRỌNG]: Tự động nối đường đi giữa các điểm (Interpolation)
        để robot không bị nhảy cóc hoặc đi vòng vô lý.
        """
        if not cluster_nodes: return []
        
        # 1. Xác định điểm đầu/cuối
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
            
            # Tìm điểm gần nhất về mặt ĐI LẠI (không phải chim bay)
            for t in targets:
                try:
                    # Dùng weight='weight' để tính đúng khoảng cách sau khi cắt tỉa
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
                # Nếu không tìm được đường đến các điểm còn lại (bị cô lập)
                break
                

        full_path = []
        for i in range(len(ordered_stops) - 1):
            u = ordered_stops[i]
            v = ordered_stops[i+1]
            try:
                # Tìm đường đi ngắn nhất giữa 2 điểm mốc
                segment = nx.shortest_path(self.graph, u, v, weight='weight')
                # Bỏ điểm đầu của segment để tránh trùng lặp (trừ segment đầu tiên)
                if i == 0:
                    full_path.extend(segment)
                else:
                    full_path.extend(segment[1:])
            except nx.NetworkXNoPath:
                pass
        
        return full_path
    

    def get_nearest_retreat_node(self, current_id):
        """
        Tìm nút lui (Retreat Node) gần nhất bằng thuật toán A*.
        Theo Lemma IV.7 [cite: 666-669]: "path length to each retreat node is calculated using A*"
        """
        # 1. Xác định danh sách các Nút lui (Retreat Nodes) [cite: 523-525]
        # Định nghĩa: Nút Open nằm liền kề với nút Closed
        retreat_candidates = []
        
        # Lấy danh sách Open nodes
        open_nodes = [n for n, data in self.nodes.items() if data['status'] == 'Open']
        
        for nid in open_nodes:
            # Kiểm tra hàng xóm
            try:
                neighbors = self.graph.neighbors(nid)
                for nb in neighbors:
                    if self.nodes[nb]['status'] == 'Closed':
                        retreat_candidates.append(nid)
                        break
            except: continue
        
        if not retreat_candidates: 
            return None

        # 2. Tìm nút gần nhất bằng A* (Graph Distance) [cite: 667]
        # Thay vì đo khoảng cách hình học, ta đo khoảng cách đi lại thực tế trên đồ thị
        best_node = None
        min_dist = float('inf')
        
        curr_pos = self.nodes[current_id]['pos']

        # Hàm Heuristic cho A* (Khoảng cách Euclid)
        def heuristic(u, v):
            return distance.euclidean(self.nodes[u]['pos'], self.nodes[v]['pos'])

        for target_id in retreat_candidates:
            try:
                # Tính độ dài đường đi từ vị trí hiện tại đến target bằng A*
                # weight='weight' sử dụng độ dài cạnh thực tế mà ta đã add trong hàm expand/prune
                dist_astar = nx.astar_path_length(
                    self.graph, 
                    source=current_id, 
                    target=target_id, 
                    heuristic=heuristic, 
                    weight='weight'
                )
                
                if dist_astar < min_dist:
                    min_dist = dist_astar
                    best_node = target_id
                    
            except nx.NetworkXNoPath:
                # Nếu không có đường đi đến nút này (do đồ thị bị ngắt), bỏ qua
                continue
                
        return best_node

# --- HÀM VẼ ---
def to_screen(x, y):
    return int(x * CELL_SIZE), int((MAP_SIZE - y) * CELL_SIZE)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("C* Algorithm: Sparse Graph Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    env = Environment(MAP_SIZE)
    rcg = RCG()
    
    start_pos = (5, 5)
    curr_node = rcg.add_node(*start_pos, status='Closed')
    rcg.current_node_id = curr_node
    
    path_queue = [] 
    step = 0
    mode = "Start"
    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused

        if not paused:
            c_pos = rcg.nodes[rcg.current_node_id]['pos']
            env.sense(c_pos[0], c_pos[1], SENSOR_RANGE)
            
            # 1. Mở rộng RCG (Luôn chạy để cập nhật môi trường)
            rcg.expand(env, c_pos)
            
            # [QUAN TRỌNG] Chỉ cắt tỉa khi KHÔNG có đường dẫn định sẵn
            # Nếu đang bận chạy A* hoặc TSP thì đừng cắt tỉa, kẻo mất đường
            if not path_queue:
                rcg.prune_rcg(env)
            
            next_node = None
            mode = "Zig-Zag"

            # 2. Decision
            if path_queue:
                mode = "Following Path"
                next_node = path_queue.pop(0)
            else:
                tentative_next = rcg.select_zigzag_goal(rcg.current_node_id)
                holes = rcg.detect_holes(env, rcg.current_node_id, tentative_next)
                
                if holes:
                    mode = "Hole -> TSP"
                    tsp_path = rcg.solve_tsp(holes[0], rcg.current_node_id, tentative_next)
                    if len(tsp_path) > 1: 
                        add_path = tsp_path[1:] if tsp_path[0] == rcg.current_node_id else tsp_path
                        path_queue.extend(add_path)
                        next_node = path_queue.pop(0)
                else:
                    next_node = tentative_next
                    if next_node is None:
                        mode = "Dead-end Escape"
                        retreat = rcg.get_nearest_retreat_node(rcg.current_node_id)
                        if retreat:
                            try:
                                p = nx.shortest_path(rcg.graph, rcg.current_node_id, retreat, weight='weight')
                                if len(p) > 1:
                                    path_queue.extend(p[1:])
                                    next_node = path_queue.pop(0)
                            except: mode = "No Path"; paused = True
                        else:
                            mode = "COMPLETE COVERAGE"; paused = True 

            if next_node is not None and next_node in rcg.nodes:
                rcg.nodes[next_node]['status'] = 'Closed'
                rcg.current_node_id = next_node
                step += 1

        # --- DRAWING ---
        screen.fill(COLOR_BG)

        # 1. Map
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                val = env.grid[x, y]
                if val != UNKNOWN_COLOR:
                    col = COLOR_FREE if val == FREE_COLOR else COLOR_OBS
                    pygame.draw.rect(screen, col, (*to_screen(x, y+1), CELL_SIZE, CELL_SIZE))

        # 2. [VISUALIZATION] Pruned Elements (Màu xám)
        for p1, p2 in rcg.pruned_edges:
            pygame.draw.line(screen, COLOR_PRUNED_EDGE, to_screen(*p1), to_screen(*p2), 1)
        for p in rcg.pruned_nodes:
            pygame.draw.circle(screen, COLOR_PRUNED_NODE, to_screen(*p), 2)

        # 3. Active Graph (Màu sáng)
        for u, v in rcg.graph.edges():
            if u in rcg.nodes and v in rcg.nodes:
                p1, p2 = rcg.nodes[u]['pos'], rcg.nodes[v]['pos']
                pygame.draw.line(screen, COLOR_EDGE, to_screen(*p1), to_screen(*p2), 2)

        # 4. Nodes
        for nid, data in rcg.nodes.items():
            col = COLOR_NODE_CLOSED if data['status']=='Closed' else COLOR_NODE_OPEN
            if nid == next_node: col = COLOR_NODE_GOAL
            pygame.draw.circle(screen, col, to_screen(*data['pos']), 4)

        # 5. Robot
        pygame.draw.circle(screen, COLOR_ROBOT, to_screen(*c_pos), 7)

        # UI
        pygame.draw.rect(screen, (30, 30, 30), (0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 50))
        op_cnt = len([n for n in rcg.nodes if rcg.nodes[n]['status']=='Open'])
        info = f"Step: {step} | {mode} | Active Nodes: {len(rcg.nodes)} | Pruned: {len(rcg.pruned_nodes)}"
        screen.blit(font.render(info, True, (255, 255, 255)), (10, SCREEN_HEIGHT - 35))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()