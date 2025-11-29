import pygame
import numpy as np
import math
import sys

# --- CẤU HÌNH ---
WINDOW_SIZE = 800  # Màn hình lớn hơn chút để dễ nhìn
SCALE = 120        # Phóng to hơn
FPS = 30

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200) # Màu cho các nhánh tìm kiếm
CYAN = (0, 255, 255)   # Màu cho các nút đã thăm

# Setup Bài toán
START_POS = np.array([0.5, 0.5])
GOAL_POS = np.array([5.5, 5.5])
OBSTACLES = [
    (2.5, 2.5, 1.0), 
    (4.0, 1.5, 0.6),
    (1.5, 4.0, 0.5)  # Thêm vật cản để thấy rõ khả năng luồn lách
]

# Paper Parameters
EPSILON = 0.3          
EXPLORATION_C = 1.414  
GAMMA = 1.0            
MAX_ITERATIONS = 1000   # Tăng số lần suy nghĩ để cây dày hơn, dễ nhìn hơn
SEARCH_DEPTH = 15      

# --- HÀM PHỤ TRỢ ---
def to_screen(pos):
    return int(pos[0] * SCALE), int(pos[1] * SCALE)

def dist(a, b):
    return np.linalg.norm(a - b)

def is_collision(pos):
    if pos[0] < 0 or pos[0] > WINDOW_SIZE/SCALE or pos[1] < 0 or pos[1] > WINDOW_SIZE/SCALE:
        return True
    for (ox, oy, r) in OBSTACLES:
        if dist(pos, np.array([ox, oy])) <= r:
            return True
    return False

# --- CẤU TRÚC DỮ LIỆU ---
class QNode:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action        
        self.n = 0                  
        self.Q = 0.0                
        self.child_v_node = None    
        self.r_cumulative = 0.0     

class VNode:
    def __init__(self, state):
        self.state = state          
        self.N = 0                  
        self.V = 0.0                
        self.children = {}          

# --- MCPP PLANNER (ALGORITHM 1) ---
class MCPP_Planner:
    def __init__(self, start, goal):
        self.root = VNode(start)
        self.goal = goal

    def get_action_ucb(self, v_node):
        best_score = -float('inf')
        best_action = None
        for action, q_node in v_node.children.items():
            if q_node.n == 0:
                return action, q_node
            score = q_node.Q + EXPLORATION_C * math.sqrt(math.log(max(1, v_node.N)) / q_node.n)
            if score > best_score:
                best_score = score
                best_action = (action, q_node)
        return best_action

    def expand(self, v_node):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.1, EPSILON) 
        dx = r * np.cos(angle)
        dy = r * np.sin(angle)
        new_pos = v_node.state + np.array([dx, dy])
        
        if is_collision(new_pos):
            return None

        action = tuple(new_pos)
        if action not in v_node.children:
            v_node.children[action] = QNode(v_node, action)
        return action

    def rollout(self, state):
        return -dist(state, self.goal) 

    def simulate_v(self, v_node, depth):
        if depth == 0 or dist(v_node.state, self.goal) < 0.2:
            return self.rollout(v_node.state)

        if len(v_node.children) < 5: # Giảm số nhánh con để cây sâu hơn thay vì rộng
            action = self.expand(v_node)
            if action:
                q_node = v_node.children[action]
                R = self.simulate_q(v_node, action, q_node, depth)
                v_node.N += 1
                return R

        if not v_node.children:
             return self.rollout(v_node.state)

        action, q_node = self.get_action_ucb(v_node)
        R = self.simulate_q(v_node, action, q_node, depth)
        v_node.N += 1
        return R

    def simulate_q(self, parent_v_node, action, q_node, depth):
        step_dist = dist(parent_v_node.state, np.array(action))
        r_immediate = -step_dist 

        if q_node.child_v_node is None:
            q_node.child_v_node = VNode(np.array(action))
            future_val = self.rollout(q_node.child_v_node.state)
        else:
            future_val = self.simulate_v(q_node.child_v_node, depth - 1)

        total_return = r_immediate + GAMMA * future_val
        q_node.n += 1
        q_node.r_cumulative += total_return
        q_node.Q = q_node.r_cumulative / q_node.n
        return total_return

    def search(self):
        for _ in range(MAX_ITERATIONS):
            self.simulate_v(self.root, SEARCH_DEPTH)
        
        if not self.root.children:
            return self.root.state
        best_action = max(self.root.children.items(), key=lambda item: item[1].Q)[0]
        return np.array(best_action)

# --- HÀM VẼ CÂY TÌM KIẾM (VISUALIZATION) ---

def draw_tree(surface, v_node, parent_pos=None):
    """Đệ quy vẽ toàn bộ cây MCTS"""
    if v_node is None: return

    current_pos_screen = to_screen(v_node.state)

    # Vẽ đường nối từ cha đến con
    if parent_pos is not None:
        # Độ dày nét vẽ phụ thuộc vào số lần thăm N (càng thăm nhiều càng đậm)
        thickness = 1
        if v_node.N > 10: thickness = 2
        pygame.draw.line(surface, GRAY, parent_pos, current_pos_screen, thickness)

    # Vẽ nút
    # pygame.draw.circle(surface, CYAN, current_pos_screen, 2)

    # Đệ quy vẽ con
    for action, q_node in v_node.children.items():
        if q_node.child_v_node:
            draw_tree(surface, q_node.child_v_node, current_pos_screen)

def draw_best_predicted_path(surface, v_node):
    """Vẽ dự đoán đường đi tốt nhất hiện tại (Greedy theo Q)"""
    path_points = []
    curr = v_node
    
    while curr and curr.children:
        path_points.append(to_screen(curr.state))
        # Chọn con có Q cao nhất
        best_action, best_q_node = max(curr.children.items(), key=lambda item: item[1].Q)
        curr = best_q_node.child_v_node
        if dist(curr.state, GOAL_POS) < 0.5: break # Gần đích thì dừng vẽ
        
    if len(path_points) > 1:
        pygame.draw.lines(surface, RED, False, path_points, 2)


# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("MCPP Visualization - Seeing the Thoughts")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    current_pos = START_POS
    path_history = [START_POS]
    finished = False
    
    # Biến planner lưu bên ngoài vòng lặp để ta có thể vẽ cây sau khi search
    planner = None 

    print("Blue/Gray lines: Exploration Tree (Thinking)")
    print("Red line: Planned Best Path")
    print("Black line: Actual Path")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # 1. Vẽ Map
        for (ox, oy, r) in OBSTACLES:
            center = to_screen((ox, oy))
            radius = int(r * SCALE)
            pygame.draw.circle(screen, (255, 200, 200), center, radius) # Vật cản màu hồng nhạt
            pygame.draw.circle(screen, RED, center, radius, 2)

        pygame.draw.circle(screen, GREEN, to_screen(START_POS), 8)
        pygame.draw.circle(screen, BLUE, to_screen(GOAL_POS), 10)

        # 2. Logic Robot & Xây dựng cây
        if not finished:
            if dist(current_pos, GOAL_POS) < 0.3:
                finished = True
                print("Goal Reached!")
            else:
                planner = MCPP_Planner(current_pos, GOAL_POS)
                next_pos = planner.search() # Xây dựng cây tìm kiếm tại đây
                
                # Di chuyển thật
                current_pos = next_pos
                path_history.append(current_pos)

        # 3. VISUALIZATION: VẼ "SUY NGHĨ" CỦA ROBOT
        if planner and planner.root:
            # Vẽ đám mây các khả năng đã tìm kiếm
            draw_tree(screen, planner.root)
            # Vẽ đường dự kiến tốt nhất (Màu đỏ)
            draw_best_predicted_path(screen, planner.root)

        # 4. Vẽ đường đi thực tế (Lịch sử)
        if len(path_history) > 1:
            points = [to_screen(p) for p in path_history]
            pygame.draw.lines(screen, BLACK, False, points, 3)

        pygame.draw.circle(screen, BLACK, to_screen(current_pos), 6)
        
        # Legend
        screen.blit(font.render("Gray: Search Tree (Thinking)", True, (100, 100, 100)), (10, 10))
        screen.blit(font.render("Red: Predicted Best Path", True, RED), (10, 30))
        screen.blit(font.render("Black: Actual Path", True, BLACK), (10, 50))

        pygame.display.flip()
        # Giảm tốc độ xuống m
        clock.tick(5) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()