import pygame
import numpy as np
import math
import sys

# --- CẤU HÌNH ---
WINDOW_SIZE = 800
SCALE = 100
FPS = 30

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)      # Vật cản đã biết (Known)
GHOST_GRAY = (220, 220, 220) # Vật cản ẩn (Hidden)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)   # Màu cảnh báo va chạm

# Setup Bài toán
START_POS = np.array([0.5, 3.0])
GOAL_POS = np.array([7.5, 3.0])

# Bản đồ vật cản THỰC TẾ (Robot không biết lúc đầu)
# (x, y, radius)
REAL_OBSTACLES = [
    (2.5, 3.0, 0.8), 
    (4.5, 2.0, 0.8),
    (4.5, 4.0, 0.8),
    (6.0, 3.0, 0.6)
]

# Tham số MCPP
EPSILON = 0.3          
EXPLORATION_C = 1.414  
GAMMA = 1.0            
MAX_ITERATIONS = 1000   
SEARCH_DEPTH = 15      
SENSOR_RANGE = 0.3 # Khoảng cách robot phát hiện ra vật cản (Assumption 2)

# --- HÀM PHỤ TRỢ ---
def to_screen(pos):
    return int(pos[0] * SCALE), int(pos[1] * SCALE)

def dist(a, b):
    return np.linalg.norm(a - b)

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

# --- MCPP PLANNER (POMDP VERSION) ---
class MCPP_Planner:
    def __init__(self, start, goal, known_obstacles):
        self.root = VNode(start)
        self.goal = goal
        self.known_obstacles = known_obstacles # Chỉ tránh những gì đã biết

    def is_collision_belief(self, pos):
        """Chỉ kiểm tra va chạm với các vật cản ĐÃ BIẾT"""
        if pos[0] < 0 or pos[0] > WINDOW_SIZE/SCALE or pos[1] < 0 or pos[1] > WINDOW_SIZE/SCALE:
            return True
        for (ox, oy, r) in self.known_obstacles:
            if dist(pos, np.array([ox, oy])) <= r + 0.1: # +0.1 padding an toàn
                return True
        return False

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
        
        # Kiểm tra va chạm dựa trên NIỀM TIN (Belief)
        if self.is_collision_belief(new_pos):
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

        if len(v_node.children) < 8:
            action = self.expand(v_node)
            if action:
                q_node = v_node.children[action]
                R = self.simulate_q(v_node, action, q_node, depth)
                v_node.N += 1
                self.update_V(v_node) # Strict update
                return R

        if not v_node.children:
             return self.rollout(v_node.state)

        action, q_node = self.get_action_ucb(v_node)
        R = self.simulate_q(v_node, action, q_node, depth)
        v_node.N += 1
        self.update_V(v_node)
        return R
    
    def update_V(self, v_node):
        if v_node.N == 0: return
        weighted_sum = 0
        for action, q_node in v_node.children.items():
            weighted_sum += (q_node.n / v_node.N) * q_node.Q
        v_node.V = weighted_sum

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

def draw_tree(surface, v_node, parent_pos=None):
    if v_node is None: return
    current_pos_screen = to_screen(v_node.state)
    if parent_pos is not None:
        thickness = 1 if v_node.N < 5 else 2
        pygame.draw.line(surface, (150, 150, 150), parent_pos, current_pos_screen, thickness)
    for action, q_node in v_node.children.items():
        if q_node.child_v_node:
            draw_tree(surface, q_node.child_v_node, current_pos_screen)

# --- MAIN LOOP ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, 600)) # Cao 600 là đủ
    pygame.display.set_caption("POMDP MCPP - Hidden Obstacles")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    current_pos = START_POS
    path_history = [START_POS]
    
    # DANH SÁCH VẬT CẢN ĐÃ BIẾT (Ban đầu rỗng)
    known_obstacles = [] 
    
    finished = False
    collision_effect = 0 # Hiệu ứng nháy màn hình khi va chạm

    print("POMDP Simulation Started.")
    print("Robot does NOT know obstacles initially.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # 1. Vẽ Vật Cản
        for ob in REAL_OBSTACLES:
            ox, oy, r = ob
            center = to_screen((ox, oy))
            radius = int(r * SCALE)
            
            # Kiểm tra xem vật cản này đã được phát hiện chưa
            is_known = False
            for ko in known_obstacles:
                if ko == ob: is_known = True
            
            if is_known:
                # Đã phát hiện: Vẽ màu đỏ đậm
                pygame.draw.circle(screen, RED, center, radius)
                pygame.draw.circle(screen, BLACK, center, radius, 2)
            else:
                # Chưa phát hiện: Vẽ nét đứt xám (Ghost)
                pygame.draw.circle(screen, GHOST_GRAY, center, radius, 2)
                # Vẽ tâm nhỏ để biết vị trí
                pygame.draw.circle(screen, GHOST_GRAY, center, 3)

        pygame.draw.circle(screen, GREEN, to_screen(START_POS), 8)
        pygame.draw.circle(screen, BLUE, to_screen(GOAL_POS), 10)

        # 2. Logic Robot POMDP
        planner = None
        if not finished:
            if dist(current_pos, GOAL_POS) < 0.3:
                finished = True
                print("Goal Reached!")
            else:
                # A. Lập kế hoạch dựa trên NIỀM TIN hiện tại (known_obstacles)
                planner = MCPP_Planner(current_pos, GOAL_POS, known_obstacles)
                proposed_next_pos = planner.search() 
                
                # B. Thử di chuyển (Thực tế)
                # Kiểm tra va chạm với vật cản THỰC TẾ
                actual_collision = False
                hit_obstacle = None
                
                # Kiểm tra va chạm cứng (Crash)
                for ob in REAL_OBSTACLES:
                    ox, oy, r = ob
                    if dist(proposed_next_pos, np.array([ox, oy])) <= r:
                        actual_collision = True
                        hit_obstacle = ob
                        break
                
                # Kiểm tra cảm biến (Sensor Discovery) - Assumption 2
                # Nếu robot đến gần vật cản ẩn, nó sẽ phát hiện ra
                for ob in REAL_OBSTACLES:
                    ox, oy, r = ob
                    if ob not in known_obstacles:
                        if dist(current_pos, np.array([ox, oy])) <= r + SENSOR_RANGE:
                            known_obstacles.append(ob)
                            collision_effect = 5 # Visual effect
                            print(f"Sensor detected obstacle at {ox}, {oy}!")

                if actual_collision:
                    # Nếu va chạm thực sự:
                    # 1. Không di chuyển
                    # 2. Cập nhật vật cản vào bộ nhớ
                    if hit_obstacle not in known_obstacles:
                        known_obstacles.append(hit_obstacle)
                        collision_effect = 10
                        print("CRASH! Obstacle discovered via collision.")
                    
                    # 3. Lùi lại một chút (Backtracking) để không bị kẹt
                    # Vector lùi lại hướng về cha
                    retreat_vec = (path_history[-2] - current_pos) if len(path_history) > 1 else np.array([0,0])
                    if np.linalg.norm(retreat_vec) > 0:
                        retreat_vec = retreat_vec / np.linalg.norm(retreat_vec) * 0.2
                        current_pos = current_pos + retreat_vec
                    
                else:
                    # Nếu an toàn, di chuyển thật
                    current_pos = proposed_next_pos
                    path_history.append(current_pos)

        # 3. Visualization
        if planner and planner.root:
            draw_tree(screen, planner.root)

        if len(path_history) > 1:
            points = [to_screen(p) for p in path_history]
            pygame.draw.lines(screen, BLACK, False, points, 3)

        # Vẽ Robot
        pygame.draw.circle(screen, BLACK, to_screen(current_pos), 6)

        # Hiệu ứng va chạm (Màn hình chớp cam)
        if collision_effect > 0:
            s = pygame.Surface((WINDOW_SIZE, 600))
            s.set_alpha(50)
            s.fill(ORANGE)
            screen.blit(s, (0,0))
            collision_effect -= 1

        # Text info
        status = "Goal Reached" if finished else f"Known Obstacles: {len(known_obstacles)}/{len(REAL_OBSTACLES)}"
        screen.blit(font.render(status, True, BLACK), (10, 10))
        screen.blit(font.render("Gray Circles: Hidden (Unknown)", True, (150, 150, 150)), (10, 30))
        screen.blit(font.render("Red Circles: Discovered (Memory)", True, RED), (10, 50))

        pygame.display.flip()
        clock.tick(FPS) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()