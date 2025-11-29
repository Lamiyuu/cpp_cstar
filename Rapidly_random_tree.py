import pygame
import random
import math

# --- CẤU HÌNH ---
WIDTH, HEIGHT = 800, 600
MAX_ITERATIONS = 5000   
DELTA_Q = 20            
GOAL_RADIUS = 20        
FPS = 60                # Tốc độ vẽ (chỉnh xuống 30 nếu muốn chậm hơn)

# MÀU SẮC
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)       # Màu vật cản (Đen)
GREEN = (0, 200, 0)     # Màu cây RRT (Xanh lá)
RED = (255, 0, 0)       # Màu đường đi cuối cùng (Đỏ)
BLUE = (0, 0, 255)      # Màu điểm Xuất phát (Start)
MAGENTA = (200, 0, 200) # Màu điểm Đích (Goal)

class RRTGraph:
    def __init__(self, start_pos, goal_pos, obstacles):
        self.start = start_pos
        self.goal = goal_pos
        self.obstacles = obstacles
        
        self.vertices = [start_pos]
        self.edges = []
        self.parents = {start_pos: None}
        self.goal_reached = False
        self.final_node = None

    def rand_conf(self):
        # Goal Bias: 5% cơ hội nhắm thẳng vào đích
        if random.random() < 0.01:
            return self.goal
        return (random.randint(0, WIDTH), random.randint(0, HEIGHT))

    def nearest_vertex(self, q_rand):
        nearest_node = None
        min_dist = float('inf')
        for v in self.vertices:
            dist = math.hypot(q_rand[0] - v[0], q_rand[1] - v[1])
            if dist < min_dist:
                min_dist = dist
                nearest_node = v
        return nearest_node

    def new_conf(self, q_near, q_rand, delta_q):
        x_near, y_near = q_near
        x_rand, y_rand = q_rand
        theta = math.atan2(y_rand - y_near, x_rand - x_near)
        dist = math.hypot(x_rand - x_near, y_rand - y_near)
        
        step = min(delta_q, dist)
        new_x = x_near + step * math.cos(theta)
        new_y = y_near + step * math.sin(theta)
        return (int(new_x), int(new_y))

    def is_collision(self, p1, p2):
        steps = 10
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + (p2[0] - p1[0]) * t
            y = p1[1] + (p2[1] - p1[1]) * t
            for rect in self.obstacles:
                if rect.collidepoint(x, y):
                    return True
        return False

    def add_node(self, q_new, q_near):
        self.vertices.append(q_new)
        self.edges.append((q_near, q_new))
        self.parents[q_new] = q_near

        dist_to_goal = math.hypot(q_new[0] - self.goal[0], q_new[1] - self.goal[1])
        if dist_to_goal < GOAL_RADIUS:
            self.goal_reached = True
            self.final_node = q_new

    def get_path(self):
        path = []
        curr = self.final_node
        while curr is not None:
            path.append(curr)
            curr = self.parents[curr]
        return path

def generate_obstacles(num_obstacles):
    obs = []
    for _ in range(num_obstacles):
        w = random.randint(60, 150)
        h = random.randint(60, 150)
        x = random.randint(0, WIDTH - w)
        y = random.randint(0, HEIGHT - h)
        obs.append(pygame.Rect(x, y, w, h))
    return obs

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RRT - Black Obstacles Visualization")
    clock = pygame.time.Clock()

    # SETUP
    start_pos = (50, 50)
    goal_pos = (WIDTH - 50, HEIGHT - 50)
    obstacles = generate_obstacles(10)
    
    # Xóa vật cản đè lên start/goal
    obstacles = [o for o in obstacles if not o.collidepoint(start_pos) and not o.collidepoint(goal_pos)]

    rrt = RRTGraph(start_pos, goal_pos, obstacles)
    
    running = True
    path = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset
                    obstacles = generate_obstacles(10)
                    obstacles = [o for o in obstacles if not o.collidepoint(start_pos) and not o.collidepoint(goal_pos)]
                    rrt = RRTGraph(start_pos, goal_pos, obstacles)
                    path = []
                    screen.fill(WHITE)

        # --- XỬ LÝ LOGIC ---
        if not rrt.goal_reached and len(rrt.vertices) < MAX_ITERATIONS:
            q_rand = rrt.rand_conf()
            q_near = rrt.nearest_vertex(q_rand)
            q_new = rrt.new_conf(q_near, q_rand, DELTA_Q)

            if not rrt.is_collision(q_near, q_new):
                rrt.add_node(q_new, q_near)
        
        if rrt.goal_reached and not path:
            path = rrt.get_path()
            print("DONE! Vẽ đường đi.")

        # --- VẼ HÌNH ---
        screen.fill(WHITE)

        # 1. Vẽ vật cản (MÀU ĐEN)
        for rect in rrt.obstacles:
            pygame.draw.rect(screen, BLACK, rect)

        # 2. Vẽ cây RRT (Màu xanh lá)
        for edge in rrt.edges:
            pygame.draw.line(screen, GREEN, edge[0], edge[1], 1)

        # 3. Vẽ điểm Start và Goal
        # Start: Xanh dương
        pygame.draw.circle(screen, BLUE, start_pos, 8)  
        # Goal: Tím (Vùng và Tâm)
        pygame.draw.circle(screen, MAGENTA, goal_pos, GOAL_RADIUS, 2) 
        pygame.draw.circle(screen, MAGENTA, goal_pos, 5)

        # 4. Vẽ đường đi (Màu Đỏ Đậm)
        if path:
            if len(path) >= 2:
                pygame.draw.lines(screen, RED, False, path, 4)

        pygame.display.flip()
        clock.tick(FPS) 

    pygame.quit()

if __name__ == "__main__":
    main()