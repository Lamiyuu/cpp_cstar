import random
import math
import numpy as np
from config import *
from core_math import dist, normalize_angle, simulate_step, reeds_shepp_planning
from environment import check_path_collision

class Node:
    def __init__(self, x, y, yaw, parent=None, is_dubins=False, direction=1):
        self.x = x; self.y = y; self.yaw = yaw
        self.parent = parent
        self.path_x = []; self.path_y = []; self.path_yaw = []
        self.is_dubins = is_dubins
        self.direction = direction

class KinematicRRT:
    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, bounds, dyn_obs=None):
        self.start = Node(start[0], start[1], start[2])
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes 
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.node_list = [self.start]
        # Ép RRT ngó lơ bóng động để đường vẽ không bị kẹt rễ
        self.dyn_obs = None 

    def plan_step(self):
        if random.random() < RRT_GOAL_PROB: rnd = (self.goal_pos[0], self.goal_pos[1])
        else: rnd = (random.uniform(self.min_x, self.max_x), random.uniform(self.min_y, self.max_y))
        
        dists = [(node.x - rnd[0])**2 + (node.y - rnd[1])**2 for node in self.node_list]
        nearest = self.node_list[dists.index(min(dists))]
        
        dx = rnd[0] - nearest.x; dy = rnd[1] - nearest.y
        target_yaw = math.atan2(dy, dx)
        diff_head = normalize_angle(target_yaw - nearest.yaw)
        diff_tail = normalize_angle(target_yaw - (nearest.yaw + math.pi))
        
        direction = 1
        if random.random() < PROB_REVERSE: 
            direction = -1; diff = diff_tail
        else: 
            direction = 1; diff = diff_head
            
        steer = max(-MAX_STEER, min(MAX_STEER, diff))
        nx, ny, nyaw, px, py, pyaw = simulate_step(nearest.x, nearest.y, nearest.yaw, steer, direction)
        
        if not check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs):
            new_node = Node(nx, ny, nyaw, nearest, is_dubins=False, direction=direction)
            new_node.path_x = px; new_node.path_y = py; new_node.path_yaw = pyaw
            self.node_list.append(new_node)
            
            if dist((nx, ny), self.goal_pos) <= DUBINS_CONNECT_DIST:
                dpath = reeds_shepp_planning(nx, ny, nyaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
                if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
                    goal_node = Node(self.goal_pos[0], self.goal_pos[1], self.goal_yaw, new_node, is_dubins=True)
                    goal_node.path_x = dpath.x; goal_node.path_y = dpath.y; goal_node.path_yaw = dpath.yaw
                    return self.extract_path(goal_node)
            
            if dist((nx, ny), self.goal_pos) < GOAL_RADIUS:
                return self.extract_path(new_node)
        return None

    def extract_path(self, node):
        full_path = []
        while node.parent:
            points = list(zip(node.path_x, node.path_y, node.path_yaw))
            segment = {'points': points, 'is_dubins': node.is_dubins, 'direction': node.direction}
            full_path.insert(0, segment)
            node = node.parent
        return full_path

class KinematicMCPP:
    class VNode: 
        def __init__(self, state, is_dubins=False, direction=1):
            self.state = state 
            self.N = 0; self.children = {}
            self.parent_node = None
            self.path_x = []; self.path_y = []; self.path_yaw = []
            self.is_dubins = is_dubins
            self.direction = direction
    
    class QNode: 
        def __init__(self, parent, action):
            self.parent = parent; self.action = action
            self.n = 0; self.Q = 0.0; self.child_v = None

    def __init__(self, start, goal_pos, goal_yaw, outer, known_holes, global_penalties=None, dyn_obs=None):
        self.root = self.VNode(start)
        self.goal_pos = goal_pos; self.goal_yaw = goal_yaw
        self.outer = outer; self.known_holes = known_holes
        self.node_list = [self.root]
        
        # Ép MCPP ngó lơ bóng động để đường vẽ không bị kẹt rễ
        self.dyn_obs = None
        
        self.grid_visits = {}     
        self.grid_penalties = global_penalties if global_penalties is not None else {} 

    def get_dist_to_nearest_obstacle(self, state):
        min_d = 50.0 
        px, py = state[0], state[1]
        for hole in self.known_holes:
            for vertex in hole:
                d = math.sqrt((px - vertex[0])**2 + (py - vertex[1])**2)
                if d < min_d: min_d = d
        return min_d

    def get_action_ucb(self, v):
        best_s = -float('inf'); best_a = None
        for a, q in v.children.items():
            if q.n == 0: return a, q
            curr_c = MCPP_C * (1.5 if self.get_dist_to_nearest_obstacle(v.state) < 10.0 else 1.0)
            s = q.Q + curr_c * math.sqrt(math.log(max(1, v.N)) / q.n)
            
            if q.child_v:
                gid = (int(q.child_v.state[0] // BIG_GRID_SIZE), int(q.child_v.state[1] // BIG_GRID_SIZE))
                s -= self.grid_penalties.get(gid, 0)
                
            if s > best_s: best_s = s; best_a = (a, q)
        return best_a

    def expand(self, v):
        d_obs = self.get_dist_to_nearest_obstacle(v.state)
        if d_obs < 15.0:
            steer = random.choice([-MAX_STEER, MAX_STEER, random.uniform(-MAX_STEER, MAX_STEER)])
        else:
            steer = random.uniform(-MAX_STEER, MAX_STEER)
            
        direction = -1 if random.random() < PROB_REVERSE else 1
        nx, ny, nyaw, px, py, pyaw = simulate_step(v.state[0], v.state[1], v.state[2], steer, direction)
        
        if check_path_collision(px, py, pyaw, self.outer, self.known_holes, self.dyn_obs): return None
        
        action_key = (round(steer, 2), direction)
        if action_key not in v.children:
            qnode = self.QNode(v, action_key)
            v.children[action_key] = qnode
            return action_key
        return None

    def sim_v(self, v, d):
        sx, sy, syaw = v.state[0], v.state[1], v.state[2]
        
        if dist((sx, sy), self.goal_pos) < DUBINS_CONNECT_DIST:
            dpath = reeds_shepp_planning(sx, sy, syaw, self.goal_pos[0], self.goal_pos[1], self.goal_yaw, MIN_TURN_RADIUS)
            
            if dpath and not check_path_collision(dpath.x, dpath.y, dpath.yaw, self.outer, self.known_holes, self.dyn_obs):
                goal_v = self.VNode((self.goal_pos[0], self.goal_pos[1], self.goal_yaw), is_dubins=True)
                goal_v.parent_node = v
                goal_v.path_x, goal_v.path_y, goal_v.path_yaw = dpath.x, dpath.y, dpath.yaw
                goal_v.direction = -1 if any(l < 0 for l in dpath.lengths) else 1
                self.node_list.append(goal_v) 
                return 2000.0

        if d == 0 or dist(v.state[:2], self.goal_pos) < GOAL_RADIUS:
            d_goal = dist(v.state[:2], self.goal_pos)
            d_obs = self.get_dist_to_nearest_obstacle(v.state)
            return -(1.0 * d_goal) + (2.0 * d_obs if d_obs < 10.0 else 0.5 * d_obs)

        max_b = MCPP_BRANCHES * (2 if self.get_dist_to_nearest_obstacle(v.state) < 10.0 else 1)
        if len(v.children) < max_b:
            act = self.expand(v)
            if act: return self.sim_q(v.children[act], d)
        
        if not v.children: return -dist(v.state[:2], self.goal_pos)
        
        res = self.get_action_ucb(v)
        if res: return self.sim_q(res[1], d)
        return -dist(v.state[:2], self.goal_pos)

    def sim_q(self, q, d):
        steer, direction = q.action
        if not q.child_v:
            nx, ny, nyaw, px, py, pyaw = simulate_step(q.parent.state[0], q.parent.state[1], q.parent.state[2], steer, direction)
            q.child_v = self.VNode((nx, ny, nyaw), is_dubins=False, direction=direction)
            q.child_v.parent_node = q.parent
            q.child_v.path_x, q.child_v.path_y, q.child_v.path_yaw = px, py, pyaw
            self.node_list.append(q.child_v)
            
            gid = (int(nx // BIG_GRID_SIZE), int(ny // BIG_GRID_SIZE))
            self.grid_visits[gid] = self.grid_visits.get(gid, 0) + 1
            
            if self.grid_visits[gid] > MAX_STEPS_PER_GRID:
                self.grid_penalties[gid] = self.grid_penalties.get(gid, 0) + 2000.0
                self.grid_visits[gid] = 0 
            
            curr = (nx, ny, nyaw)
            survive_steps = 0
            for _ in range(3):
                rx, ry, ryaw, rpx, rpy, _ = simulate_step(curr[0], curr[1], curr[2], random.uniform(-MAX_STEER, MAX_STEER), 1)
                if check_path_collision(rpx, rpy, [0]*len(rpx), self.outer, self.known_holes, self.dyn_obs): break
                curr = (rx, ry, ryaw)
            return -dist(curr[:2], self.goal_pos) + (0.5 * self.get_dist_to_nearest_obstacle(curr))

        r = self.sim_v(q.child_v, d - 1)
        q.n += 1; q.Q += (r - q.Q) / q.n; q.parent.N += 1
        return r

    def plan_step(self):
        for _ in range(MCPP_ITER // 40):
            self.sim_v(self.root, MCPP_DEPTH)
        
        for node in self.node_list:
            if dist(node.state[:2], self.goal_pos) < GOAL_RADIUS and abs(normalize_angle(node.state[2] - self.goal_yaw)) < 0.2:
                return self.extract_path(node)
        return None

    def extract_path(self, node):
        full_path = []
        curr = node
        while curr and curr.parent_node:
            points = list(zip(curr.path_x, curr.path_y, curr.path_yaw))
            full_path.insert(0, {'points': points, 'is_dubins': curr.is_dubins, 'direction': curr.direction})
            curr = curr.parent_node
        return full_path