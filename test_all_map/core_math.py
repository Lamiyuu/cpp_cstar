import numpy as np
import math
from config import *

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def mod2pi(theta): 
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)

def dist(a, b): 
    return np.linalg.norm(np.array(a) - np.array(b))

def point_in_polygon(point, polygon):
    x, y = point
    if not polygon: return False
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters: inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def point_to_segment_dist(px, py, ax, ay, bx, by):
    l2 = (bx - ax)**2 + (by - ay)**2
    if l2 == 0: return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    return math.hypot(px - proj_x, py - proj_y)

def get_car_corners(x, y, yaw):
    rear_overhang = 1.0; front_overhang = 1.0
    local = [(-rear_overhang, CAR_WIDTH/2), (-rear_overhang, -CAR_WIDTH/2),
             (CAR_L + front_overhang, -CAR_WIDTH/2), (CAR_L + front_overhang, CAR_WIDTH/2)]
    world = []
    c, s = math.cos(yaw), math.sin(yaw)
    for lx, ly in local:
        world.append((lx*c - ly*s + x, lx*s + ly*c + y))
    return world

def simulate_step(x, y, yaw, steer, direction, sim_time=SIM_TIME):
    path_x, path_y, path_yaw = [x], [y], [yaw]
    steps = int(sim_time / DT)
    base_vel = VELOCITY_MIN if direction == -1 else VELOCITY_MAX
    steer_factor = abs(steer) / MAX_STEER
    current_vel = base_vel * (1.0 - 0.5 * steer_factor)
    v = current_vel * direction

    for _ in range(steps):
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += (v / CAR_L) * math.tan(steer) * DT
        yaw = normalize_angle(yaw)
        path_x.append(x); path_y.append(y); path_yaw.append(yaw)
    return x, y, yaw, path_x, path_y, path_yaw

# --- BỘ GIẢI REEDS-SHEPP ---
class RSPath:
    def __init__(self, lengths, modes, total_len):
        self.lengths = lengths  
        self.modes = modes      
        self.length = total_len
        self.x, self.y, self.yaw = [], [], []

def LSL(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = 2 + d*d - (2*math.cos(a - b)) + (2*d*(sa - sb))
    if p_sq < 0: return None
    tmp = math.atan2((cb - ca), (d + sa - sb))
    return mod2pi(-a + tmp), math.sqrt(p_sq), mod2pi(b - tmp), ["L", "S", "L"]

def RSR(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = 2 + d*d - (2*math.cos(a - b)) + (2*d*(sb - sa))
    if p_sq < 0: return None
    tmp = math.atan2((ca - cb), (d - sa + sb))
    return mod2pi(a - tmp), math.sqrt(p_sq), mod2pi(-b + tmp), ["R", "S", "R"]

def LSR(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = -2 + d*d + (2*math.cos(a - b)) + (2*d*(sa + sb))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    return mod2pi(-a + tmp), p, mod2pi(-mod2pi(b) + tmp), ["L", "S", "R"]

def RSL(a, b, d):
    sa, sb, ca, cb = math.sin(a), math.sin(b), math.cos(a), math.cos(b)
    p_sq = d*d - 2 + (2*math.cos(a - b)) - (2*d*(sa + sb))
    if p_sq < 0: return None
    p = math.sqrt(p_sq)
    tmp = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    return mod2pi(a - tmp), p, mod2pi(b - tmp), ["R", "S", "L"]

def CCC_FBF(alpha, beta, d):
    x = d * math.cos(alpha)
    y = -d * math.sin(alpha)
    phi = normalize_angle(beta - alpha)
    best_t, best_p, best_q, best_mode = None, None, None, None
    min_len = float('inf')

    def calc_LRL(X, Y, PHI):
        u1 = math.sqrt((X - math.sin(PHI))**2 + (Y - 1.0 + math.cos(PHI))**2)
        if u1 <= 4.0:
            P = -2.0 * math.asin(0.25 * u1)
            T = mod2pi(math.atan2(Y - 1.0 + math.cos(PHI), X - math.sin(PHI)) + 0.5 * P + math.pi)
            Q = mod2pi(PHI - T + P)
            return T, P, Q
        return None

    res_LRL = calc_LRL(x, y, phi)
    if res_LRL:
        t, p, q = res_LRL
        l = abs(t) + abs(p) + abs(q)
        if l < min_len:
            min_len = l
            best_t, best_p, best_q, best_mode = t, p, q, ["L", "R", "L"]

    res_RLR = calc_LRL(x, -y, -phi)
    if res_RLR:
        t, p, q = res_RLR
        l = abs(t) + abs(p) + abs(q)
        if l < min_len:
            best_t, best_p, best_q, best_mode = t, p, q, ["R", "L", "R"]

    if best_mode: return best_t, best_p, best_q, best_mode
    return None

def reeds_shepp_planning(sx, sy, syaw, ex, ey, eyaw, c):
    dx = ex - sx; dy = ey - sy
    lex = math.cos(syaw) * dx + math.sin(syaw) * dy
    ley = math.sin(syaw) * dx - math.cos(syaw) * dy  
    leyaw = normalize_angle(-(eyaw - syaw)) 
    D = math.sqrt(lex**2 + ley**2); d = D / c 
    phi = math.atan2(ley, lex)
    
    alpha = normalize_angle(-phi); beta = normalize_angle(leyaw - phi)
    best_p = None; min_l = float('inf')
    
    def flip_modes(modes): return ["R" if m == "L" else ("L" if m == "R" else "S") for m in modes]

    symmetries = [
        (alpha, beta, False, False), 
        (normalize_angle(-alpha - math.pi), normalize_angle(-beta - math.pi), True, False), 
        (normalize_angle(-alpha), normalize_angle(-beta), False, True), 
        (normalize_angle(alpha + math.pi), normalize_angle(beta + math.pi), True, True) 
    ]
    
    for f in [LSL, RSR, LSR, RSL, CCC_FBF]:
        for a, b, is_timeflip, is_reflect in symmetries:
            res = f(a, b, d)
            if res:
                t, p, q, m = res
                if is_timeflip: t, p, q = -t, -p, -q
                if is_reflect: m = flip_modes(m)
                l = (abs(t) + abs(p) + abs(q)) * c
                if l < min_l and l < (D * 3.0 + 25.0): 
                    min_l = l
                    best_p = RSPath([t, p, q], m, l)

    if best_p:
        best_p.x, best_p.y, best_p.yaw = generate_rs_points(best_p, c, sx, sy, syaw)
        return best_p
    return None

def generate_rs_points(path, c, sx, sy, syaw):
    px, py, pyaw = [sx], [sy], [syaw]
    cx, cy, cyaw = sx, sy, syaw
    for i, mode in enumerate(path.modes):
        L = path.lengths[i] * c
        gear = 1 if L >= 0 else -1
        abs_L = abs(L)
        step = 0.1 
        n = max(1, int(abs_L / step))
        d_step = abs_L / n
        steer = -1 if mode == 'L' else (1 if mode == 'R' else 0)
        for _ in range(n):
            cx += d_step * gear * math.cos(cyaw)
            cy += d_step * gear * math.sin(cyaw)
            if mode != 'S': cyaw += (d_step * gear / c) * steer
            px.append(cx); py.append(cy); pyaw.append(normalize_angle(cyaw))
    return px, py, pyaw