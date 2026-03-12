import math

# ==========================================
# CẤU HÌNH & THAM SỐ HỆ THỐNG
# ==========================================
WINDOW_SIZE = 700
FPS = 60
DATASET_DIR = "AC300"
RESULT_DIR = "Results"

# --- THÔNG SỐ XE (KINEMATIC) ---
CAR_L = 3.0           # Chiều dài cơ sở
CAR_WIDTH = 1.5       
MAX_STEER = 0.6       # ~40 độ
VELOCITY_MAX = 5.0    
VELOCITY_MIN = 2.0    # Tốc độ lùi/cua gắt
DT = 0.2              
SIM_TIME = 1.0        

# --- THÔNG SỐ CẢM BIẾN & RADAR ---
LOOKAHEAD_STEPS = 5   
SENSOR_RADIUS = 25.0  # Bán kính tầm nhìn của Radar

# --- THÔNG SỐ CHƯỚNG NGẠI VẬT ĐỘNG ---
NUM_DYN_OBS = 6       
DYN_OBS_RADIUS = 2.0  
DYN_OBS_SPEED = 12.0  

# --- THÔNG SỐ DUBINS / REEDS-SHEPP ---
MIN_TURN_RADIUS = CAR_L / math.tan(MAX_STEER)
DUBINS_STEP_SIZE = VELOCITY_MAX * DT 
DUBINS_CONNECT_DIST = 20.0 
GOAL_RADIUS = 3.0     

# --- THAM SỐ THUẬT TOÁN ---
RRT_MAX_ITER = 5000   
RRT_GOAL_PROB = 0.05  
PROB_REVERSE = 0.4   

MCPP_EPSILON = 5.0    
MCPP_C = 1.414        
MCPP_ITER = 2000      
MCPP_DEPTH = 30       
MCPP_BRANCHES = 5

BIG_GRID_SIZE = 20.0    
MAX_STEPS_PER_GRID = 25  

# --- MÀU SẮC ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)
CAR_COLOR = (50, 50, 200)
LOOKAHEAD_COLOR = (255, 140, 0) 
GHOST_GRAY = (245, 245, 245)
DUBINS_COLOR = (180, 0, 180) 
REVERSE_COLOR = (255, 100, 0) 
DYN_COLOR = (255, 100, 150)