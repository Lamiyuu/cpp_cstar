import math
# ==========================================
# CẤU HÌNH & THAM SỐ HỆ THỐNG
# ==========================================
WINDOW_SIZE = 700
FPS = 60
DATASET_DIR = "AC300"
RESULT_DIR = "Results"

# =====================================================================
# CHẾ ĐỘ BẢN ĐỒ (MAP PROFILE):
# - "AUTO"       : Tự động nhận diện theo kích thước map (<= 250m là AC300, > 250m là Ảnh 500x500)
# - "POLYGON_100": Ép cứng bộ thông số cho tập map Đa giác nhỏ (AC300 ~ 100x100)
# - "IMAGE_500"  : Ép cứng bộ thông số cho tập map Ảnh lớn (HUST_MAP / MAP01 ~ 500x500)
# =====================================================================
MAP_PROFILE = "AUTO"


# =====================================================================
# CẤU HÌNH XE & RADAR (CHỌN NHANH BẰNG CÁCH ĐÓNG/MỞ COMMENT KHỐI 1 HOẶC 2)
# =====================================================================

# --- [KHỐI 1] DÀNH CHO TẬP MAP ĐA GIÁC NHỎ (AC300 ~ 100x100) ---
CAR_L = 3.0               # Chiều dài cơ sở nhỏ gọn cho map 100x100
CAR_WIDTH = 1.5           # Chiều rộng xe
MAX_STEER = 0.6           # Góc bẻ lái tối đa (~35-40 độ)
VELOCITY_MAX = 4.0        # Tốc độ tiến vừa phải để nhánh cây không vọt quá dài
VELOCITY_MIN = 2.0        # Tốc độ lùi/cua gắt
DT = 0.2
SIM_TIME = 0.4            # Thời gian mô phỏng 1 bước ngắn (~1.6m/bước)

LOOKAHEAD_STEPS = 5
SENSOR_RADIUS = 15.0      # Bán kính tầm nhìn của Radar trên map AC300
DUBINS_CONNECT_DIST = 15.0
GOAL_RADIUS = 1.0


# --- [KHỐI 2] DÀNH CHO TẬP MAP ẢNH LỚN (HUST_MAP / MAP01 ~ 500x500) ---
# CAR_L = 4.0               # Chiều dài xe thực tế ~4.5 mét
# CAR_WIDTH = 2.0           # Chiều rộng ~2.0 mét
# MAX_STEER = 0.6           # Góc bẻ lái tối đa
# VELOCITY_MAX = 8.0        # Tốc độ tiến (8m/s)
# VELOCITY_MIN = 4.0        # Tốc độ lùi (4m/s)
# DT = 0.2
# SIM_TIME = 0.5
#
# LOOKAHEAD_STEPS = 10
# SENSOR_RADIUS = 50.0      # Bán kính tầm nhìn của Radar trên map 500x500
# DUBINS_CONNECT_DIST = 35.0
# GOAL_RADIUS = 1.5


# =====================================================================
# BẢNG THÔNG SỐ ĐỘNG HỌC MCPP TỰ ĐỘNG NẠP THEO KÍCH THƯỚC BẢN ĐỒ
# =====================================================================
MCPP_PROFILES = {
    # 1. Tự động kích hoạt khi chạy main.py / main_2.py (Map đa giác AC300 <= 250x250)
    "POLYGON_100": {
        "MACRO_STEPS": 3,           # Nhảy 1 bước ngắn để rễ mịn, không nhảy cóc
        "MACRO_SIM_TIME": 0.35,     # Thời gian tích phân mỗi nhánh ngắn
        "LOOKAHEAD_DIST": 15.0,      # Tầm nhìn Carrot 5m bám sát đường cam A*
        "EXPAND_WP_SCAN": 4,        # Quét tối đa 4 waypoint phía trước khi expand
        "SIMQ_WP_SCAN": 4,          # Khoá bước tiến tối đa 4 waypoint (chống nhảy cóc)
        "GRID_RES_XY": 1.0,         # Ô lưới 1.0m cho bản đồ nhỏ
        "GRID_RES_YAW": 10.0,       # Chia góc 10 độ
        "RS_WP_LEFT": 4,            # Còn <= 4 waypoint mới bật đường tím Reeds-Shepp
        "RS_MAX_DIST": 15.0,        # Khoảng cách tối đa gọi Reeds-Shepp (15m)
        "CORRIDOR_WIDTH": 2.5       # Bề rộng hành lang tự do quanh A* (2.5m)
    },

    # 2. Tự động kích hoạt khi chạy main_image.py (Map ảnh 500x500)
    "IMAGE_500": {
        "MACRO_STEPS": 10,           # Bước nhảy vừa phải cho bản đồ rộng 500x500
        "MACRO_SIM_TIME": 0.5,      # Thời gian tích phân tiêu chuẩn
        "LOOKAHEAD_DIST": 20.0,     # Tầm nhìn xa 20m trên đường lớn
        "EXPAND_WP_SCAN": 15,       # Quét tối đa 15 waypoint
        "SIMQ_WP_SCAN": 10,         # Tiến tối đa 10 waypoint mỗi bước
        "GRID_RES_XY": 2.0,         # Ô lưới 2.0m
        "GRID_RES_YAW": 15.0,       # Chia góc 15 độ
        "RS_WP_LEFT": 8,            # Còn <= 8 waypoint bật Reeds-Shepp
        "RS_MAX_DIST": 40.0,        # Khoảng cách gọi Reeds-Shepp (40m)
        "CORRIDOR_WIDTH": 8.0       # Hành lang rộng 8.0m
    }
}

def get_mcpp_profile(bounds):
    """Tự động trả về bộ thông số MCPP chuẩn dựa trên kích thước thực của bản đồ"""
    if MAP_PROFILE in MCPP_PROFILES:
        return MCPP_PROFILES[MAP_PROFILE]
    map_span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    return MCPP_PROFILES["POLYGON_100"] if map_span <= 250.0 else MCPP_PROFILES["IMAGE_500"]


# --- THÔNG SỐ CHƯỚNG NGẠI VẬT ĐỘNG ---
NUM_DYN_OBS = 6       
DYN_OBS_RADIUS = 2.0  
DYN_OBS_SPEED = 12.0  

# --- THÔNG SỐ DUBINS / REEDS-SHEPP ---
MIN_TURN_RADIUS = CAR_L / math.tan(MAX_STEER)
DUBINS_STEP_SIZE = VELOCITY_MAX * DT 

# --- THAM SỐ THUẬT TOÁN RRT ---
RRT_MAX_ITER = 3000   
RRT_GOAL_PROB = 0.1   
PROB_REVERSE = 0.35

# --- THAM SỐ THUẬT TOÁN MCPP ---
MCPP_C = 1.41          
MCPP_ITER = 150       
MCPP_DEPTH = 60       
MCPP_BRANCHES = 3    

BIG_GRID_SIZE = 20.0    
MAX_STEPS_PER_GRID = 25  

# --- TRỌNG SỐ TỐI ƯU GÓC QUỸ ĐẠO (CHỐNG ZICZAC) ---
WEIGHT_TRAJ_ANGLE = 35.0   # Trọng số ép quỹ đạo song song với xương sống A*
WEIGHT_STEER_SMOOTH = 25.0 # Trọng số phạt giật vô lăng Trái <-> Phải liên tục

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