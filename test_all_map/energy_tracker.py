import math
import os
import csv
from datetime import datetime

class EnergyTracker:
    def __init__(self, cost_straight=1.0, cost_turn=1.5, cost_gear_shift=50.0):
        self.COST_STRAIGHT = cost_straight
        self.COST_TURN = cost_turn
        self.COST_GEAR_SHIFT = cost_gear_shift
        
        self.total_dist = 0.0
        self.straight_dist = 0.0
        self.turn_dist = 0.0
        
        self.energy_straight = 0.0
        self.energy_turn = 0.0
        self.energy_gear_shifts = 0.0
        
        self.replan_count = 0
        self.obstacle_detections = 0
        self.gear_shifts_count = 0
        
        # --- CÁC BIẾN LƯU TRỮ ĐỘ MƯỢT (SMOOTHNESS) ---
        self.smoothness_penalty = 0.0  # Tổng bình phương góc bẻ lái (dùng làm hàm cost tối ưu)
        self.total_turn_angle = 0.0    # Tổng góc bẻ lái tuyệt đối (rad)
        self.max_turn_angle = 0.0      # Góc bẻ lái gắt nhất (rad)
        self.movement_steps = 0        # Đếm số bước để tính trung bình
        
        self.last_pos = None
        self.last_yaw = None
        self.last_direction = None
        self.start_time = datetime.now()

    def add_replan(self): 
        self.replan_count += 1

    def add_obstacle(self): 
        self.obstacle_detections += 1

    def update_movement(self, x, y, yaw, direction, is_dubins):
        # 1. Tính quãng đường và năng lượng
        if self.last_pos is not None:
            dist = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
            self.total_dist += dist
            if is_dubins:
                self.turn_dist += dist
                self.energy_turn += dist * self.COST_TURN
            else:
                self.straight_dist += dist
                self.energy_straight += dist * self.COST_STRAIGHT
                
        # 2. TÍNH TOÁN VÀ LƯU THÔNG SỐ ĐỘ MƯỢT
        if self.last_yaw is not None:
            dyaw = abs(yaw - self.last_yaw)
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw)) # Chuẩn hóa chống lỗi 360 độ
            dyaw_abs = abs(dyaw)
            
            self.smoothness_penalty += dyaw_abs**2
            self.total_turn_angle += dyaw_abs
            self.movement_steps += 1
            
            if dyaw_abs > self.max_turn_angle:
                self.max_turn_angle = dyaw_abs
            
        # 3. Phạt năng lượng khi động cơ đảo chiều
        if self.last_direction is not None and direction != self.last_direction:
            self.gear_shifts_count += 1
            self.energy_gear_shifts += self.COST_GEAR_SHIFT
            
        self.last_pos = (x, y)
        self.last_yaw = yaw
        self.last_direction = direction

    def get_total_energy(self):
        return self.energy_straight + self.energy_turn + self.energy_gear_shifts

    def export_report(self, filename="Results/Energy_Report.txt"):
        total_energy = self.get_total_energy()
        duration = (datetime.now() - self.start_time).total_seconds()
        
        # Tính điểm F_S (Average Smoothness) giống công thức trong bài báo
        fs_score = (self.total_turn_angle / self.movement_steps) if self.movement_steps > 0 else 0.0
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(" BÁO CÁO TỔNG HỢP NĂNG LƯỢNG & QUỸ ĐẠO\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Thời gian chạy: {duration:.2f} giây\n")
            f.write(f"Số lần phát hiện chướng ngại vật: {self.obstacle_detections}\n")
            f.write(f"Số lần Replan: {self.replan_count}\n\n")
            
            f.write("--- 1. THỐNG KÊ QUÃNG ĐƯỜNG & NĂNG LƯỢNG ---\n")
            f.write(f"- Đi thẳng: {self.straight_dist:.2f} px | {self.energy_straight:.2f} EU\n")
            f.write(f"- Bẻ lái: {self.turn_dist:.2f} px | {self.energy_turn:.2f} EU\n")
            f.write(f"- Đảo chiều: {self.gear_shifts_count} lần | {self.energy_gear_shifts:.2f} EU\n")
            f.write(f"=> TỔNG NĂNG LƯỢNG: {total_energy:.2f} EU\n\n")
            
            f.write("--- 2. LƯU TRỮ THÔNG SỐ ĐỘ MƯỢT (SMOOTHNESS) ---\n")
            f.write(f"- Tổng góc bẻ lái: {math.degrees(self.total_turn_angle):.2f} độ\n")
            f.write(f"- Góc bẻ lái gắt nhất (Max): {math.degrees(self.max_turn_angle):.2f} độ\n")
            f.write(f"- Hàm phạt bình phương góc (Penalty): {self.smoothness_penalty:.4f} rad^2\n")
            f.write(f"=> Điểm độ mượt trung bình (F_S): {fs_score:.6f} rad/bước\n")

    def export_csv(self, map_name="Unknown_Map", filename="Results/Bao_Cao_Nang_Luong_Tong_Hop.csv"):
        total_energy = self.get_total_energy()
        duration = (datetime.now() - self.start_time).total_seconds()
        fs_score = (self.total_turn_angle / self.movement_steps) if self.movement_steps > 0 else 0.0
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.isfile(filename)
        
        # Cập nhật tên cột (Headers) sang tiếng Việt
        headers = [
            "Tên Bản Đồ", 
            "Thời Gian (s)", 
            "Vật Cản Phát Hiện", 
            "Số Lần Replan", 
            "Tổng Quãng Đường (px)", 
            "Quãng Đường Thẳng (px)", 
            "Quãng Đường Cua/Rẽ (px)",
            "Năng Lượng Đi Thẳng (EU)", 
            "Năng Lượng Cua/Rẽ (EU)", 
            "Số Lần Đảo Chiều", 
            "Năng Lượng Đảo Chiều (EU)", 
            "Tổng Năng Lượng Tiêu Thụ (EU)",
            "Hàm Phạt Độ Mượt (rad^2)", 
            "Góc Rẽ Gắt Nhất (độ)", 
            "Điểm Độ Mượt Trung Bình F_S (rad)"
        ]
        
        row_data = [
            map_name, round(duration, 2), self.obstacle_detections, self.replan_count,
            round(self.total_dist, 2), round(self.straight_dist, 2), round(self.turn_dist, 2),
            round(self.energy_straight, 2), round(self.energy_turn, 2), self.gear_shifts_count,
            round(self.energy_gear_shifts, 2), round(total_energy, 2), 
            round(self.smoothness_penalty, 4),
            round(math.degrees(self.max_turn_angle), 2),
            round(fs_score, 6)
        ]
        
        # Ghi vào file với encoding 'utf-8-sig' để Excel không bị lỗi font tiếng Việt
        with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row_data)