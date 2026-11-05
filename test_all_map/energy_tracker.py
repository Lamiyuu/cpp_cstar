import math
import os
import csv
from datetime import datetime

class EnergyTracker:
    def __init__(self, e_m=1.0, e_r=0.5, e_t_per_rad=0.8):
        # Khởi tạo các hằng số năng lượng
        self.E_M = e_m                # Năng lượng di chuyển giữa 2 waypoint cố định
        self.E_R = e_r                # Năng lượng dừng/nghỉ tại waypoint
        self.E_T_PER_RAD = e_t_per_rad # Năng lượng tiêu hao trên TỪNG RADIAN góc xoay
        
        self.n_waypoints = 0
        self.replan_count = 0
        self.obstacle_detections = 0
        
        # Biến tích lũy năng lượng xoay thực tế
        self.accumulated_turn_energy = 0.0
        
        # --- CÁC BIẾN LƯU TRỮ ĐỘ MƯỢT (SMOOTHNESS) ---
        self.smoothness_penalty = 0.0
        self.total_turn_angle = 0.0
        self.max_turn_angle = 0.0
        self.movement_steps = 0
        
        self.last_pos = None
        self.last_yaw = None
        self.start_time = datetime.now()

    def add_replan(self): 
        self.replan_count += 1

    def add_obstacle(self): 
        self.obstacle_detections += 1

    def add_waypoint(self, x, y, yaw):
        self.n_waypoints += 1
        self.movement_steps += 1
        
        if self.last_yaw is not None:
            # Tính góc lệch chuẩn hóa
            dyaw = abs(yaw - self.last_yaw)
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw)) 
            dyaw_abs = abs(dyaw)
            
            # 1. Tính và cộng dồn năng lượng xoay tỉ lệ thuận với góc (Radian)
            self.accumulated_turn_energy += dyaw_abs * self.E_T_PER_RAD
            
            # 2. Lưu trữ thống kê độ mượt
            self.smoothness_penalty += dyaw_abs**2
            self.total_turn_angle += dyaw_abs
            
            if dyaw_abs > self.max_turn_angle:
                self.max_turn_angle = dyaw_abs
            
        self.last_pos = (x, y)
        self.last_yaw = yaw

    def get_energy_components(self):
        # Năng lượng xoay lấy trực tiếp từ biến cộng dồn thực tế
        energy_turn = self.accumulated_turn_energy
        
        # Năng lượng nghỉ tính theo tổng số waypoint
        energy_rest = self.n_waypoints * self.E_R
        
        # Năng lượng di chuyển tính cho n-1 chặng
        energy_move = (self.n_waypoints - 1) * self.E_M if self.n_waypoints > 1 else 0.0
        
        return energy_move, energy_turn, energy_rest

    def get_total_energy(self):
        e_move, e_turn, e_rest = self.get_energy_components()
        return e_move + e_turn + e_rest

    def export_report(self, filename="Results/Energy_Report.txt"):
        total_energy = self.get_total_energy()
        e_move, e_turn, e_rest = self.get_energy_components()
        duration = (datetime.now() - self.start_time).total_seconds()
        
        fs_score = (self.total_turn_angle / self.movement_steps) if self.movement_steps > 0 else 0.0
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write(" BÁO CÁO TỔNG HỢP NĂNG LƯỢNG (ĐỘNG HỌC GÓC XOAY)\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Thời gian chạy: {duration:.2f} giây\n")
            f.write(f"Số lượng Waypoint (n): {self.n_waypoints}\n")
            f.write(f"Số lần phát hiện chướng ngại vật: {self.obstacle_detections}\n")
            f.write(f"Số lần Replan: {self.replan_count}\n\n")
            
            f.write("--- 1. THỐNG KÊ NĂNG LƯỢNG ---\n")
            f.write(f"- Năng lượng di chuyển: {e_move:.2f} EU\n")
            f.write(f"- Năng lượng xoay (động): {e_turn:.2f} EU\n")
            f.write(f"- Năng lượng dừng/nghỉ: {e_rest:.2f} EU\n")
            f.write(f"=> TỔNG NĂNG LƯỢNG: {total_energy:.2f} EU\n\n")
            
            f.write("--- 2. LƯU TRỮ THÔNG SỐ ĐỘ MƯỢT (SMOOTHNESS) ---\n")
            f.write(f"- Tổng góc bẻ lái: {math.degrees(self.total_turn_angle):.2f} độ\n")
            f.write(f"- Góc bẻ lái gắt nhất (Max): {math.degrees(self.max_turn_angle):.2f} độ\n")
            f.write(f"- Hàm phạt bình phương góc (Penalty): {self.smoothness_penalty:.4f} rad^2\n")
            f.write(f"=> Điểm độ mượt trung bình (F_S): {fs_score:.6f} rad/bước\n")

    def export_csv(self, map_name="Unknown_Map", filename="Results/Bao_Cao_Nang_Luong_Tong_Hop.csv"):
        total_energy = self.get_total_energy()
        e_move, e_turn, e_rest = self.get_energy_components()
        duration = (datetime.now() - self.start_time).total_seconds()
        fs_score = (self.total_turn_angle / self.movement_steps) if self.movement_steps > 0 else 0.0
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_exists = os.path.isfile(filename)
        
        headers = [
            "Tên Bản Đồ", 
            "Thời Gian (s)", 
            "Vật Cản Phát Hiện", 
            "Số Lần Replan", 
            "Số Lượng Waypoint (n)", 
            "Năng Lượng Di Chuyển (EU)", 
            "Năng Lượng Xoay Động (EU)", 
            "Năng Lượng Nghỉ (EU)", 
            "Tổng Năng Lượng Tiêu Thụ (EU)",
            "Hàm Phạt Độ Mượt (rad^2)", 
            "Góc Rẽ Gắt Nhất (độ)", 
            "Điểm Độ Mượt Trung Bình F_S (rad)"
        ]
        
        row_data = [
            map_name, round(duration, 2), self.obstacle_detections, self.replan_count,
            self.n_waypoints, round(e_move, 2), round(e_turn, 2), round(e_rest, 2), 
            round(total_energy, 2), 
            round(self.smoothness_penalty, 4),
            round(math.degrees(self.max_turn_angle), 2),
            round(fs_score, 6)
        ]
        
        with open(filename, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(row_data)