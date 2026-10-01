import math
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

def generate_filtered_pointcloud_map(filepath):
    x_points = []
    y_points = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != 'FLASER':
                    continue
                
                num_readings = int(parts[1])
                readings = [float(val) for val in parts[2 : 2 + num_readings]]
                
                # Trích xuất tọa độ Robot
                idx_pose = 2 + num_readings
                robot_x = float(parts[idx_pose])
                robot_y = float(parts[idx_pose+1])
                robot_theta = float(parts[idx_pose+2])
                
                start_angle = -math.pi / 2
                angle_increment = math.pi / (num_readings - 1)
                
                for i, r in enumerate(readings):
                    # Chỉ lấy các tia chạm tường trong bán kính 30m để giảm nhiễu tầm xa
                    if 0.1 < r < 30.0:
                        global_angle = robot_theta + start_angle + i * angle_increment
                        x_points.append(robot_x + r * math.cos(global_angle))
                        y_points.append(robot_y + r * math.sin(global_angle))
                        
    except Exception as e:
        print(f"Lỗi đọc file: {e}")
        return

    if not x_points:
        print(f"Không tìm thấy dữ liệu hợp lệ trong {filepath}.")
        return

    print(f"[{os.path.basename(filepath)}] Đã nạp {len(x_points)} điểm. Đang lọc nhiễu...")

    # --- BỘ LỌC NHIỄU (DENSITY FILTER) ---
    x_arr = np.array(x_points)
    y_arr = np.array(y_points)
    
    # Chia không gian thành các ô lưới nhỏ 5cm (0.05m)
    RESOLUTION = 0.05
    min_x, max_x = np.min(x_arr), np.max(x_arr)
    min_y, max_y = np.min(y_arr), np.max(y_arr)
    
    width = int((max_x - min_x) / RESOLUTION) + 1
    height = int((max_y - min_y) / RESOLUTION) + 1
    
    # Đếm số lượng điểm rơi vào từng ô
    H, _, _ = np.histogram2d(x_arr, y_arr, bins=(width, height), range=[[min_x, max_x], [min_y, max_y]])
    
    # Tính toán xem mỗi điểm thuộc ô nào
    x_idx = np.clip(((x_arr - min_x) / RESOLUTION).astype(int), 0, width - 1)
    y_idx = np.clip(((y_arr - min_y) / RESOLUTION).astype(int), 0, height - 1)
    
    # Lấy ra những điểm nằm trong ô có ít nhất 3 tia laser đập vào (Ngưỡng lọc = 3)
    THRESHOLD = 3
    valid_mask = H[x_idx, y_idx] >= THRESHOLD
    
    filtered_x = x_arr[valid_mask]
    filtered_y = y_arr[valid_mask]

    # --- TIẾN HÀNH VẼ BẢN ĐỒ ---
    print(f"[{os.path.basename(filepath)}] Đã lọc còn {len(filtered_x)} điểm sắc nét. Đang vẽ...")
    
    plt.figure(figsize=(15, 15), dpi=300)
    plt.style.use('dark_background') 
    
    # Tăng nhẹ alpha lên 0.8 vì các điểm thưa thớt đã bị xóa đi
    plt.scatter(filtered_x, filtered_y, s=0.01, c='cyan', alpha=0.8, marker='.')
    
    plt.axis('equal')
    plt.axis('off') 
    
    # Xuất file
    base_name = os.path.basename(filepath)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}_pointcloud_sharp.png"
    
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Hoàn tất! Đã lưu bản đồ tại: {output_filename}\n")

def main():
    input_dir = 'file_log_map'
    
    # Lấy toàn bộ file .txt trong thư mục
    filepaths = sorted(glob.glob(os.path.join(input_dir, '*.txt')))
    
    if not filepaths:
        print(f"Không tìm thấy file nào trong thư mục '{input_dir}'")
        return
        
    for file in filepaths:
        generate_filtered_pointcloud_map(file)

if __name__ == '__main__':
    main()