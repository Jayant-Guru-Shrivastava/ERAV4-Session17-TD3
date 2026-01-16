import sys
import os
import math
import numpy as np
import random
from collections import deque

# --- PYTORCH ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- TD3 IMPORT ---
# Ensure the current directory is in path to import td3
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from td3 import TD3, ReplayBuffer

# --- PYQT ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGraphicsScene, 
                             QGraphicsView, QGraphicsItem, QFrame, QFileDialog,
                             QTextEdit, QGridLayout)
from PyQt6.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, QPainter, 
                         QPolygonF, QFont, QPainterPath)
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF

# ==========================================
# 1. CONFIGURATION & THEME
# ==========================================
# Nordic Theme
C_BG_DARK   = QColor("#2E3440") 
C_PANEL     = QColor("#3B4252")
C_INFO_BG   = QColor("#4C566A") 
C_ACCENT    = QColor("#88C0D0") 
C_TEXT      = QColor("#ECEFF4") 
C_SUCCESS   = QColor("#A3BE8C") 
C_FAILURE   = QColor("#BF616A") 
C_SENSOR_ON = QColor("#A3BE8C") # Green
C_SENSOR_OFF= QColor("#BF616A") # Red

# Physics Tweaks
CAR_WIDTH = 14     
CAR_HEIGHT = 8   
SENSOR_DIST = 20 # Intermediate distance
SENSOR_ANGLE = 45
MAX_SPEED = 5       
TURN_SPEED = 30 # Max turn angle in degrees

# RL
BATCH_SIZE = 100
MAX_CONSECUTIVE_CRASHES = 10 

# Target Colors
TARGET_COLORS = [
    QColor(0, 255, 255),    QColor(255, 100, 255),    QColor(0, 255, 100),
    QColor(255, 150, 0),    QColor(100, 150, 255),    QColor(255, 50, 150),
    QColor(150, 255, 50),   QColor(255, 255, 0),
]

# ==========================================
# 2. PHYSICS & LOGIC (TD3 AGENT)
# ==========================================
class CarBrain:
    def __init__(self, map_image: QImage):
        self.map = map_image
        self.w, self.h = map_image.width(), map_image.height()
        
        # RL Init
        self.state_dim = 9  # 7 sensors + angle_to_target + distance_to_target
        self.action_dim = 2 # Steering (continuous), Velocity (continuous)
        self.max_action = 1.0
        
        # TD3 Agent
        self.agent = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim)
        
        self.steps = 0
        self.consecutive_crashes = 0
        self.episode_count = 0 # Track total episodes
        self.episode_scores = deque(maxlen=100) # Re-added for tracking
        
        # Locations
        self.start_pos = QPointF(100, 100) 
        self.car_pos = QPointF(100, 100)   
        self.car_angle = 0
        self.target_pos = QPointF(200, 200) 
        
        self.targets = [] 
        self.current_target_idx = 0 
        self.targets_reached = 0 
        
        self.alive = True
        self.score = 0
        self.sensor_coords = [] 
        self.prev_dist = None
        self.last_action = [0, 0] # For display

    def set_start_pos(self, point):
        self.start_pos = point
        self.car_pos = point

    def reset(self):
        self.alive = True
        self.score = 0
        self.car_pos = QPointF(self.start_pos.x(), self.start_pos.y())
        self.car_angle = random.randint(0, 360)
        self.current_target_idx = 0
        self.targets_reached = 0
        if len(self.targets) > 0:
            self.target_pos = self.targets[0]
        state, dist = self.get_state()
        self.prev_dist = dist
        return state
    
    def add_target(self, point):
        self.targets.append(QPointF(point.x(), point.y()))
        if len(self.targets) == 1:
            self.target_pos = self.targets[0]
            self.current_target_idx = 0
    
    def switch_to_next_target(self):
        if self.current_target_idx < len(self.targets) - 1:
            self.current_target_idx += 1
            self.target_pos = self.targets[self.current_target_idx]
            self.targets_reached += 1
            return True 
        return False 

    def get_state(self):
        sensor_vals = []
        self.sensor_coords = []
        # 7 sensors: -45, -30, -15, 0, 15, 30, 45
        angles = [-45, -30, -15, 0, 15, 30, 45]
        
        for a in angles:
            rad = math.radians(self.car_angle + a)
            sx = self.car_pos.x() + math.cos(rad) * SENSOR_DIST
            sy = self.car_pos.y() + math.sin(rad) * SENSOR_DIST
            self.sensor_coords.append(QPointF(sx, sy))
            
            val = 0.0
            if 0 <= sx < self.w and 0 <= sy < self.h:
                c = QColor(self.map.pixel(int(sx), int(sy)))
                brightness = (c.red() + c.green() + c.blue()) / 3.0
                val = brightness / 255.0
            sensor_vals.append(val)
            
        dx = self.target_pos.x() - self.car_pos.x()
        dy = self.target_pos.y() - self.car_pos.y()
        dist = math.sqrt(dx*dx + dy*dy)
        
        rad_to_target = math.atan2(dy, dx)
        angle_to_target = math.degrees(rad_to_target)
        
        angle_diff = (angle_to_target - self.car_angle) % 360
        if angle_diff > 180: angle_diff -= 360
        
        norm_dist = dist / 800.0
        norm_angle = angle_diff / 180.0
        
        state = sensor_vals + [norm_angle, norm_dist]
        return np.array(state, dtype=np.float32), dist

    def step(self, action):
        # Action is [steering, velocity] in range [-1, 1]
        
        # 1. Steering
        # Map [-1, 1] to [-TURN_SPEED, TURN_SPEED]
        steering_action = action[0]
        turn_angle = steering_action * TURN_SPEED
        self.car_angle += turn_angle

        # 2. Velocity
        # Map [-1, 1] to [0, MAX_SPEED]
        # (action + 1) / 2 -> [0, 1] -> * MAX_SPEED
        velocity_action = action[1]
        speed = ((velocity_action + 1) / 2) * MAX_SPEED
        
        # Store for display
        self.last_action = [turn_angle, speed]

        rad = math.radians(self.car_angle)
        new_x = self.car_pos.x() + math.cos(rad) * speed
        new_y = self.car_pos.y() + math.sin(rad) * speed
        self.car_pos = QPointF(new_x, new_y)
        
        next_state, dist = self.get_state()
        
        reward = -0.1 # Living penalty
        done = False
        
        car_center_val = self.check_pixel(self.car_pos.x(), self.car_pos.y())
        
        if car_center_val < 0.4:
            reward = -100
            done = True
            self.alive = False
        elif dist < 20: 
            reward = 100
            if self.switch_to_next_target():
                done = False
                _, new_dist = self.get_state()
                self.prev_dist = new_dist
            else:
                done = True
        else:
            if self.prev_dist is not None:
                # Distance Reward: +5 for closer, -5 for farther
                diff = self.prev_dist - dist
                if diff > 0:
                    reward += 5.0
                else:
                    reward += -5.0
            self.prev_dist = dist
            
        self.score += reward
        return next_state, reward, done

    def check_pixel(self, x, y):
        if 0 <= x < self.w and 0 <= y < self.h:
            c = QColor(self.map.pixel(int(x), int(y)))
            return ((c.red() + c.green() + c.blue()) / 3.0) / 255.0
        return 0.0

    def optimize(self):
        if self.replay_buffer.size < BATCH_SIZE: return
        self.agent.train(self.replay_buffer, BATCH_SIZE)

    def finalize_episode(self, score):
        self.episode_scores.append(score)
        self.episode_count += 1

        
# ==========================================
# 3. CUSTOM WIDGETS
# ==========================================
class RewardChart(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        self.scores = []
        self.max_points = 50

    def update_chart(self, new_score):
        self.scores.append(new_score)
        if len(self.scores) > self.max_points:
            self.scores.pop(0)
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width(); h = self.height()
        painter.fillRect(0, 0, w, h, C_PANEL)
        if len(self.scores) < 2: return
        min_val = min(self.scores); max_val = max(self.scores)
        if max_val == min_val: max_val += 1
        points = []
        step_x = w / (self.max_points - 1)
        for i, score in enumerate(self.scores):
            x = i * step_x
            ratio = (score - min_val) / (max_val - min_val)
            y = h - (ratio * (h * 0.8) + (h * 0.1))
            points.append(QPointF(x, y))
        path = QPainterPath(); path.moveTo(points[0])
        for p in points[1:]: path.lineTo(p)
        painter.setPen(QPen(C_ACCENT, 2)); painter.drawPath(path)

class SensorItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(90)
        self.is_detecting = True 
    def set_detecting(self, detecting):
        self.is_detecting = detecting
        self.update()
    def boundingRect(self):
        return QRectF(-3, -3, 6, 6)
    def paint(self, painter, option, widget):
        color = C_SENSOR_ON if self.is_detecting else C_SENSOR_OFF
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(0, 0), 2, 2)

class CarItem(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.setZValue(100)
        self.brush = QBrush(C_ACCENT)
        self.pen = QPen(Qt.GlobalColor.white, 1)
    def boundingRect(self):
        return QRectF(-CAR_WIDTH/2, -CAR_HEIGHT/2, CAR_WIDTH, CAR_HEIGHT)
    def paint(self, painter, option, widget):
        painter.setBrush(self.brush); painter.setPen(self.pen)
        painter.drawRoundedRect(self.boundingRect(), 2, 2)
        painter.setBrush(Qt.GlobalColor.white)
        painter.drawRect(int(CAR_WIDTH/2)-2, -3, 2, 6)

class TargetItem(QGraphicsItem):
    def __init__(self, color=None, is_active=True, number=1):
        super().__init__()
        self.setZValue(50)
        self.color = color if color else QColor(0, 255, 255)
        self.is_active = is_active
        self.number = number
    def boundingRect(self): return QRectF(-20, -20, 40, 40)
    def paint(self, painter, option, widget):
        alpha = 255 if self.is_active else 100
        c = QColor(self.color); c.setAlpha(alpha)
        
        # Larger circle
        r = 12
        painter.setBrush(QBrush(c)); painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.drawEllipse(QPointF(0,0), r, r)
        
        # Centered Text
        painter.setPen(QPen(Qt.GlobalColor.black)) # Black text for contrast
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        # Draw text in a rect centered on 0,0
        painter.drawText(QRectF(-r, -r, 2*r, 2*r), Qt.AlignmentFlag.AlignCenter, str(self.number))

# ==========================================
# 4. APP
# ==========================================
class NeuralNavApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuralNav: TD3 Agent")
        self.resize(1300, 850)
        
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # LEFT PANEL
        panel = QFrame(); panel.setFixedWidth(280)
        panel.setStyleSheet(f"background-color: {C_BG_DARK.name()};")
        vbox = QVBoxLayout(panel)
        
        self.lbl_status = QLabel("1. Click Map -> CAR\n2. Click Map -> TARGET(S)")
        self.lbl_status.setStyleSheet("background-color: #4C566A; padding: 10px; color: white;")
        vbox.addWidget(self.lbl_status)
        
        self.btn_run = QPushButton("▶ START (Space)")
        self.btn_run.setCheckable(True); self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.toggle_training)
        vbox.addWidget(self.btn_run)
        
        self.btn_reset = QPushButton("↺ RESET ALL")
        self.btn_reset.clicked.connect(self.full_reset)
        vbox.addWidget(self.btn_reset)
        
        self.btn_load = QPushButton("📂 LOAD MAP")
        self.btn_load.clicked.connect(self.load_map_dialog)
        vbox.addWidget(self.btn_load)
        
        self.btn_save_brain = QPushButton("💾 SAVE BRAIN")
        self.btn_save_brain.clicked.connect(self.save_brain)
        vbox.addWidget(self.btn_save_brain)
        
        self.btn_load_brain = QPushButton("📂 LOAD BRAIN")
        self.btn_load_brain.clicked.connect(self.load_brain)
        vbox.addWidget(self.btn_load_brain)
        
        self.chart = RewardChart(); vbox.addWidget(self.chart)
        
        # Stats Grid
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"background-color: {C_PANEL.name()}; border-radius: 5px;")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 10, 10, 10)
        
        self.val_steps = QLabel("0")
        self.val_steps.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Steps:"), 0,0)
        sf_layout.addWidget(self.val_steps, 0,1)
        
        self.val_episodes = QLabel("0")
        self.val_episodes.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Episodes:"), 1,0)
        sf_layout.addWidget(self.val_episodes, 1,1)
        
        self.val_action = QLabel("0.0, 0.0")
        self.val_action.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Act (St, V):"), 2,0)
        sf_layout.addWidget(self.val_action, 2,1)
        
        self.val_rew = QLabel("0")
        self.val_rew.setStyleSheet(f"color: {C_ACCENT.name()}; font-weight: bold;")
        sf_layout.addWidget(QLabel("Last Reward:"), 3,0)
        sf_layout.addWidget(self.val_rew, 3,1)
        
        vbox.addWidget(stats_frame)
        
        self.log_console = QTextEdit(); self.log_console.setReadOnly(True)
        vbox.addWidget(self.log_console)
        main_layout.addWidget(panel)
        
        # RIGHT PANEL
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.mousePressEvent = self.on_scene_click
        main_layout.addWidget(self.view)
        
        # Use map from the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        map_path = os.path.join(script_dir, "city_map.png")
        self.setup_map(map_path)
        self.setup_state = 0 
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.game_loop)
        
        self.car_item = CarItem()
        self.target_items = []
        self.sensor_items = []
        for _ in range(7):
            si = SensorItem(); self.scene.addItem(si); self.sensor_items.append(si)

    def log(self, msg): self.log_console.append(msg)

    def setup_map(self, path):
        if not os.path.exists(path): self.create_dummy_map(path)
        self.map_img = QImage(path).convertToFormat(QImage.Format.Format_RGB32)
        self.scene.clear(); self.scene.addPixmap(QPixmap.fromImage(self.map_img))
        self.brain = CarBrain(self.map_img)
        self.log(f"Map Loaded (TD3 Agent).")

    def create_dummy_map(self, path):
        img = QImage(1000, 800, QImage.Format.Format_RGB32); img.fill(C_BG_DARK)
        img.save(path)

    def load_map_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load Map", "", "Images (*.png *.jpg)")
        if f: self.full_reset(); self.setup_map(f)

    def on_scene_click(self, event):
        pt = self.view.mapToScene(event.pos())
        if self.setup_state == 0:
            self.brain.set_start_pos(pt); self.scene.addItem(self.car_item); self.car_item.setPos(pt)
            self.setup_state = 1; self.lbl_status.setText("Click Map -> TARGET(S)")
        elif self.setup_state == 1:
            if event.button() == Qt.MouseButton.LeftButton:
                self.brain.add_target(pt)
                ti = TargetItem(TARGET_COLORS[len(self.brain.targets)%8], True, len(self.brain.targets))
                ti.setPos(pt); self.scene.addItem(ti); self.target_items.append(ti)
                self.log(f"Target added.")
            elif event.button() == Qt.MouseButton.RightButton and len(self.brain.targets)>0:
                self.setup_state = 2; self.lbl_status.setText("READY. Press SPACE.")
                self.btn_run.setEnabled(True)

    def full_reset(self):
        self.sim_timer.stop(); self.btn_run.setChecked(False); self.btn_run.setEnabled(False)
        self.setup_state = 0; self.scene.removeItem(self.car_item)
        for t in self.target_items: self.scene.removeItem(t)
        self.target_items = []
        self.brain.targets = []; self.brain.current_target_idx = 0
        self.chart.scores = []; self.chart.update()
        self.log("--- RESET ---")

    def toggle_training(self):
        if self.btn_run.isChecked(): self.sim_timer.start(16); self.btn_run.setText("⏸ PAUSE")
        else: self.sim_timer.stop(); self.btn_run.setText("▶ RESUME")

    def keyPressEvent(self, f):
        if f.key() == Qt.Key.Key_Space and self.setup_state == 2: self.btn_run.click()

    def save_brain(self):
        self.brain.agent.save("td3_brain")
        self.log("Saved TD3 Brain.")
        
    def load_brain(self):
        # Implement load if needed
        self.brain.agent.load("td3_brain")
        self.log("Loaded TD3 Brain.")

    def game_loop(self):
        if self.setup_state != 2: return
        
        state, _ = self.brain.get_state()
        
        # Select action with noise
        # WARMUP PHASE: Pure random exploration to fill buffer
        if self.brain.steps < 1000:
            action = np.random.uniform(-1, 1, size=2)
        else:
            # Select action with noise
            action = self.brain.agent.select_action(np.array(state))
            # Add exploration noise
            noise = np.random.normal(0, 0.1, size=2)
            action = (action + noise).clip(-1, 1)
        
        next_s, rew, done = self.brain.step(action)
        self.brain.steps += 1
        
        self.brain.replay_buffer.add(state, action, next_s, rew, done)
        
        # Only train after warmup
        if self.brain.steps >= 1000:
            self.brain.optimize()
        
        # Update Visuals
        self.car_item.setPos(self.brain.car_pos)
        self.car_item.setRotation(self.brain.car_angle)
        
        # Update Sensors
        for i, coord in enumerate(self.brain.sensor_coords):
            self.sensor_items[i].setPos(coord)
            # Visualize detection (simple threshold check for visual)
            val = next_s[i] # 0-1 brightness
            self.sensor_items[i].set_detecting(val > 0.4) 

        # Update Targets
        for i, t_item in enumerate(self.target_items):
            t_item.is_active = (i == self.brain.current_target_idx)
            t_item.update()

        if rew >= 90: # Check for target reward (approx 100)
            if not done:
                # Intermediate target reached
                current_goal = self.brain.targets_reached
                msg = (f"<font color='#A3BE8C'><b>🏆 GOAL {current_goal} REACHED!</b></font> "
                       f"Steps: {self.brain.steps}")
                self.log(msg)
        
        self.val_rew.setText(f"{self.brain.score:.1f}")
        self.val_steps.setText(f"{self.brain.steps}")
        self.val_episodes.setText(f"{self.brain.episode_count}")
        # Show Steering (deg) and Speed
        st, sp = self.brain.last_action
        self.val_action.setText(f"{st:.1f}°, {sp:.1f}")

        if done:
            # Detailed Logging matching previous assignment style
            mem_size = self.brain.replay_buffer.size
            if not self.brain.alive:
                txt = "CRASH"
                col = "#BF616A" # Red
            else:
                txt = "ALL TARGETS COMPLETED"
                col = "#A3BE8C" # Green
                
            msg = (f"<font color='{col}'><b>{txt}</b> (Scr: {self.brain.score:.0f}) | "
                   f"Steps: {self.brain.steps} | "
                   f"Mem: {mem_size}</font>")
            self.log(msg)
            
            self.brain.finalize_episode(self.brain.score) 
            self.chart.update_chart(self.brain.score)
            
            if not self.brain.alive:
                self.brain.consecutive_crashes += 1
                if self.brain.consecutive_crashes % 10 == 0:
                    self.log(f"<font color='#EBCB8B'>⚠️ {self.brain.consecutive_crashes} crashes in a row...</font>")
            else:
                self.brain.consecutive_crashes = 0
                
            self.brain.reset()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeuralNavApp()
    window.show()
    sys.exit(app.exec())