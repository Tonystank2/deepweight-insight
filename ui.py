import cv2
import numpy as np
import time
import sys
import subprocess
import os
from collections import deque
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
                               QGroupBox, QGridLayout, QComboBox, QFileDialog,
                               QSplitter, QFrame, QScrollArea, QMessageBox, QSizePolicy,
                               QDockWidget, QTabWidget, QCheckBox, QDoubleSpinBox)
from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QImage, QPixmap, QPalette, QColor, QFont, QWheelEvent, QPainter, QPen, QBrush

# === CONFIG ===
VIDEO_PATH = "your_video.mp4"
OUTPUT_PATH = "barbell_tracked.mp4"
TRAIL_MAX_POINTS = 300
SPEED_VIEW_SCALE = 2

# === Premium Color Scheme (BGR) ===
class Theme:
    PRIMARY = (82, 196, 26)
    PRIMARY_DIM = (65, 157, 20)
    ACCENT = (255, 159, 28)
    DANGER = (45, 84, 255)
    TEXT = (255, 255, 255)
    TEXT_SECONDARY = (180, 180, 180)
    TEXT_DIM = (120, 120, 120)
    BG_DARK = (20, 20, 22)
    BG_PANEL = (35, 35, 38)
    BG_CONTROL = (45, 45, 48)
    OVERLAY_BG = (25, 25, 28)
    TRAIL_START = (255, 100, 50)
    TRAIL_END = (100, 255, 200)
    SHADOW = (0, 0, 0)
    
    SELECTION_COLORS = [
        (82, 196, 26), (255, 159, 28), (255, 71, 255), (0, 255, 255),
        (45, 84, 255), (147, 20, 255), (0, 255, 127), (255, 191, 0),
    ]

# === Sensitivity Settings ===
class SensitivitySettings:
    def __init__(self):
        self.speed_low_threshold = 1
        self.speed_high_threshold = 100.0
        self.acc_low_threshold = 50.0
        self.acc_high_threshold = 500.0
        self.dwell_threshold = 10
        self.dwell_high_threshold = 30

# === Selection class ===
class Selection:
    def __init__(self, start_frame, end_frame, color_idx):
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.color = Theme.SELECTION_COLORS[color_idx % len(Theme.SELECTION_COLORS)]
        self.color_idx = color_idx
    
    def contains_frame(self, frame_idx):
        return self.start_frame <= frame_idx <= self.end_frame

# === State ===
class AppState:
    def __init__(self):
        self.frame = None
        self.frame_idx = 0
        self.paused = True
        self.bbox = None
        self.tracker = None
        self.tracking_active = False
        self.path_by_frame = []
        self.path_points = deque(maxlen=TRAIL_MAX_POINTS)
        self.recording = False
        self.drawing = False
        self.draw_start = None
        self.draw_current = None
        self.fps_display = 25.0
        self.last_time = time.time()
        self.selections = []
        self.current_selection_idx = 0
        self.selecting = False
        self.select_start_progress = 0.0
        self.select_current_progress = 0.0
        self.viz_mode = 1
        self.total_frames = 0
        self.video_fps = 25.0
        self.W = 0
        self.H = 0
        self.sensitivity = SensitivitySettings()

# === Tracker ===
def create_tracker():
    """Create a tracker using the best available method"""
    # Try modern TrackerCSRT first (OpenCV 4.5.1+)
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        pass
    
    # Try legacy TrackerCSRT (OpenCV 4.5.0 and below)
    try:
        return cv2.legacy.TrackerCSRT_create()
    except (AttributeError, cv2.error):
        pass
    
    # Try TrackerKCF
    try:
        return cv2.TrackerKCF_create()
    except AttributeError:
        pass
    
    # Try legacy TrackerKCF
    try:
        return cv2.legacy.TrackerKCF_create()
    except (AttributeError, cv2.error):
        pass
    
    # Try other tracker types
    tracker_types = [
        ('TrackerMIL_create', 'MIL'),
        ('TrackerBoosting_create', 'Boosting'),
        ('TrackerMedianFlow_create', 'MedianFlow'),
        ('TrackerTLD_create', 'TLD'),
        ('TrackerMOSSE_create', 'MOSSE')
    ]
    
    for tracker_func, name in tracker_types:
        # Try modern API
        try:
            func = getattr(cv2, tracker_func)
            return func()
        except (AttributeError, cv2.error):
            pass
        
        # Try legacy API
        try:
            func = getattr(cv2.legacy, tracker_func)
            return func()
        except (AttributeError, cv2.error):
            pass
    
    return None

# === Drawing Utilities ===
def draw_glow(img, center, radius, color, intensity=0.5):
    overlay = img.copy()
    for i in range(3, 0, -1):
        alpha = intensity / (i * 1.5)
        cv2.circle(overlay, center, radius + i * 8, color, -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_text(img, text, pos, size=0.6, color=Theme.TEXT, shadow=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    x, y = pos
    if shadow:
        cv2.putText(img, text, (x + 2, y + 2), font, size, Theme.SHADOW, thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, size, color, thickness, cv2.LINE_AA)

def get_selection_color(sel_idx):
    if sel_idx == -1:
        return Theme.PRIMARY
    return Theme.SELECTION_COLORS[sel_idx % len(Theme.SELECTION_COLORS)]

def draw_trail_colored(img, points_with_sel, thickness=2):
    if not points_with_sel:
        return
    
    segments = []
    current_segment = []
    current_sel_idx = None
    
    for item in points_with_sel:
        if item is None:
            if current_segment:
                segments.append((current_segment, current_sel_idx))
                current_segment = []
                current_sel_idx = None
            continue
        
        pos, sel_idx = item
        if sel_idx != current_sel_idx:
            if current_segment:
                segments.append((current_segment, current_sel_idx))
            current_segment = [pos]
            current_sel_idx = sel_idx
        else:
            current_segment.append(pos)
    
    if current_segment:
        segments.append((current_segment, current_sel_idx))
    
    for segment_points, sel_idx in segments:
        if len(segment_points) < 2:
            if len(segment_points) == 1:
                pt = tuple(map(int, segment_points[0]))
                color = get_selection_color(sel_idx)
                cv2.circle(img, pt, 6, color, -1, lineType=cv2.LINE_AA)
            continue
        
        base_color = get_selection_color(sel_idx)
        n = len(segment_points)
        for i in range(1, n):
            pt1 = tuple(map(int, segment_points[i - 1]))
            pt2 = tuple(map(int, segment_points[i]))
            alpha = i / n
            color = tuple(int(base_color[j] * (0.6 + 0.4 * alpha)) for j in range(3))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        
        last_pt = tuple(map(int, segment_points[-1]))
        draw_glow(img, last_pt, 10, base_color, 0.5)
        cv2.circle(img, last_pt, 6, base_color, -1, lineType=cv2.LINE_AA)

def draw_bbox(img, bbox, color=Theme.PRIMARY, thickness=2):
    if bbox is None:
        return
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)
    draw_glow(img, (x + w // 2, y + h // 2), 12, color, 0.3)

def speed_to_heatmap_color(ratio, sensitivity):
    if ratio < 0.33:
        t = ratio / 0.33
        r = int(255 * t)
        g = 255
        b = 0
    elif ratio < 0.66:
        t = (ratio - 0.33) / 0.33
        r = 255
        g = int(255 - 128 * t)
        b = 0
    else:
        t = (ratio - 0.66) / 0.34
        r = 255
        g = int(127 - 127 * t)
        b = 0
    return (b, g, r)

def acceleration_to_heatmap_color(ratio, sensitivity):
    if ratio < 0.25:
        t = ratio / 0.25
        r = 0
        g = int(255 - 255 * t)
        b = 255
    elif ratio < 0.5:
        t = (ratio - 0.25) / 0.25
        r = int(255 * t)
        g = int(255 * t)
        b = int(255 - 255 * t)
    elif ratio < 0.75:
        t = (ratio - 0.5) / 0.25
        r = 255
        g = int(255 - 128 * t)
        b = 0
    else:
        t = (ratio - 0.75) / 0.25
        r = 255
        g = int(127 - 127 * t)
        b = 0
    return (b, g, r)

def dwell_to_heatmap_color(ratio, sensitivity):
    if ratio < 0.33:
        t = ratio / 0.33
        r = int(255 * t)
        g = 255
        b = 0
    elif ratio < 0.66:
        t = (ratio - 0.33) / 0.33
        r = 255
        g = int(255 - 128 * t)
        b = 0
    else:
        t = (ratio - 0.66) / 0.34
        r = 255
        g = int(127 - 127 * t)
        b = 0
    return (b, g, r)

def calculate_metrics(points_with_sel, fps, sensitivity):
    if not points_with_sel or len(points_with_sel) < 2:
        return None
    
    pts = [item[0] for item in points_with_sel if item is not None]
    if len(pts) < 2:
        return None
    
    speeds = []
    velocities = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        dist = np.sqrt(dx*dx + dy*dy)
        speeds.append(dist * fps)
        velocities.append((dx * fps, dy * fps))
    
    accs = []
    for i in range(1, len(velocities)):
        dvx = velocities[i][0] - velocities[i-1][0]
        dvy = velocities[i][1] - velocities[i-1][1]
        acc = np.sqrt(dvx*dvx + dvy*dvy) * fps
        accs.append(acc)
    
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    
    dwell_times = []
    current_dwell = 0
    for speed in speeds:
        if speed < sensitivity.dwell_threshold:
            current_dwell += 1
        else:
            current_dwell = 0
        dwell_times.append(current_dwell)
    
    metrics = {
        'total_distance': sum([np.sqrt((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2) 
                              for i in range(1, len(pts))]),
        'vertical_rom': max(ys) - min(ys) if ys else 0,
        'horizontal_sway': max(xs) - min(xs) if xs else 0,
        'avg_speed': np.mean(speeds) if speeds else 0,
        'max_speed': max(speeds) if speeds else 0,
        'min_speed': min(speeds) if speeds else 0,
        'avg_acceleration': np.mean(accs) if accs else 0,
        'max_acceleration': max(accs) if accs else 0,
        'max_dwell': max(dwell_times) if dwell_times else 0,
        'total_frames': len(pts),
        'duration': len(pts) / fps if fps > 0 else 0,
        'speeds': speeds,
        'accelerations': accs,
        'dwell_times': dwell_times
    }
    
    return metrics

def draw_analytics_speed_view(points_with_sel, viz_mode, W, H, FPS, sensitivity, scale=SPEED_VIEW_SCALE):
    if points_with_sel is None:
        return np.zeros((H, W, 3), dtype=np.uint8)
    
    hr_w, hr_h = W * scale, H * scale
    hr = np.zeros((hr_h, hr_w, 3), dtype=np.uint8)

    grid_size = 50 * scale
    for x in range(0, hr_w, int(grid_size)):
        cv2.line(hr, (x, 0), (x, hr_h), (40, 40, 40), 1)
    for y in range(0, hr_h, int(grid_size)):
        cv2.line(hr, (0, y), (hr_w, y), (40, 40, 40), 1)
    
    cv2.line(hr, (0, hr_h//2), (hr_w, hr_h//2), (80, 80, 80), 2)
    cv2.line(hr, (hr_w//2, 0), (hr_w//2, hr_h), (80, 80, 80), 2)

    selection_groups = {}
    for item in points_with_sel:
        if item is None:
            continue
        pos, sel_idx = item
        if sel_idx not in selection_groups:
            selection_groups[sel_idx] = []
        selection_groups[sel_idx].append(pos)

    if not selection_groups:
        return cv2.resize(hr, (W, H), interpolation=cv2.INTER_AREA)

    for sel_idx, pts in selection_groups.items():
        if len(pts) < 2:
            if len(pts) == 1:
                pt = (int(pts[0][0] * scale), int(pts[0][1] * scale))
                color = get_selection_color(sel_idx)
                cv2.circle(hr, pt, 8 * scale, color, -1, lineType=cv2.LINE_AA)
            continue

        speeds = []
        accs = []
        dwell_times = []
        velocities = []
        
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            dist = np.sqrt(dx * dx + dy * dy)
            speeds.append(dist * FPS)
            vx = dx * FPS
            vy = dy * FPS
            velocities.append((vx, vy))
        
        for i in range(1, len(velocities)):
            dvx = velocities[i][0] - velocities[i - 1][0]
            dvy = velocities[i][1] - velocities[i - 1][1]
            acc = np.sqrt(dvx * dvx + dvy * dvy) * FPS
            accs.append(acc)
        
        current_dwell = 0
        for speed in speeds:
            if speed < sensitivity.dwell_threshold:
                current_dwell += 1
            else:
                current_dwell = 0
            dwell_times.append(current_dwell)

        if viz_mode == 1:
            max_speed_ref = sensitivity.speed_high_threshold
            for i in range(1, len(pts)):
                p1 = (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale))
                p2 = (int(pts[i][0] * scale), int(pts[i][1] * scale))
                ratio = min(speeds[i-1] / max_speed_ref, 1.0) if max_speed_ref > 0 else 0
                color = speed_to_heatmap_color(ratio, sensitivity)
                cv2.line(hr, p1, p2, color, 8 * scale, cv2.LINE_AA)
        
        elif viz_mode == 2:
            max_acc_ref = sensitivity.acc_high_threshold
            for i in range(2, len(pts)):
                p1 = (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale))
                p2 = (int(pts[i][0] * scale), int(pts[i][1] * scale))
                acc_val = accs[i - 2] if (i - 2) < len(accs) else 0
                ratio = min(acc_val / max_acc_ref, 1.0) if max_acc_ref > 0 else 0
                color = acceleration_to_heatmap_color(ratio, sensitivity)
                cv2.line(hr, p1, p2, color, 8 * scale, cv2.LINE_AA)
        
        elif viz_mode == 3:
            max_dwell_ref = sensitivity.dwell_high_threshold
            for i in range(1, len(pts)):
                p1 = (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale))
                p2 = (int(pts[i][0] * scale), int(pts[i][1] * scale))
                ratio = min(dwell_times[i-1] / max_dwell_ref, 1.0) if max_dwell_ref > 0 else 0
                color = dwell_to_heatmap_color(ratio, sensitivity)
                cv2.line(hr, p1, p2, color, 8 * scale, cv2.LINE_AA)

        base_color = get_selection_color(sel_idx)
        last = (int(pts[-1][0] * scale), int(pts[-1][1] * scale))
        draw_glow(hr, last, 12 * scale, base_color, 0.8)
        cv2.circle(hr, last, 7 * scale, base_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(hr, last, 4 * scale, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        
        first = (int(pts[0][0] * scale), int(pts[0][1] * scale))
        cv2.circle(hr, first, 6 * scale, base_color, 2 * scale, lineType=cv2.LINE_AA)

    out_img = cv2.resize(hr, (W, H), interpolation=cv2.INTER_AREA)
    return out_img

# === Collapsible Group Box ===
class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self.on_toggled)
        self.animation = QPropertyAnimation(self, b"maximumHeight")
        self.animation.setDuration(300)
        self.animation.setEasingCurve(QEasingCurve.OutCubic)
        
    def on_toggled(self, checked):
        if checked:
            self.animation.setStartValue(0)
            self.animation.setEndValue(16777215)
        else:
            self.animation.setStartValue(self.height())
            self.animation.setEndValue(0)
        self.animation.start()

# === Zoomable Video Label ===
class ZoomableVideoLabel(QLabel):
    mouse_pressed = Signal(int, int, Qt.MouseButton)
    mouse_moved = Signal(int, int)
    mouse_released = Signal(int, int, Qt.MouseButton)
    
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1c; border: 2px solid #2d2d2f;")
        self.setMouseTracking(True)
        self.zoom_level = 1.0
        self.original_pixmap = None
        
    def set_image(self, pixmap):
        self.original_pixmap = pixmap
        self.update_zoom()
        
    def update_zoom(self):
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                int(self.width() * self.zoom_level),
                int(self.height() * self.zoom_level),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(scaled)
            
    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom_level = min(self.zoom_level * 1.1, 5.0)
        else:
            self.zoom_level = max(self.zoom_level / 1.1, 0.5)
        self.update_zoom()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_zoom()
        
    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        self.mouse_pressed.emit(pos.x(), pos.y(), event.button())
        
    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        self.mouse_moved.emit(pos.x(), pos.y())
        
    def mouseReleaseEvent(self, event):
        pos = event.position().toPoint()
        self.mouse_released.emit(pos.x(), pos.y(), event.button())

# === Custom Timeline Widget ===
class TimelineWidget(QWidget):
    value_changed = Signal(int)
    selection_started = Signal(float)
    selection_updated = Signal(float)
    selection_finished = Signal(float, float)
    
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(60)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.maximum = 100
        self.value = 0
        self.selections = []
        self.selecting = False
        self.select_start = 0.0
        self.select_current = 0.0
        self.dragging = False
        self.setMouseTracking(True)
        
    def set_maximum(self, maximum):
        self.maximum = max(1, maximum)
        self.update()
        
    def set_value(self, value):
        self.value = max(0, min(value, self.maximum))
        self.update()
        
    def set_selections(self, selections):
        self.selections = selections
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.fillRect(self.rect(), QColor(35, 35, 38))
        
        margin = 60
        bar_y = self.height() // 2
        bar_h = 10
        bar_w = self.width() - 2 * margin
        
        painter.setBrush(QBrush(QColor(45, 45, 48)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(margin, bar_y - bar_h // 2, bar_w, bar_h, 5, 5)
        
        for sel in self.selections:
            if self.maximum > 1:
                s_prog = sel.start_frame / max(1, self.maximum)
                e_prog = sel.end_frame / max(1, self.maximum)
                left = margin + int(bar_w * s_prog)
                right = margin + int(bar_w * e_prog)
                
                color = QColor(sel.color[2], sel.color[1], sel.color[0], 90)
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(sel.color[2], sel.color[1], sel.color[0]), 2))
                painter.drawRoundedRect(left, bar_y - 15, right - left, 30, 3, 3)
                
                painter.setBrush(QBrush(QColor(sel.color[2], sel.color[1], sel.color[0])))
                painter.drawEllipse(left - 5, bar_y - 5, 10, 10)
                painter.drawEllipse(right - 5, bar_y - 5, 10, 10)
        
        if self.selecting:
            s_prog = min(self.select_start, self.select_current)
            e_prog = max(self.select_start, self.select_current)
            left = margin + int(bar_w * s_prog)
            right = margin + int(bar_w * e_prog)
            
            color = QColor(82, 196, 26, 60)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(82, 196, 26), 2))
            painter.drawRoundedRect(left, bar_y - 15, right - left, 30, 3, 3)
            painter.drawEllipse(left - 5, bar_y - 5, 10, 10)
            painter.drawEllipse(right - 5, bar_y - 5, 10, 10)
        
        if self.maximum > 1:
            progress = self.value / self.maximum
            fill_w = int(bar_w * progress)
            painter.setBrush(QBrush(QColor(82, 196, 26)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(margin, bar_y - bar_h // 2, fill_w, bar_h, 5, 5)
            
            head_x = margin + fill_w
            painter.setBrush(QBrush(QColor(82, 196, 26)))
            painter.drawEllipse(head_x - 8, bar_y - 8, 16, 16)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(head_x - 4, bar_y - 4, 8, 8)
    
    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        margin = 60
        bar_w = self.width() - 2 * margin
        
        if event.button() == Qt.LeftButton:
            self.dragging = True
            progress = max(0.0, min(1.0, (pos.x() - margin) / bar_w))
            new_value = int(progress * self.maximum)
            self.value = new_value
            self.value_changed.emit(new_value)
            self.update()
            
        elif event.button() == Qt.RightButton:
            self.selecting = True
            progress = max(0.0, min(1.0, (pos.x() - margin) / bar_w))
            self.select_start = progress
            self.select_current = progress
            self.selection_started.emit(progress)
            self.update()
    
    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        margin = 60
        bar_w = self.width() - 2 * margin
        progress = max(0.0, min(1.0, (pos.x() - margin) / bar_w))
        
        if self.dragging:
            new_value = int(progress * self.maximum)
            self.value = new_value
            self.value_changed.emit(new_value)
            self.update()
            
        elif self.selecting:
            self.select_current = progress
            self.selection_updated.emit(progress)
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            
        elif event.button() == Qt.RightButton and self.selecting:
            self.selecting = False
            margin = 60
            bar_w = self.width() - 2 * margin
            pos = event.position().toPoint()
            progress = max(0.0, min(1.0, (pos.x() - margin) / bar_w))
            self.select_current = progress
            self.selection_finished.emit(self.select_start, self.select_current)
            self.update()

# === Sensitivity Controls Widget ===
class SensitivityControls(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        speed_group = QGroupBox("Speed Sensitivity")
        speed_layout = QGridLayout()
        
        speed_layout.addWidget(QLabel("Low Threshold:"), 0, 0)
        self.speed_low_spin = QDoubleSpinBox()
        self.speed_low_spin.setRange(0, 999999)
        self.speed_low_spin.setValue(self.state.sensitivity.speed_low_threshold)
        self.speed_low_spin.valueChanged.connect(self.on_speed_low_changed)
        speed_layout.addWidget(self.speed_low_spin, 0, 1)
        
        speed_layout.addWidget(QLabel("High Threshold:"), 1, 0)
        self.speed_high_spin = QDoubleSpinBox()
        self.speed_high_spin.setRange(0, 999999)
        self.speed_high_spin.setValue(self.state.sensitivity.speed_high_threshold)
        self.speed_high_spin.valueChanged.connect(self.on_speed_high_changed)
        speed_layout.addWidget(self.speed_high_spin, 1, 1)
        
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        acc_group = QGroupBox("Acceleration Sensitivity")
        acc_layout = QGridLayout()
        
        acc_layout.addWidget(QLabel("Low Threshold:"), 0, 0)
        self.acc_low_spin = QDoubleSpinBox()
        self.acc_low_spin.setRange(0, 999999)
        self.acc_low_spin.setValue(self.state.sensitivity.acc_low_threshold)
        self.acc_low_spin.valueChanged.connect(self.on_acc_low_changed)
        acc_layout.addWidget(self.acc_low_spin, 0, 1)
        
        acc_layout.addWidget(QLabel("High Threshold:"), 1, 0)
        self.acc_high_spin = QDoubleSpinBox()
        self.acc_high_spin.setRange(0, 999999)
        self.acc_high_spin.setValue(self.state.sensitivity.acc_high_threshold)
        self.acc_high_spin.valueChanged.connect(self.on_acc_high_changed)
        acc_layout.addWidget(self.acc_high_spin, 1, 1)
        
        acc_group.setLayout(acc_layout)
        layout.addWidget(acc_group)
        
        dwell_group = QGroupBox("Dwell Sensitivity")
        dwell_layout = QGridLayout()
        
        dwell_layout.addWidget(QLabel("Dwell Threshold:"), 0, 0)
        self.dwell_thresh_spin = QSpinBox()
        self.dwell_thresh_spin.setRange(0, 999999)
        self.dwell_thresh_spin.setValue(self.state.sensitivity.dwell_threshold)
        self.dwell_thresh_spin.valueChanged.connect(self.on_dwell_thresh_changed)
        dwell_layout.addWidget(self.dwell_thresh_spin, 0, 1)
        
        dwell_layout.addWidget(QLabel("High Dwell:"), 1, 0)
        self.dwell_high_spin = QSpinBox()
        self.dwell_high_spin.setRange(0, 999999)
        self.dwell_high_spin.setValue(self.state.sensitivity.dwell_high_threshold)
        self.dwell_high_spin.valueChanged.connect(self.on_dwell_high_changed)
        dwell_layout.addWidget(self.dwell_high_spin, 1, 1)
        
        dwell_group.setLayout(dwell_layout)
        layout.addWidget(dwell_group)
        
        layout.addStretch()
        
    def on_speed_low_changed(self, value):
        self.state.sensitivity.speed_low_threshold = value
        
    def on_speed_high_changed(self, value):
        self.state.sensitivity.speed_high_threshold = value
        
    def on_acc_low_changed(self, value):
        self.state.sensitivity.acc_low_threshold = value
        
    def on_acc_high_changed(self, value):
        self.state.sensitivity.acc_high_threshold = value
        
    def on_dwell_thresh_changed(self, value):
        self.state.sensitivity.dwell_threshold = value
        
    def on_dwell_high_changed(self, value):
        self.state.sensitivity.dwell_high_threshold = value

# === Main Window ===
class BarbellTrackerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state = AppState()
        self.cap = None
        self.out = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Barbell Tracker Pro - Advanced Analytics")
        self.setMinimumSize(1400, 900)
        
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(28, 28, 30))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(35, 35, 38))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(45, 45, 48))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        self.setPalette(palette)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        
        video_splitter = QSplitter(Qt.Vertical)
        
        main_widget = QWidget()
        main_layout_inner = QVBoxLayout(main_widget)
        main_layout_inner.addWidget(QLabel("<h3 style='color: #52c41a;'>Main View (Scroll to Zoom)</h3>"))
        self.main_video = ZoomableVideoLabel()
        self.main_video.mouse_pressed.connect(self.on_video_mouse_press)
        self.main_video.mouse_moved.connect(self.on_video_mouse_move)
        self.main_video.mouse_released.connect(self.on_video_mouse_release)
        main_layout_inner.addWidget(self.main_video)
        
        speed_widget = QWidget()
        speed_layout_inner = QVBoxLayout(speed_widget)
        speed_layout_inner.addWidget(QLabel("<h3 style='color: #ff9f1c;'>Analytics View</h3>"))
        self.speed_video = ZoomableVideoLabel()
        speed_layout_inner.addWidget(self.speed_video)
        
        video_splitter.addWidget(main_widget)
        video_splitter.addWidget(speed_widget)
        video_splitter.setSizes([400, 400])
        
        video_layout.addWidget(video_splitter)
        
        timeline_group = QGroupBox("Timeline & Selection (Right-click to select range)")
        timeline_layout = QVBoxLayout()
        
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.value_changed.connect(self.on_timeline_change)
        self.timeline_widget.selection_started.connect(self.on_selection_started)
        self.timeline_widget.selection_updated.connect(self.on_selection_updated)
        self.timeline_widget.selection_finished.connect(self.on_selection_finished)
        
        timeline_info = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.time_label = QLabel("Time: 00:00.00")
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.time_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        timeline_info.addWidget(self.frame_label)
        timeline_info.addStretch()
        timeline_info.addWidget(self.time_label)
        
        timeline_layout.addWidget(self.timeline_widget)
        timeline_layout.addLayout(timeline_info)
        timeline_group.setLayout(timeline_layout)
        video_layout.addWidget(timeline_group)
        
        controls_container = QTabWidget()
        controls_container.setMaximumWidth(450)
        controls_container.setMinimumWidth(350)
        
        controls_tab = QWidget()
        controls_layout = QVBoxLayout(controls_tab)
        
        file_group = CollapsibleGroupBox("File Controls")
        file_layout = QVBoxLayout()
        
        load_btn = QPushButton("📁 Load Video")
        load_btn.clicked.connect(self.load_video)
        load_btn.setMinimumHeight(35)
        load_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        load_btn.setStyleSheet("background-color: #52c41a; font-weight: bold;")
        
        self.record_btn = QPushButton("⏺ Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setMinimumHeight(35)
        self.record_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.record_btn.setEnabled(False)
        
        body_metrics_btn = QPushButton("📊 Body Metrics")
        body_metrics_btn.clicked.connect(self.open_body_metrics)
        body_metrics_btn.setMinimumHeight(35)
        body_metrics_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        body_metrics_btn.setStyleSheet("background-color: #9c27b0; font-weight: bold;")
        
        file_layout.addWidget(load_btn)
        file_layout.addWidget(self.record_btn)
        file_layout.addWidget(body_metrics_btn)
        file_group.setLayout(file_layout)
        controls_layout.addWidget(file_group)
        
        playback_group = CollapsibleGroupBox("Playback")
        playback_layout = QGridLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        self.play_btn.setMinimumHeight(30)
        self.play_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        prev_btn = QPushButton("⏮ Prev")
        prev_btn.clicked.connect(self.prev_frame)
        prev_btn.setEnabled(False)
        prev_btn.setMinimumHeight(30)
        prev_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.prev_btn = prev_btn
        
        next_btn = QPushButton("Next ⏭")
        next_btn.clicked.connect(self.next_frame)
        next_btn.setEnabled(False)
        next_btn.setMinimumHeight(30)
        next_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.next_btn = next_btn
        
        playback_layout.addWidget(prev_btn, 0, 0)
        playback_layout.addWidget(self.play_btn, 0, 1)
        playback_layout.addWidget(next_btn, 0, 2)
        playback_group.setLayout(playback_layout)
        controls_layout.addWidget(playback_group)
        
        track_group = CollapsibleGroupBox("Tracking Controls")
        track_layout = QVBoxLayout()
        
        self.track_btn = QPushButton("🎯 Start Tracking")
        self.track_btn.clicked.connect(self.start_tracking)
        self.track_btn.setEnabled(False)
        self.track_btn.setMinimumHeight(35)
        self.track_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.track_btn.setStyleSheet("background-color: #ff9f1c; font-weight: bold;")
        
        clear_trail_btn = QPushButton("Clear Trail")
        clear_trail_btn.clicked.connect(self.clear_trail)
        clear_trail_btn.setMinimumHeight(30)
        clear_trail_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        reset_btn = QPushButton("🔄 Reset All")
        reset_btn.clicked.connect(self.reset_all)
        reset_btn.setStyleSheet("background-color: #d9534f;")
        reset_btn.setMinimumHeight(30)
        reset_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        track_layout.addWidget(self.track_btn)
        track_layout.addWidget(clear_trail_btn)
        track_layout.addWidget(reset_btn)
        
        self.tracking_status = QLabel("● Status: Ready")
        self.tracking_status.setStyleSheet("color: #888; font-weight: bold;")
        self.tracking_status.setWordWrap(True)
        track_layout.addWidget(self.tracking_status)
        
        track_group.setLayout(track_layout)
        controls_layout.addWidget(track_group)
        
        sel_group = CollapsibleGroupBox("Selection Management")
        sel_layout = QVBoxLayout()
        
        sel_buttons = QHBoxLayout()
        delete_btn = QPushButton("Delete Last")
        delete_btn.clicked.connect(self.delete_last_selection)
        delete_btn.setMinimumHeight(28)
        delete_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        clear_sel_btn = QPushButton("Clear All")
        clear_sel_btn.clicked.connect(self.clear_selections)
        clear_sel_btn.setMinimumHeight(28)
        clear_sel_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        sel_buttons.addWidget(delete_btn)
        sel_buttons.addWidget(clear_sel_btn)
        sel_layout.addLayout(sel_buttons)
        
        self.selection_list = QLabel("No selections")
        self.selection_list.setStyleSheet("color: #aaa; padding: 8px; background: #23232a; border-radius: 5px;")
        self.selection_list.setWordWrap(True)
        self.selection_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sel_layout.addWidget(self.selection_list)
        
        sel_group.setLayout(sel_layout)
        controls_layout.addWidget(sel_group)
        
        viz_group = CollapsibleGroupBox("Analytics Mode")
        viz_layout = QVBoxLayout()
        
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["Speed (Green→Yellow→Orange→Red)", 
                                 "Acceleration (Cyan→Blue→Yellow→Orange→Red)",
                                 "Dwell Time (Green→Yellow→Orange→Red)"])
        self.viz_combo.currentIndexChanged.connect(self.change_viz_mode)
        self.viz_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        viz_layout.addWidget(self.viz_combo)
        
        viz_group.setLayout(viz_layout)
        controls_layout.addWidget(viz_group)
        
        controls_layout.addStretch()
        
        sensitivity_tab = QWidget()
        sensitivity_layout = QVBoxLayout(sensitivity_tab)
        
        sensitivity_scroll = QScrollArea()
        sensitivity_scroll.setWidgetResizable(True)
        
        self.sensitivity_controls = SensitivityControls(self.state)
        sensitivity_scroll.setWidget(self.sensitivity_controls)
        
        sensitivity_layout.addWidget(sensitivity_scroll)
        
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        
        stats_container = QWidget()
        stats_container_layout = QVBoxLayout(stats_container)
        
        overall_group = CollapsibleGroupBox("📊 Live Statistics")
        overall_layout = QVBoxLayout()
        self.overall_stats = QLabel("Load video and track to see stats")
        self.overall_stats.setStyleSheet("color: #aaa; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
        self.overall_stats.setWordWrap(True)
        self.overall_stats.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overall_layout.addWidget(self.overall_stats)
        overall_group.setLayout(overall_layout)
        stats_container_layout.addWidget(overall_group)
        
        current_group = CollapsibleGroupBox("🎯 Current Frame Stats")
        current_layout = QVBoxLayout()
        self.current_stats = QLabel("No tracking data")
        self.current_stats.setStyleSheet("color: #52c41a; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
        self.current_stats.setWordWrap(True)
        self.current_stats.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        current_layout.addWidget(self.current_stats)
        current_group.setLayout(current_layout)
        stats_container_layout.addWidget(current_group)
        
        sel_stats_group = CollapsibleGroupBox("📈 Current Selection Stats")
        sel_stats_layout = QVBoxLayout()
        self.selection_stats = QLabel("Move timeline to selection range")
        self.selection_stats.setStyleSheet("color: #aaa; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
        self.selection_stats.setWordWrap(True)
        self.selection_stats.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sel_stats_layout.addWidget(self.selection_stats)
        sel_stats_group.setLayout(sel_stats_layout)
        stats_container_layout.addWidget(sel_stats_group)
        
        stats_container_layout.addStretch()
        stats_scroll.setWidget(stats_container)
        stats_layout.addWidget(stats_scroll)
        
        controls_container.addTab(controls_tab, "Controls")
        controls_container.addTab(sensitivity_tab, "Sensitivity")
        controls_container.addTab(stats_tab, "Statistics")
        
        main_splitter.addWidget(video_container)
        main_splitter.addWidget(controls_container)
        main_splitter.setSizes([1000, 400])
        
    def open_body_metrics(self):
        try:
            script_path = "bodyui.py"
            
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
            else:
                filename, _ = QFileDialog.getOpenFileName(
                    self, 
                    "Locate Body Metrics Script", 
                    "", 
                    "Python Files (*.py);;All Files (*)"
                )
                if filename:
                    subprocess.Popen([sys.executable, filename])
                else:
                    QMessageBox.warning(
                        self, 
                        "Script Not Found", 
                        "bodyui.py not found in current directory."
                    )
                    
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to open Body Metrics: {str(e)}"
            )
        
    def load_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if not filename:
            return
            
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
            
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Cannot open video file")
            return
            
        self.state.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.state.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.state.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.state.video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 25.0
        
        self.state.path_by_frame = [None] * max(1, self.state.total_frames)
        self.timeline_widget.set_maximum(max(1, self.state.total_frames - 1))
        
        ret, self.state.frame = self.cap.read()
        if ret:
            self.update_display()
            self.play_btn.setEnabled(True)
            self.track_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.tracking_status.setText("● Status: Video Loaded")
            self.tracking_status.setStyleSheet("color: #52c41a; font-weight: bold;")
            
    def toggle_playback(self):
        self.state.paused = not self.state.paused
        if self.state.paused:
            self.timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.timer.start(int(1000 / self.state.video_fps))
            self.play_btn.setText("⏸ Pause")
            
    def prev_frame(self):
        if self.state.frame_idx > 0:
            self.state.paused = True
            self.timer.stop()
            self.play_btn.setText("▶ Play")
            self.state.frame_idx -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.state.frame_idx)
            ret, self.state.frame = self.cap.read()
            self.update_display()
            
    def next_frame(self):
        if self.state.frame_idx < self.state.total_frames - 1:
            self.state.paused = True
            self.timer.stop()
            self.play_btn.setText("▶ Play")
            self.state.frame_idx += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.state.frame_idx)
            ret, self.state.frame = self.cap.read()
            self.update_display()
            
    def update_frame(self):
        if not self.cap:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return
                
        self.state.frame = frame
        self.state.frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        
        if self.state.tracking_active and self.state.tracker:
            ok, new_bbox = self.state.tracker.update(frame)
            if ok:
                self.state.bbox = tuple(map(int, new_bbox))
                x, y, w, h = self.state.bbox
                center = (x + w // 2, y + h // 2)
                
                active_sel = self.get_active_selection_at_frame(self.state.frame_idx)
                
                if active_sel:
                    sel_idx = self.state.selections.index(active_sel)
                    if 0 <= self.state.frame_idx < len(self.state.path_by_frame):
                        self.state.path_by_frame[self.state.frame_idx] = (center, sel_idx)
                        self.state.path_points.append((center, sel_idx))
                else:
                    self.state.tracking_active = False
                    self.tracking_status.setText("● Status: Left selection range")
                    self.tracking_status.setStyleSheet("color: #ff9f1c; font-weight: bold;")
            else:
                self.state.tracking_active = False
                self.tracking_status.setText("● Status: Tracking lost")
                self.tracking_status.setStyleSheet("color: #d9534f; font-weight: bold;")
        
        self.update_display()
        
    def on_timeline_change(self, value):
        if not self.cap or self.timer.isActive():
            return
        self.state.frame_idx = value
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.state.frame_idx)
        ret, self.state.frame = self.cap.read()
        if ret:
            self.update_path_points_from_frames()
            self.update_display()
    
    def on_selection_started(self, progress):
        self.state.selecting = True
        self.state.select_start_progress = progress
        self.state.select_current_progress = progress
    
    def on_selection_updated(self, progress):
        self.state.select_current_progress = progress
        self.timeline_widget.update()
    
    def on_selection_finished(self, start_prog, end_prog):
        self.state.selecting = False
        sprog = min(start_prog, end_prog)
        eprog = max(start_prog, end_prog)
        sframe = max(0, min(int(sprog * (self.state.total_frames - 1)), self.state.total_frames - 1))
        eframe = max(0, min(int(eprog * (self.state.total_frames - 1)), self.state.total_frames - 1))
        
        if eframe - sframe >= 1:
            new_selection = Selection(sframe, eframe, self.state.current_selection_idx)
            self.state.selections.append(new_selection)
            self.state.current_selection_idx += 1
            self.timeline_widget.set_selections(self.state.selections)
            self.update_selections_display()
            self.update_display()
        else:
            QMessageBox.warning(self, "Selection Too Small", "Selection must span at least 2 frames")
            
    def start_tracking(self):
        if self.state.bbox is None:
            QMessageBox.warning(self, "No ROI", "Please draw a region of interest first")
            return
        
        tracker = create_tracker()
        if tracker is None:
            error_msg = ("No tracker available in your OpenCV installation.\n\n"
                        "To fix this, install opencv-contrib-python:\n"
                        "pip uninstall opencv-python\n"
                        "pip install opencv-contrib-python\n\n"
                        "Or update your OpenCV version:\n"
                        "pip install --upgrade opencv-contrib-python")
            QMessageBox.critical(self, "Tracker Error", error_msg)
            return
        
        try:
            tracker.init(self.state.frame, tuple(map(int, self.state.bbox)))
            self.state.tracker = tracker
            self.state.tracking_active = True
            
            x, y, w, h = self.state.bbox
            center = (x + w // 2, y + h // 2)
            
            active_sel = self.get_active_selection_at_frame(self.state.frame_idx)
            sel_idx = self.state.selections.index(active_sel) if active_sel else -1
            
            self.state.path_by_frame[self.state.frame_idx] = (center, sel_idx)
            self.update_path_points_from_frames()
            
            self.tracking_status.setText("● Status: Tracking Active")
            self.tracking_status.setStyleSheet("color: #52c41a; font-weight: bold;")
            self.update_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Tracking initialization failed: {e}\n\nTry reinstalling opencv-contrib-python")
            self.state.tracking_active = False
            self.state.tracker = None
            
    def clear_trail(self):
        self.state.path_by_frame = [None] * max(1, self.state.total_frames)
        self.state.path_points.clear()
        self.update_display()
        self.update_stats()
        
    def reset_all(self):
        reply = QMessageBox.question(self, "Reset All", 
                                     "This will clear all tracking data and selections. Continue?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.state.tracker = None
            self.state.bbox = None
            self.state.tracking_active = False
            self.state.path_points.clear()
            self.state.path_by_frame = [None] * max(1, self.state.total_frames)
            self.state.drawing = False
            self.state.draw_start = self.state.draw_current = None
            self.state.selections = []
            self.state.current_selection_idx = 0
            self.state.selecting = False
            self.timeline_widget.set_selections(self.state.selections)
            self.tracking_status.setText("● Status: Reset Complete")
            self.tracking_status.setStyleSheet("color: #888; font-weight: bold;")
            self.update_display()
            self.update_stats()
            self.update_selections_display()
            
    def delete_last_selection(self):
        if self.state.selections:
            deleted = self.state.selections.pop()
            for i in range(len(self.state.path_by_frame)):
                if self.state.path_by_frame[i] is not None:
                    pos, sel_idx = self.state.path_by_frame[i]
                    if sel_idx == len(self.state.selections):
                        self.state.path_by_frame[i] = None
            self.timeline_widget.set_selections(self.state.selections)
            self.update_selections_display()
            self.update_display()
        else:
            QMessageBox.information(self, "No Selections", "No selections to delete")
            
    def clear_selections(self):
        if self.state.selections:
            reply = QMessageBox.question(self, "Clear Selections", 
                                        "Clear all selections but keep tracked paths?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.state.selections = []
                self.state.current_selection_idx = 0
                self.state.selecting = False
                self.timeline_widget.set_selections(self.state.selections)
                self.update_selections_display()
                self.update_display()
                
    def change_viz_mode(self, index):
        self.state.viz_mode = index + 1
        self.update_display()
        
    def toggle_recording(self):
        self.state.recording = not self.state.recording
        if self.state.recording:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.out = cv2.VideoWriter(OUTPUT_PATH, fourcc, self.state.video_fps, 
                                      (self.state.W, self.state.H))
            self.record_btn.setText("⏹ Stop Recording")
            self.record_btn.setStyleSheet("background-color: #d9534f; font-weight: bold;")
        else:
            if self.out:
                self.out.release()
            self.record_btn.setText("⏺ Start Recording")
            self.record_btn.setStyleSheet("background-color: #52c41a; font-weight: bold;")
            QMessageBox.information(self, "Recording Saved", f"Video saved to {OUTPUT_PATH}")
            
    def update_display(self):
        if self.state.frame is None:
            return
            
        display_points = [p for p in self.state.path_by_frame[:self.state.frame_idx + 1] if p is not None]
        self.update_path_points_from_frames()
        
        display = self.state.frame.copy()
        draw_trail_colored(display, display_points)
        
        if self.state.bbox:
            active_sel = self.get_active_selection_at_frame(self.state.frame_idx)
            bbox_color = active_sel.color if active_sel else Theme.PRIMARY
            if not self.state.tracking_active:
                bbox_color = Theme.ACCENT
            draw_bbox(display, self.state.bbox, bbox_color)
        
        if self.state.drawing and self.state.draw_start and self.state.draw_current:
            x1, y1 = self.state.draw_start
            x2, y2 = self.state.draw_current
            draw_bbox(display, (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)), Theme.ACCENT)
        
        rgb_image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.main_video.set_image(QPixmap.fromImage(qt_image))
        
        speed_view = draw_analytics_speed_view(display_points, self.state.viz_mode, 
                                               self.state.W, self.state.H, self.state.video_fps,
                                               self.state.sensitivity)
        rgb_speed = cv2.cvtColor(speed_view, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_speed.shape
        bytes_per_line = ch * w
        qt_speed = QImage(rgb_speed.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.speed_video.set_image(QPixmap.fromImage(qt_speed))
        
        self.timeline_widget.set_value(self.state.frame_idx)
        self.timeline_widget.set_selections(self.state.selections)
        
        self.frame_label.setText(f"Frame: {self.state.frame_idx} / {self.state.total_frames - 1}")
        time_sec = self.state.frame_idx / self.state.video_fps
        mins = int(time_sec // 60)
        secs = time_sec % 60
        self.time_label.setText(f"Time: {mins:02d}:{secs:05.2f}")
        
        if self.state.recording and self.out:
            self.out.write(display)
            
        self.update_stats()
        
    def update_stats(self):
        overall_text = "<b>📊 Overall Statistics</b><br><br>"
        overall_text += f"<b>Current Frame:</b> {self.state.frame_idx} / {self.state.total_frames - 1}<br>"
        overall_text += f"<b>Video FPS:</b> {self.state.video_fps:.2f}<br>"
        overall_text += f"<b>Selections:</b> {len(self.state.selections)}<br><br>"
        
        overall_text += "<b>Sensitivity Settings:</b><br>"
        overall_text += f"Speed: {self.state.sensitivity.speed_low_threshold:.1f}-{self.state.sensitivity.speed_high_threshold:.1f} px/s<br>"
        overall_text += f"Accel: {self.state.sensitivity.acc_low_threshold:.1f}-{self.state.sensitivity.acc_high_threshold:.1f} px/s²<br>"
        overall_text += f"Dwell: {self.state.sensitivity.dwell_threshold}-{self.state.sensitivity.dwell_high_threshold} frames<br><br>"
        
        all_points = [p for p in self.state.path_by_frame if p is not None]
        
        if all_points:
            metrics = calculate_metrics(all_points, self.state.video_fps, self.state.sensitivity)
            if metrics:
                overall_text += f"<b>Total Frames Tracked:</b> {metrics['total_frames']}<br>"
                overall_text += f"<b>Total Duration:</b> {metrics['duration']:.2f}s<br>"
        
        self.overall_stats.setText(overall_text)
        self.overall_stats.setStyleSheet("color: #fff; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
        
        if not any(self.state.path_by_frame):
            self.current_stats.setText("No tracking data yet")
            self.current_stats.setStyleSheet("color: #aaa; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
        else:
            all_points = [p for p in self.state.path_by_frame if p is not None]
            
            if not all_points:
                self.current_stats.setText("No tracking data yet")
                self.current_stats.setStyleSheet("color: #aaa; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
            else:
                metrics = calculate_metrics(all_points, self.state.video_fps, self.state.sensitivity)
                
                if metrics:
                    stats_text = "<b>Overall Tracking Stats</b><br><br>"
                    stats_text += f"<b>Total Frames Tracked:</b> {metrics['total_frames']}<br>"
                    stats_text += f"<b>Duration:</b> {metrics['duration']:.2f}s<br>"
                    stats_text += f"<b>Total Distance:</b> {metrics['total_distance']:.1f} px<br>"
                    stats_text += f"<b>Vertical ROM:</b> {metrics['vertical_rom']:.1f} px<br>"
                    stats_text += f"<b>Horizontal Sway:</b> {metrics['horizontal_sway']:.1f} px<br><br>"
                    stats_text += f"<b>Speed Statistics:</b><br>"
                    stats_text += f"  • Avg: {metrics['avg_speed']:.1f} px/s<br>"
                    stats_text += f"  • Max: {metrics['max_speed']:.1f} px/s<br>"
                    stats_text += f"  • Min: {metrics['min_speed']:.1f} px/s<br><br>"
                    stats_text += f"<b>Acceleration:</b><br>"
                    stats_text += f"  • Avg: {metrics['avg_acceleration']:.1f} px/s²<br>"
                    stats_text += f"  • Max: {metrics['max_acceleration']:.1f} px/s²<br><br>"
                    stats_text += f"<b>Max Dwell Time:</b> {metrics['max_dwell']:.0f} frames<br>"
                    
                    if 0 <= self.state.frame_idx < len(self.state.path_by_frame):
                        current_data = self.state.path_by_frame[self.state.frame_idx]
                        if current_data:
                            pos, sel_idx = current_data
                            stats_text += f"<br><b>Current Position:</b> ({pos[0]:.0f}, {pos[1]:.0f})<br>"
                            
                            if self.state.frame_idx > 0:
                                prev_data = self.state.path_by_frame[self.state.frame_idx - 1]
                                if prev_data:
                                    prev_pos, _ = prev_data
                                    dx = pos[0] - prev_pos[0]
                                    dy = pos[1] - prev_pos[1]
                                    current_speed = np.sqrt(dx*dx + dy*dy) * self.state.video_fps
                                    stats_text += f"<b>Current Speed:</b> {current_speed:.1f} px/s<br>"
                                    
                                    if self.state.frame_idx > 1:
                                        prev_prev_data = self.state.path_by_frame[self.state.frame_idx - 2]
                                        if prev_prev_data:
                                            prev_prev_pos, _ = prev_prev_data
                                            prev_dx = prev_pos[0] - prev_prev_pos[0]
                                            prev_dy = prev_pos[1] - prev_prev_pos[1]
                                            prev_speed = np.sqrt(prev_dx*prev_dx + prev_dy*prev_dy) * self.state.video_fps
                                            current_acc = abs(current_speed - prev_speed) * self.state.video_fps
                                            stats_text += f"<b>Current Acceleration:</b> {current_acc:.1f} px/s²<br>"
                    
                    self.current_stats.setText(stats_text)
                    self.current_stats.setStyleSheet("color: #52c41a; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
                else:
                    self.current_stats.setText("Not enough data for statistics")
                    self.current_stats.setStyleSheet("color: #aaa; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
        
        self.update_selection_stats()
    
    def update_selection_stats(self):
        active_sel = self.get_active_selection_at_frame(self.state.frame_idx)
        
        if not active_sel:
            self.selection_stats.setText("Move timeline into a selection range to see stats")
            self.selection_stats.setStyleSheet("color: #aaa; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
            return
        
        sel_idx = self.state.selections.index(active_sel)
        color_hex = f"#{active_sel.color[2]:02x}{active_sel.color[1]:02x}{active_sel.color[0]:02x}"
        
        stats_text = f"<b>Selection #{sel_idx + 1}</b><br>"
        stats_text += f"<span style='color: {color_hex};'>● Frames {active_sel.start_frame} → {active_sel.end_frame}</span><br>"
        stats_text += f"<b>Duration:</b> {(active_sel.end_frame - active_sel.start_frame) / self.state.video_fps:.2f}s<br><br>"
        
        sel_points = []
        for i in range(active_sel.start_frame, active_sel.end_frame + 1):
            if i < len(self.state.path_by_frame) and self.state.path_by_frame[i]:
                sel_points.append(self.state.path_by_frame[i])
        
        if sel_points:
            stats_text += "<b>Tracking in Selection:</b><br>"
            metrics = calculate_metrics(sel_points, self.state.video_fps, self.state.sensitivity)
            if metrics:
                stats_text += f"  • Frames: {metrics['total_frames']}<br>"
                stats_text += f"  • Distance: {metrics['total_distance']:.1f} px<br>"
                stats_text += f"  • V-ROM: {metrics['vertical_rom']:.1f} px<br>"
                stats_text += f"  • H-Sway: {metrics['horizontal_sway']:.1f} px<br>"
                stats_text += f"  • Avg Speed: {metrics['avg_speed']:.1f} px/s<br>"
                stats_text += f"  • Max Speed: {metrics['max_speed']:.1f} px/s<br>"
                stats_text += f"  • Min Speed: {metrics['min_speed']:.1f} px/s<br>"
                stats_text += f"  • Avg Accel: {metrics['avg_acceleration']:.1f} px/s²<br>"
                stats_text += f"  • Max Accel: {metrics['max_acceleration']:.1f} px/s²<br>"
                stats_text += f"  • Max Dwell: {metrics['max_dwell']:.0f} frames<br>"
        else:
            stats_text += "No tracking data in this selection<br>"
        
        self.selection_stats.setText(stats_text)
        self.selection_stats.setStyleSheet(f"color: {color_hex}; padding: 10px; background: #23232a; border-radius: 5px; font-size: 11px;")
            
    def update_selections_display(self):
        if self.state.selections:
            sel_text = "<b>Active Selections:</b><br>"
            for i, sel in enumerate(self.state.selections):
                color_hex = f"#{sel.color[2]:02x}{sel.color[1]:02x}{sel.color[0]:02x}"
                duration = (sel.end_frame - sel.start_frame) / self.state.video_fps
                sel_text += f"<span style='color: {color_hex};'>● #{i+1}</span>: {sel.start_frame}→{sel.end_frame} ({duration:.2f}s)<br>"
            self.selection_list.setText(sel_text)
            self.selection_list.setStyleSheet("color: #fff; padding: 8px; background: #23232a; border-radius: 5px;")
        else:
            self.selection_list.setText("No selections")
            self.selection_list.setStyleSheet("color: #aaa; padding: 8px; background: #23232a; border-radius: 5px;")
            
    def update_path_points_from_frames(self):
        items = [p for p in self.state.path_by_frame[:self.state.frame_idx + 1] if p is not None]
        self.state.path_points.clear()
        for item in items[-TRAIL_MAX_POINTS:]:
            self.state.path_points.append(item)
            
    def get_active_selection_at_frame(self, frame_idx):
        for sel in self.state.selections:
            if sel.contains_frame(frame_idx):
                return sel
        return None
        
    def on_video_mouse_press(self, x, y, button):
        if button == Qt.LeftButton:
            self.state.drawing = True
            label_w = self.main_video.width()
            label_h = self.main_video.height()
            pixmap = self.main_video.pixmap()
            if pixmap:
                pix_w = pixmap.width()
                pix_h = pixmap.height()
                offset_x = (label_w - pix_w) // 2
                offset_y = (label_h - pix_h) // 2
                scale_x = self.state.W / pix_w if pix_w > 0 else 1
                scale_y = self.state.H / pix_h if pix_h > 0 else 1
                vid_x = int((x - offset_x) * scale_x)
                vid_y = int((y - offset_y) * scale_y)
                self.state.draw_start = (vid_x, vid_y)
                self.state.draw_current = (vid_x, vid_y)
                
    def on_video_mouse_move(self, x, y):
        if self.state.drawing:
            label_w = self.main_video.width()
            label_h = self.main_video.height()
            pixmap = self.main_video.pixmap()
            if pixmap:
                pix_w = pixmap.width()
                pix_h = pixmap.height()
                offset_x = (label_w - pix_w) // 2
                offset_y = (label_h - pix_h) // 2
                scale_x = self.state.W / pix_w if pix_w > 0 else 1
                scale_y = self.state.H / pix_h if pix_h > 0 else 1
                vid_x = int((x - offset_x) * scale_x)
                vid_y = int((y - offset_y) * scale_y)
                self.state.draw_current = (vid_x, vid_y)
                self.update_display()
                
    def on_video_mouse_release(self, x, y, button):
        if button == Qt.LeftButton and self.state.drawing:
            self.state.drawing = False
            if self.state.draw_start and self.state.draw_current:
                x1, y1 = self.state.draw_start
                x2, y2 = self.state.draw_current
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                w, h = x_max - x_min, y_max - y_min
                if w > 30 and h > 30:
                    self.state.bbox = (x_min, y_min, w, h)
                    self.state.tracking_active = False
                    self.state.path_points.clear()
                    self.update_display()
            self.state.draw_start = self.state.draw_current = None
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'main_video'):
            self.main_video.update_zoom()
        if hasattr(self, 'speed_video'):
            self.speed_video.update_zoom()
            
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = BarbellTrackerUI()
    window.showMaximized()
    
    sys.exit(app.exec())