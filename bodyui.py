#!/usr/bin/env python3
"""
Responsive Pose Analysis Application with Professional Dark Mode UI
- Modern dark theme with professional styling
- Dynamic/responsive layout that adapts to screen resolution
- Enhanced visual hierarchy and spacing
- Consistent color scheme and typography
- Preserves all original functionality

Save as `pose_analysis_dark.py` and run with: `python3 pose_analysis_dark.py`
"""

import sys
import os
import time
import math
import cv2
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from io import BytesIO
from scipy import stats

# PySide6 imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QWidget, QFileDialog, QMessageBox, QSplitter, QProgressBar,
    QStatusBar, QToolBar, QFrame, QDialog, QGroupBox, QCheckBox,
    QProgressDialog, QGridLayout, QScrollArea, QListWidget, QListWidgetItem,
    QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, QSize, QThread, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap, QImage, QPainter, QFont

# Matplotlib for plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ultralytics YOLO for pose estimation
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    sys.exit(1)


class DarkTheme:
    """Centralized dark theme configuration"""
    COLORS = {
        # Primary colors
        'primary': '#2b579a',
        'primary_light': '#3d6cc4',
        'primary_dark': '#1e3d72',
        
        # Background colors
        'bg_dark': '#1a1a1a',
        'bg_medium': '#2d2d2d',
        'bg_light': '#3d3d3d',
        'bg_widget': '#252525',
        
        # Text colors
        'text_primary': '#ffffff',
        'text_secondary': '#b3b3b3',
        'text_tertiary': '#808080',
        'text_accent': '#4fc3f7',
        
        # Status colors
        'success': '#4caf50',
        'warning': '#ff9800',
        'error': '#f44336',
        'info': '#2196f3',
        
        # Graph colors
        'graph_line_1': '#00ff88',
        'graph_line_2': '#ff3366',
        'graph_line_3': '#ffaa00',
        'graph_line_4': '#3399ff',
        'graph_line_5': '#cc66ff',
        'graph_line_6': '#ffff00',
        'graph_line_7': '#ff66cc',
        'graph_line_8': '#66ffff',
        
        # UI elements
        'border': '#404040',
        'border_light': '#4d4d4d',
        'hover': '#3a3a3a',
        'pressed': '#4a4a4a'
    }
    
    STYLESHEETS = {
        'main_window': f"""
            QMainWindow {{
                background-color: {COLORS['bg_dark']};
                color: {COLORS['text_primary']};
            }}
        """,
        
        'toolbar': f"""
            QToolBar {{
                background-color: {COLORS['bg_medium']};
                border: none;
                spacing: 8px;
                padding: 6px;
            }}
            QToolBar::separator {{
                background-color: {COLORS['border']};
                width: 1px;
                margin: 4px 8px;
            }}
        """,
        
        'button_primary': f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['primary_dark']};
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_light']};
                border: 1px solid {COLORS['primary_light']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_dark']};
                border: 1px solid {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border']};
            }}
        """,
        
        'button_secondary': f"""
            QPushButton {{
                background-color: {COLORS['bg_widget']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 18px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['hover']};
                border: 1px solid {COLORS['border_light']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['pressed']};
                border: 1px solid {COLORS['border_light']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border']};
            }}
        """,
        
        'group_box': f"""
            QGroupBox {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0px 8px 0px 8px;
                color: {COLORS['text_accent']};
            }}
        """,
        
        'progress_bar': f"""
            QProgressBar {{
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                background-color: {COLORS['bg_widget']};
                text-align: center;
                color: {COLORS['text_primary']};
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 3px;
            }}
        """,
        
        'scroll_area': f"""
            QScrollArea {{
                background-color: {COLORS['bg_widget']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QScrollBar:vertical {{
                background-color: {COLORS['bg_medium']};
                width: 15px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {COLORS['bg_light']};
                border-radius: 7px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {COLORS['text_tertiary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
        """,
        
        'video_display': f"""
            QLabel {{
                background-color: {COLORS['bg_widget']};
                color: {COLORS['text_secondary']};
                border: 2px solid {COLORS['border']};
                border-radius: 6px;
                font-size: 12px;
            }}
        """,
        
        'status_bar': f"""
            QStatusBar {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_secondary']};
                border-top: 1px solid {COLORS['border']};
            }}
        """,
        
        'checkbox': f"""
            QCheckBox {{
                color: {COLORS['text_primary']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
            }}
            QCheckBox::indicator:unchecked {{
                background-color: {COLORS['bg_widget']};
                border: 1px solid {COLORS['border']};
                border-radius: 3px;
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['primary']};
                border: 1px solid {COLORS['primary_dark']};
                border-radius: 3px;
            }}
        """
    }


class OneEuroFilter:
    """Smoothing filter for time-series data"""
    def __init__(self, freq=30.0, min_cutoff=0.3, beta=0.007, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.last_time = None

    def alpha(self, cutoff):
        te = 1.0 / max(1e-8, self.freq)
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        t = time.time()
        if self.last_time is not None:
            dt = t - self.last_time
            if dt > 1e-6:
                # update internal frequency estimate
                self.freq = 1.0 / dt
        self.last_time = t

        x = np.asarray(x, dtype=float)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
            
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat.copy()
        self.dx_prev = dx_hat.copy()
        return x_hat


class PoseAnalyzer:
    def __init__(self):
        # Configuration
        self.POSE_MODEL_PATH = "yolov8m-pose.pt"
        self.FPS_FALLBACK = 25.0
        self.MIN_CONF = 0.3
        self.ROLLING_SEC = 2.5
        
        # Body keypoints indices (COCO-style)
        self.BODY_KP_INDICES = list(range(17))  # keep full COCO range (0..16)
        
        # Joint definitions
        self.JOINT_DEFS = {
            "left_elbow": (5, 7, 9),
            "right_elbow": (6, 8, 10),
            "left_knee": (11, 13, 15),
            "right_knee": (12, 14, 16),
            "left_hip": (5, 11, 13),
            "right_hip": (6, 12, 14),
            "left_shoulder": (5, 7, 11),
            "right_shoulder": (6, 8, 12)
        }
        
        self.PAIRS = [
            ("left_elbow", "right_elbow"),
            ("left_knee", "right_knee"), 
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder")
        ]
        
        # Initialize components
        try:
            self.model = YOLO(self.POSE_MODEL_PATH)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Ensure model file exists or change POSE_MODEL_PATH")
            self.model = None
            # don't exit here; allow offline testing
            
        self.cap = None
        self.angle_filters = {}
        self.kp_filters = {}
        self.time_buf = deque()
        self.angvel_buffers = {}
        self.prev_angles = {}
        self.state = {}
        self.frame_data = []
        self.current_frame = 0
        self.start_time = 0
        self.is_processing = False
        
        # Analysis results storage
        self.analysis_results = {
            'joint_angles': [],
            'angular_velocities': [],
            'rom_data': [],
            'symmetry_data': [],
            'dominance_data': []
        }
        
        self.buffer_len = 0
        
    def load_video(self, video_path):
        """Load video for analysis"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or self.FPS_FALLBACK)
        try:
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            self.total_frames = 0
        
        # Initialize filters and buffers
        self.buffer_len = int(max(10, self.ROLLING_SEC * self.fps))
        self.angle_filters = {name: OneEuroFilter(freq=self.fps) for name in self.JOINT_DEFS.keys()}
        self.kp_filters = {i: OneEuroFilter(freq=self.fps) for i in self.BODY_KP_INDICES}
        self.time_buf = deque(maxlen=self.buffer_len)
        self.angvel_buffers = {name: deque(maxlen=self.buffer_len) for name in self.JOINT_DEFS.keys()}
        
        # Reset state
        self.prev_angles = {}
        self.state = {}
        self.frame_data = []
        self.current_frame = 0
        self.start_time = time.time()
        
        return True
    
    def process_next_frame(self):
        """Process a single frame from the video"""
        if not self.cap or not self.cap.isOpened():
            return False, None, None, None, {}, {}
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None, None, None, {}, {}
        
        # Calculate timing
        now = time.time()
        dt = 1.0 / max(1e-6, self.fps)
        rel_t = now - self.start_time
        
        # Run pose estimation (if model loaded)
        person_kps = None
        if self.model is not None:
            try:
                results = self.model.predict(frame, imgsz=640, verbose=False)
                person_kps = self._extract_person_keypoints(results)
            except Exception as e:
                # Keep robustness: if predict fails, continue with None keypoints
                print(f"YOLO predict error: {e}")
                person_kps = None
        
        # Smooth keypoints
        if person_kps is not None:
            person_kps = self._smooth_keypoints(person_kps)
        
        # Compute metrics
        metrics, joints_data = self._compute_metrics(person_kps, dt, rel_t)
        
        # Compute dominance
        dominance = self._compute_dominance(metrics)
        
        # Draw visualization
        display_frame = self._draw_skeleton_and_metrics(frame, person_kps, dominance, metrics)
        
        # Create plots (may return None if insufficient data)
        plot_img = self._create_angular_velocity_plot(rel_t)
        bilateral_img = self._create_bilateral_display()
        
        # Store frame data for export
        frame_record = {
            'frame_number': self.current_frame,
            'timestamp': rel_t,
            'joints': joints_data,
            'dominance': dominance,
            'metrics': metrics
        }
        self.frame_data.append(frame_record)
        
        self.current_frame += 1
        
        return True, display_frame, plot_img, bilateral_img, metrics, dominance
    
    def _extract_person_keypoints(self, results):
        """Extract main person keypoints from YOLO results"""
        if results is None or len(results) == 0:
            return None
            
        r0 = results[0]
        kps_raw = None
        
        # Extract keypoints from different possible attributes
        try:
            kps_raw = self._to_numpy_safe(r0.keypoints.xy)
        except Exception:
            try:
                kps_raw = self._to_numpy_safe(r0.keypoints)
            except Exception:
                kps_raw = None
                
        if kps_raw is None:
            try:
                kps_raw = self._to_numpy_safe(r0.poses)
            except Exception:
                kps_raw = None
        
        if isinstance(kps_raw, np.ndarray) and kps_raw.ndim == 3:
            centers = []
            for i in range(kps_raw.shape[0]):
                arr = kps_raw[i]
                vals = []
                for idx in (5, 6, 11, 12):  # Torso points
                    if idx < len(arr):
                        p = arr[idx]
                        if not np.isnan(p[0]) and not np.isnan(p[1]):
                            vals.append((float(p[0]), float(p[1])))
                if len(vals) == 0:
                    centers.append((1e9, 1e9))
                else:
                    cx = sum(v[0] for v in vals) / len(vals)
                    cy = sum(v[1] for v in vals) / len(vals)
                    centers.append((cx, cy))
            
            frame_center = (self.width / 2.0, self.height / 2.0)
            dists = [math.hypot(c[0] - frame_center[0], c[1] - frame_center[1]) for c in centers]
            idx = int(np.argmin(dists))
            return kps_raw[idx]
        elif kps_raw is not None and kps_raw.ndim == 2:
            return kps_raw
        
        return None
    
    def _smooth_keypoints(self, keypoints):
        """Apply smoothing filter to keypoints"""
        smoothed = []
        n = max(int(keypoints.shape[0]), 17)
        for i in range(n):
            if i in self.BODY_KP_INDICES and i < keypoints.shape[0]:
                p = keypoints[i]
                try:
                    if np.any(np.isnan(p)):
                        smoothed.append((np.nan, np.nan))
                    else:
                        f = self.kp_filters.get(i)
                        if f is None:
                            smoothed.append((float(p[0]), float(p[1])))
                        else:
                            filtered = f.filter([float(p[0]), float(p[1])])
                            smoothed.append((filtered[0], filtered[1]))
                except Exception:
                    smoothed.append((np.nan, np.nan))
            else:
                smoothed.append((np.nan, np.nan))
        return np.array(smoothed)
    
    def _compute_metrics(self, keypoints, dt, rel_t):
        """Compute joint angles and metrics"""
        metrics = {}
        joints_data = {}
        
        # Extract keypoints to uniform list of (x,y) or None
        kp = []
        if keypoints is not None:
            for r in keypoints:
                if r is None:
                    kp.append(None)
                elif len(r) >= 2:
                    try:
                        x = float(r[0])
                        y = float(r[1])
                        if math.isnan(x) or math.isnan(y):
                            kp.append(None)
                        else:
                            kp.append((x, y))
                    except Exception:
                        kp.append(None)
                else:
                    kp.append(None)
        else:
            kp = [None] * 17
        
        # Compute raw angles
        raw_angles = {}
        for name, (a_i, b_i, c_i) in self.JOINT_DEFS.items():
            a = self._safe_point(kp[a_i]) if a_i < len(kp) else None
            b = self._safe_point(kp[b_i]) if b_i < len(kp) else None
            c = self._safe_point(kp[c_i]) if c_i < len(kp) else None
            raw_angles[name] = self._angle_between_points(a, b, c)
        
        # Apply smoothing and store metrics
        for name, ang in raw_angles.items():
            if ang is None:
                metrics[name] = None
                joints_data[name] = {'angle': None, 'velocity': 0.0, 'rom': 0.0}
            else:
                try:
                    smoothed_ang = float(self.angle_filters[name].filter([ang])[0])
                except Exception:
                    smoothed_ang = float(ang)
                metrics[name] = smoothed_ang
                joints_data[name] = {'angle': smoothed_ang, 'velocity': 0.0, 'rom': 0.0}
        
        # Update ROM
        for name, ang in metrics.items():
            if ang is None:
                continue
            mn = self.state.get(f"{name}_min", float("inf"))
            mx = self.state.get(f"{name}_max", -float("inf"))
            mn = min(mn, ang)
            mx = max(mx, ang)
            self.state[f"{name}_min"] = mn
            self.state[f"{name}_max"] = mx
            self.state[f"{name}_rom"] = mx - mn
            joints_data[name]['rom'] = mx - mn
        
        # Compute angular velocity
        for name in self.JOINT_DEFS.keys():
            curr = metrics.get(name)
            prev = self.prev_angles.get(name)
            if prev is not None and curr is not None and dt > 0:
                vel = (curr - prev) / dt
            else:
                vel = 0.0
            metrics[f"{name}_vel"] = vel
            joints_data[name]['velocity'] = vel
            self.prev_angles[name] = curr
        
        # Update angular velocity buffers
        self.time_buf.append(rel_t)
        for name in self.JOINT_DEFS.keys():
            self.angvel_buffers[name].append(metrics.get(f"{name}_vel", 0.0))
        
        return metrics, joints_data
    
    def _compute_dominance(self, metrics):
        """Compute bilateral dominance"""
        results = {}
        for left, right in self.PAIRS:
            short = left.split("_", 1)[1]
            left_rom = self.state.get(f"{left}_rom", 0.0)
            right_rom = self.state.get(f"{right}_rom", 0.0)
            left_vel = abs(metrics.get(f"{left}_vel", 0.0))
            right_vel = abs(metrics.get(f"{right}_vel", 0.0))
            
            left_score = left_rom * 1.0 + left_vel * 0.8
            right_score = right_rom * 1.0 + right_vel * 0.8
            
            # Threshold to prevent spurious tiny differences
            if left_score > right_score * 1.05 + 1e-6:
                results[short] = "LEFT"
            elif right_score > left_score * 1.05 + 1e-6:
                results[short] = "RIGHT"
            else:
                results[short] = "EQUAL"
                
        return results
    
    def _draw_skeleton_and_metrics(self, frame, keypoints, dominance, metrics):
        """Draw skeleton and metrics on frame"""
        vis = frame.copy()
        
        # Draw limbs
        limbs = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (5, 6), (11, 12)  # Shoulders and hips
        ]
        
        if keypoints is not None:
            for a, b in limbs:
                if a < len(keypoints) and b < len(keypoints):
                    pa = self._safe_point(keypoints[a])
                    pb = self._safe_point(keypoints[b])
                    if pa and pb:
                        cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), 
                                (220, 220, 220), 2, cv2.LINE_AA)
            
            # Draw keypoints
            for i in self.BODY_KP_INDICES:
                if i < len(keypoints):
                    p = self._safe_point(keypoints[i])
                    if p:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 4, (50, 200, 50), -1, cv2.LINE_AA)
        
        # Overlay metrics
        y_offset = 20
        for joint in ["knee", "hip", "elbow", "shoulder"]:
            dom = dominance.get(joint, "EQUAL")
            l_rom = self.state.get(f"left_{joint}_rom", 0.0)
            r_rom = self.state.get(f"right_{joint}_rom", 0.0)
            txt = f"{joint.title()}: {dom} (L:{l_rom:.1f}\u00b0 R:{r_rom:.1f}\u00b0)"
            cv2.putText(vis, txt, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        return vis
    
    def _create_angular_velocity_plot(self, current_time):
        """Create angular velocity plot with dark theme"""
        if len(self.time_buf) < 2:
            return None
            
        # Apply dark theme to matplotlib
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=100)
        fig.patch.set_facecolor(DarkTheme.COLORS['bg_widget'])
        ax.set_facecolor(DarkTheme.COLORS['bg_widget'])
        ax.grid(True, linestyle='--', alpha=0.3, color=DarkTheme.COLORS['border'])

        # Use theme colors for lines
        colors = [
            DarkTheme.COLORS['graph_line_1'],
            DarkTheme.COLORS['graph_line_2'], 
            DarkTheme.COLORS['graph_line_3'],
            DarkTheme.COLORS['graph_line_4'],
            DarkTheme.COLORS['graph_line_5'],
            DarkTheme.COLORS['graph_line_6'],
            DarkTheme.COLORS['graph_line_7'],
            DarkTheme.COLORS['graph_line_8']
        ]
        
        times = np.array(list(self.time_buf))
        
        for i, (joint, buffer) in enumerate(self.angvel_buffers.items()):
            if len(buffer) == 0:
                continue
                
            y = np.array(list(buffer), dtype=float)
            if len(y) >= 5:  # Apply smoothing for better visualization
                win = min(7, max(1, len(y) // 6 + 1))
                y = np.convolve(y, np.ones(win) / win, mode='same')
            
            ax.plot(times[-len(y):], y, 
                   label=joint.replace('_', ' ').title(), 
                   linewidth=2, color=colors[i % len(colors)])
        
        ax.set_xlabel("Time (s)", color=DarkTheme.COLORS['text_primary'])
        ax.set_ylabel("Angular Velocity (deg/s)", color=DarkTheme.COLORS['text_primary'])
        ax.tick_params(colors=DarkTheme.COLORS['text_secondary'])
        ax.legend(fontsize=8, facecolor=DarkTheme.COLORS['bg_widget'], 
                 edgecolor=DarkTheme.COLORS['border'])
        
        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(DarkTheme.COLORS['border'])
        
        fig.tight_layout()
        
        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        plt.close(fig)
        
        return img
    
    def _create_bilateral_display(self):
        """Create bilateral ROM comparison display with dark theme"""
        width, height = 700, 340
        canvas = np.full((height, width, 3), 
                        [int(x * 255) for x in self._hex_to_rgb(DarkTheme.COLORS['bg_widget'])], 
                        dtype=np.uint8)
        
        sections = [
            ("KNEE", "knee", 30),
            ("HIP", "hip", 110), 
            ("ELBOW", "elbow", 190),
            ("SHOULDER", "shoulder", 270)
        ]
        
        center_x = width // 2
        bar_max_width = 270
        
        for label, key, y in sections:
            left_rom = float(self.state.get(f"left_{key}_rom", 0.0) or 0.0)
            right_rom = float(self.state.get(f"right_{key}_rom", 0.0) or 0.0)
            total = max(1.0, left_rom + right_rom)
            
            left_w = int((left_rom / total) * bar_max_width) if total > 0 else 0
            right_w = int((right_rom / total) * bar_max_width) if total > 0 else 0
            
            lx1 = center_x - 10 - left_w
            lx2 = center_x - 10
            rx1 = center_x + 10
            rx2 = center_x + 10 + right_w
            
            # Use theme colors
            left_color = self._hex_to_bgr(DarkTheme.COLORS['graph_line_1']) if left_rom > right_rom else self._hex_to_bgr(DarkTheme.COLORS['bg_light'])
            right_color = self._hex_to_bgr(DarkTheme.COLORS['graph_line_2']) if right_rom > left_rom else self._hex_to_bgr(DarkTheme.COLORS['bg_light'])
            
            cv2.rectangle(canvas, (lx1, y), (lx2, y + 40), left_color, -1)
            cv2.rectangle(canvas, (rx1, y), (rx2, y + 40), right_color, -1)
            
            # Add borders
            cv2.rectangle(canvas, (lx1, y), (lx2, y + 40), self._hex_to_bgr(DarkTheme.COLORS['border']), 1)
            cv2.rectangle(canvas, (rx1, y), (rx2, y + 40), self._hex_to_bgr(DarkTheme.COLORS['border']), 1)
            
            text_color = self._hex_to_bgr(DarkTheme.COLORS['text_primary'])
            cv2.putText(canvas, label, (20, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, text_color, 2)
            cv2.putText(canvas, f"L:{left_rom:.1f}\u00b0", (max(5, lx1 + 5), y + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(canvas, f"R:{right_rom:.1f}\u00b0", (min(width - 90, rx2 - 80), y + 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        title_color = self._hex_to_bgr(DarkTheme.COLORS['text_accent'])
        cv2.putText(canvas, "Bilateral ROM Comparison", (50, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)
        
        return canvas
    
    def get_joint_names(self):
        """Get list of all tracked joints"""
        return list(self.JOINT_DEFS.keys())
    
    def get_available_metrics(self):
        """Get list of available metrics for export"""
        return ['angles', 'velocities', 'rom', 'symmetry', 'dominance']
    
    def get_frame_data(self):
        """Get all collected frame data"""
        return self.frame_data
    
    def get_summary_statistics(self):
        """Calculate summary statistics for the analysis"""
        if not self.frame_data:
            return {}
        
        summary = {}
        for joint in self.JOINT_DEFS.keys():
            angles = [frame['joints'].get(joint, {}).get('angle') for frame in self.frame_data
                     if frame['joints'].get(joint, {}).get('angle') is not None]
            if angles:
                summary[joint] = {
                    'mean_angle': float(np.mean(angles)),
                    'std_angle': float(np.std(angles)),
                    'max_angle': float(np.max(angles)),
                    'min_angle': float(np.min(angles)),
                    'mean_rom': float(self.state.get(f"{joint}_rom", 0.0))
                }
        
        return summary
    
    # Utility methods
    def _to_numpy_safe(self, x):
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        try:
            return np.asarray(x)
        except Exception:
            return None
    
    def _safe_point(self, p):
        if p is None:
            return None
        try:
            x = float(p[0])
            y = float(p[1])
            if math.isnan(x) or math.isnan(y):
                return None
            return (x, y)
        except Exception:
            return None
    
    def _angle_between_points(self, a, b, c):
        """Angle at b formed by a-b-c, in degrees (0..180)."""
        if any(v is None for v in (a, b, c)):
            return None
        ax, ay = a
        bx, by = b
        cx, cy = c
        v1 = (ax - bx, ay - by)
        v2 = (cx - bx, cy - by)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 == 0 or n2 == 0:
            return None
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cosang = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cosang))
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to normalized RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    
    def _hex_to_bgr(self, hex_color):
        """Convert hex color to BGR tuple for OpenCV"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return (rgb[2], rgb[1], rgb[0])


class ExportWorker(QThread):
    """Worker thread for export operations"""
    progress = Signal(int)
    finished = Signal(bool, str)
    
    def __init__(self, frame_data, export_options, output_dir, analyzer):
        super().__init__()
        self.frame_data = frame_data
        self.export_options = export_options
        self.output_dir = output_dir
        self.analyzer = analyzer
    
    def run(self):
        try:
            total_steps = sum([
                1 if self.export_options.get('export_csv') else 0,
                1 if self.export_options.get('export_graphs') else 0,
                1 if self.export_options.get('export_symmetry') else 0,
                1 if self.export_options.get('export_summary') else 0
            ])
            if total_steps == 0:
                total_steps = 1
            current_step = 0
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.join(self.output_dir, f"pose_analysis_{timestamp}")
            
            # Export CSV data
            if self.export_options.get('export_csv'):
                self.progress.emit(int((current_step / total_steps) * 100))
                self.export_csv_data(base_path)
                current_step += 1
            
            # Export graphs
            if self.export_options.get('export_graphs'):
                self.progress.emit(int((current_step / total_steps) * 100))
                self.export_graphs(base_path)
                current_step += 1
            
            # Export symmetry analysis
            if self.export_options.get('export_symmetry'):
                self.progress.emit(int((current_step / total_steps) * 100))
                self.export_symmetry_analysis(base_path)
                current_step += 1
            
            # Export summary report
            if self.export_options.get('export_summary'):
                self.progress.emit(int((current_step / total_steps) * 100))
                self.export_summary_report(base_path)
                current_step += 1
            
            self.progress.emit(100)
            self.finished.emit(True, "Export completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Export failed: {str(e)}")
    
    def export_csv_data(self, base_path):
        """Export joint angles and metrics to CSV"""
        csv_data = []
        selected_joints = self.export_options.get('selected_joints', [])
        
        for frame in self.frame_data:
            row = {
                'frame_number': frame.get('frame_number'),
                'timestamp': frame.get('timestamp')
            }
            
            # Add joint data
            for joint in selected_joints:
                joint_data = frame.get('joints', {}).get(joint, {})
                if self.export_options.get('include_angles'):
                    row[f"{joint}_angle"] = joint_data.get('angle', '')
                if self.export_options.get('include_velocities'):
                    row[f"{joint}_velocity"] = joint_data.get('velocity', '')
                if self.export_options.get('include_rom'):
                    row[f"{joint}_rom"] = joint_data.get('rom', '')
            
            # Add dominance data
            if self.export_options.get('include_dominance'):
                for joint, dominance in frame.get('dominance', {}).items():
                    row[f"{joint}_dominance"] = dominance
            
            csv_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(csv_data)
        
        # Save to CSV
        csv_path = f"{base_path}_joint_data.csv"
        df.to_csv(csv_path, index=False)
    
    def export_graphs(self, base_path):
        """Export graphs and visualizations"""
        graphs_dir = f"{base_path}_graphs"
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Export frame-by-frame data plots
        self.export_time_series_plots(graphs_dir)
        
        # Export summary graphs
        self.export_summary_graphs(graphs_dir)
    
    def export_time_series_plots(self, graphs_dir):
        """Export time series plots for each joint with dark theme"""
        joints = self.export_options.get('selected_joints', [])
        metrics = []
        if self.export_options.get('include_angles'):
            metrics.append('angle')
        if self.export_options.get('include_velocities'):
            metrics.append('velocity')
        if self.export_options.get('include_rom'):
            metrics.append('rom')
        
        for joint in joints:
            if not metrics:
                continue
            
            # Apply dark theme
            plt.style.use('dark_background')
            fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), dpi=150)
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                values = []
                times = []
                for frame in self.frame_data:
                    jdata = frame.get('joints', {}).get(joint, {})
                    value = jdata.get(metric)
                    if value is not None:
                        values.append(value)
                        times.append(frame.get('timestamp'))
                
                if values:
                    ax = axes[i]
                    # Use theme colors for lines
                    color = [
                        DarkTheme.COLORS['graph_line_1'],
                        DarkTheme.COLORS['graph_line_2'],
                        DarkTheme.COLORS['graph_line_3']
                    ][i % 3]
                    
                    ax.plot(times, values, label=f'{joint} {metric}', color=color, linewidth=2)
                    ax.set_xlabel('Time (s)', color=DarkTheme.COLORS['text_primary'])
                    ax.set_ylabel(f'{metric.title()} ({self.get_metric_unit(metric)})', 
                                color=DarkTheme.COLORS['text_primary'])
                    ax.set_title(f'{joint.replace("_", " ").title()} - {metric.title()}', 
                               color=DarkTheme.COLORS['text_primary'])
                    ax.grid(True, alpha=0.3, color=DarkTheme.COLORS['border'])
                    ax.legend(facecolor=DarkTheme.COLORS['bg_widget'], 
                             edgecolor=DarkTheme.COLORS['border'])
                    
                    # Set spine and tick colors
                    ax.tick_params(colors=DarkTheme.COLORS['text_secondary'])
                    for spine in ax.spines.values():
                        spine.set_color(DarkTheme.COLORS['border'])
            
            fig.patch.set_facecolor(DarkTheme.COLORS['bg_widget'])
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, f"{joint}_timeseries.png"), 
                       dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
    
    def export_summary_graphs(self, graphs_dir):
        """Export summary graphs and comparisons with dark theme"""
        # Apply dark theme
        plt.style.use('dark_background')
        
        # ROM comparison graph
        joints = self.export_options.get('selected_joints', [])
        rom_data = []
        joint_names = []
        
        for joint in joints:
            rom_values = [frame.get('joints', {}).get(joint, {}).get('rom') for frame in self.frame_data
                         if frame.get('joints', {}).get(joint, {}).get('rom') is not None]
            if rom_values:
                rom_data.append(float(np.mean(rom_values)))
                joint_names.append(joint.replace('_', '\n').title())
        
        if rom_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = [DarkTheme.COLORS['graph_line_1'], DarkTheme.COLORS['graph_line_2'], 
                     DarkTheme.COLORS['graph_line_3'], DarkTheme.COLORS['graph_line_4']]
            bars = ax.bar(joint_names, rom_data, alpha=0.8, 
                         color=colors[:len(rom_data)])
            
            ax.set_xlabel('Joints', color=DarkTheme.COLORS['text_primary'])
            ax.set_ylabel('Average ROM (degrees)', color=DarkTheme.COLORS['text_primary'])
            ax.set_title('Range of Motion Comparison', color=DarkTheme.COLORS['text_primary'])
            ax.tick_params(colors=DarkTheme.COLORS['text_secondary'])
            ax.grid(True, alpha=0.3, color=DarkTheme.COLORS['border'])
            
            # Color spines
            for spine in ax.spines.values():
                spine.set_color(DarkTheme.COLORS['border'])
            
            plt.xticks(rotation=45, ha='right')
            fig.patch.set_facecolor(DarkTheme.COLORS['bg_widget'])
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "rom_comparison.png"), 
                       dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
        
        # Bilateral symmetry graph
        bilateral_pairs = [('left_elbow', 'right_elbow'), 
                          ('left_knee', 'right_knee'),
                          ('left_hip', 'right_hip'),
                          ('left_shoulder', 'right_shoulder')]
        
        symmetry_data = []
        pair_names = []
        
        for left, right in bilateral_pairs:
            left_angles = [frame.get('joints', {}).get(left, {}).get('angle') for frame in self.frame_data
                          if frame.get('joints', {}).get(left, {}).get('angle') is not None]
            right_angles = [frame.get('joints', {}).get(right, {}).get('angle') for frame in self.frame_data
                           if frame.get('joints', {}).get(right, {}).get('angle') is not None]
            
            if left_angles and right_angles:
                min_len = min(len(left_angles), len(right_angles))
                left_angles = left_angles[:min_len]
                right_angles = right_angles[:min_len]
                differences = np.abs(np.array(left_angles) - np.array(right_angles))
                mean_diff = float(np.mean(differences))
                mean_angle = float((np.mean(left_angles) + np.mean(right_angles)) / 2.0)
                symmetry_index = max(0.0, 1.0 - (mean_diff / mean_angle)) * 100.0 if mean_angle > 0 else 0.0
                symmetry_data.append(symmetry_index)
                pair_names.append(left.split('_')[1].title())
        
        if symmetry_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if x > 80 else 'orange' if x > 60 else 'red' for x in symmetry_data]
            bars = ax.bar(pair_names, symmetry_data, color=colors, alpha=0.8)
            for bar, value in zip(bars, symmetry_data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}%', ha='center', va='bottom', 
                        color=DarkTheme.COLORS['text_primary'])
            
            ax.set_xlabel('Joint Pairs', color=DarkTheme.COLORS['text_primary'])
            ax.set_ylabel('Symmetry Index (%)', color=DarkTheme.COLORS['text_primary'])
            ax.set_title('Bilateral Joint Symmetry Analysis', color=DarkTheme.COLORS['text_primary'])
            ax.set_ylim(0, 100)
            ax.tick_params(colors=DarkTheme.COLORS['text_secondary'])
            ax.grid(True, alpha=0.3, color=DarkTheme.COLORS['border'])
            
            for spine in ax.spines.values():
                spine.set_color(DarkTheme.COLORS['border'])
            
            fig.patch.set_facecolor(DarkTheme.COLORS['bg_widget'])
            plt.tight_layout()
            plt.savefig(os.path.join(graphs_dir, "symmetry_analysis.png"), 
                       dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close(fig)
    
    def export_symmetry_analysis(self, base_path):
        """Export detailed symmetry analysis"""
        symmetry_data = []
        bilateral_pairs = [('left_elbow', 'right_elbow'), 
                          ('left_knee', 'right_knee'),
                          ('left_hip', 'right_hip'),
                          ('left_shoulder', 'right_shoulder')]
        
        for frame in self.frame_data:
            symmetry_row = {
                'frame_number': frame.get('frame_number'),
                'timestamp': frame.get('timestamp')
            }
            
            for left_joint, right_joint in bilateral_pairs:
                left_data = frame.get('joints', {}).get(left_joint, {})
                right_data = frame.get('joints', {}).get(right_joint, {})
                
                left_angle = left_data.get('angle')
                right_angle = right_data.get('angle')
                
                if left_angle is not None and right_angle is not None:
                    absolute_diff = abs(left_angle - right_angle)
                    mean_angle = (left_angle + right_angle) / 2.0
                    symmetry_ratio = (1.0 - (absolute_diff / mean_angle)) * 100.0 if mean_angle > 0 else 0.0
                    key = left_joint.split('_')[1]
                    symmetry_row[f"{key}_left_angle"] = left_angle
                    symmetry_row[f"{key}_right_angle"] = right_angle
                    symmetry_row[f"{key}_absolute_diff"] = absolute_diff
                    symmetry_row[f"{key}_symmetry_ratio"] = symmetry_ratio
            
            symmetry_data.append(symmetry_row)
        
        if symmetry_data:
            df_symmetry = pd.DataFrame(symmetry_data)
            symmetry_path = f"{base_path}_symmetry_analysis.csv"
            df_symmetry.to_csv(symmetry_path, index=False)
    
    def export_summary_report(self, base_path):
        """Export comprehensive summary report"""
        summary = self.analyzer.get_summary_statistics()
        
        with open(f"{base_path}_summary_report.txt", 'w') as f:
            f.write("POSE ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames Analyzed: {len(self.frame_data)}\n")
            f.write(f"Duration: {self.frame_data[-1]['timestamp'] if self.frame_data else 0:.2f} seconds\n\n")
            
            f.write("JOINT SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            
            for joint, stats in summary.items():
                f.write(f"\n{joint.replace('_', ' ').title()}:\n")
                f.write(f"  Mean Angle: {stats['mean_angle']:.2f}\u00b0\n")
                f.write(f"  Std Dev: {stats['std_angle']:.2f}\u00b0\n")
                f.write(f"  Range: {stats['min_angle']:.2f}\u00b0 - {stats['max_angle']:.2f}\u00b0\n")
                f.write(f"  Average ROM: {stats['mean_rom']:.2f}\u00b0\n")
    
    def get_metric_unit(self, metric):
        """Get unit for metric"""
        units = {
            'angle': 'degrees',
            'velocity': 'deg/s', 
            'rom': 'degrees'
        }
        return units.get(metric, '')


class ExportDialog(QDialog):
    def __init__(self, frame_data, available_joints, analyzer, parent=None):
        super().__init__(parent)
        self.frame_data = frame_data
        self.available_joints = available_joints
        self.analyzer = analyzer
        self.output_dir = ""
        
        self.setWindowTitle("Export Analysis Data")
        self.setModal(True)
        self.resize(700, 750)
        
        # Apply dark theme
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {DarkTheme.COLORS['bg_dark']};
                color: {DarkTheme.COLORS['text_primary']};
            }}
            QLabel {{
                color: {DarkTheme.COLORS['text_primary']};
            }}
        """)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Output directory selection
        dir_group = QGroupBox("Output Directory")
        dir_group.setStyleSheet(DarkTheme.STYLESHEETS['group_box'])
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        self.dir_label.setStyleSheet("QLabel { color: #666666; font-style: italic; }")
        self.btn_choose_dir = QPushButton("Choose Directory")
        self.btn_choose_dir.setStyleSheet(DarkTheme.STYLESHEETS['button_primary'])
        self.btn_choose_dir.clicked.connect(self.choose_directory)
        dir_layout.addWidget(self.dir_label, 1)
        dir_layout.addWidget(self.btn_choose_dir)
        dir_group.setLayout(dir_layout)
        
        # Joint selection (scrollable)
        joints_group = QGroupBox("Select Joints to Export")
        joints_group.setStyleSheet(DarkTheme.STYLESHEETS['group_box'])
        joints_layout = QGridLayout()
        self.joint_checkboxes = {}
        for i, joint in enumerate(self.available_joints):
            checkbox = QCheckBox(joint.replace('_', ' ').title())
            checkbox.setChecked(True)
            checkbox.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
            self.joint_checkboxes[joint] = checkbox
            joints_layout.addWidget(checkbox, i // 3, i % 3)
        joints_group.setLayout(joints_layout)
        scroll = QScrollArea()
        scroll.setStyleSheet(DarkTheme.STYLESHEETS['scroll_area'])
        scroll.setWidgetResizable(True)
        scroll.setWidget(joints_group)
        
        # Data options
        data_group = QGroupBox("Data to Include")
        data_group.setStyleSheet(DarkTheme.STYLESHEETS['group_box'])
        data_layout = QVBoxLayout()
        self.export_csv = QCheckBox("Export Joint Data to CSV")
        self.export_csv.setChecked(True)
        self.export_csv.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        self.include_angles = QCheckBox("Include Joint Angles")
        self.include_angles.setChecked(True)
        self.include_angles.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        self.include_velocities = QCheckBox("Include Angular Velocities") 
        self.include_velocities.setChecked(True)
        self.include_velocities.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        self.include_rom = QCheckBox("Include Range of Motion")
        self.include_rom.setChecked(True)
        self.include_rom.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        self.include_dominance = QCheckBox("Include Dominance Data")
        self.include_dominance.setChecked(True)
        self.include_dominance.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        data_layout.addWidget(self.export_csv)
        data_layout.addWidget(self.include_angles)
        data_layout.addWidget(self.include_velocities)
        data_layout.addWidget(self.include_rom)
        data_layout.addWidget(self.include_dominance)
        data_group.setLayout(data_layout)
        
        # Additional exports
        additional_group = QGroupBox("Additional Exports")
        additional_group.setStyleSheet(DarkTheme.STYLESHEETS['group_box'])
        additional_layout = QVBoxLayout()
        self.export_graphs = QCheckBox("Export Graphs and Charts")
        self.export_graphs.setChecked(True)
        self.export_graphs.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        self.export_symmetry = QCheckBox("Export Symmetry Analysis")
        self.export_symmetry.setChecked(True)
        self.export_symmetry.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        self.export_summary = QCheckBox("Export Summary Report")
        self.export_summary.setChecked(True)
        self.export_summary.setStyleSheet(DarkTheme.STYLESHEETS['checkbox'])
        additional_layout.addWidget(self.export_graphs)
        additional_layout.addWidget(self.export_symmetry)
        additional_layout.addWidget(self.export_summary)
        additional_group.setLayout(additional_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export Data")
        self.btn_export.setStyleSheet(DarkTheme.STYLESHEETS['button_primary'])
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(DarkTheme.STYLESHEETS['button_secondary'])
        self.btn_export.clicked.connect(self.execute_export)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_export.setEnabled(False)
        button_layout.addStretch()
        button_layout.addWidget(self.btn_export)
        button_layout.addWidget(self.btn_cancel)
        
        # Add all to main layout
        layout.addWidget(dir_group)
        layout.addWidget(scroll, 1)
        layout.addWidget(data_group)
        layout.addWidget(additional_group)
        layout.addLayout(button_layout)
    
    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.dir_label.setText(directory)
            self.dir_label.setStyleSheet("QLabel { color: #ffffff; font-style: normal; }")
            self.btn_export.setEnabled(True)
    
    def execute_export(self):
        if not self.output_dir:
            QMessageBox.warning(self, "No Directory", "Please select an output directory.")
            return
        
        selected_joints = [joint for joint, checkbox in self.joint_checkboxes.items() 
                          if checkbox.isChecked()]
        
        if not selected_joints and self.export_csv.isChecked():
            QMessageBox.warning(self, "No Joints Selected", 
                              "Please select at least one joint to export.")
            return
        
        # Prepare export options
        export_options = {
            'selected_joints': selected_joints,
            'export_csv': self.export_csv.isChecked(),
            'include_angles': self.include_angles.isChecked(),
            'include_velocities': self.include_velocities.isChecked(),
            'include_rom': self.include_rom.isChecked(),
            'include_dominance': self.include_dominance.isChecked(),
            'export_graphs': self.export_graphs.isChecked(),
            'export_symmetry': self.export_symmetry.isChecked(),
            'export_summary': self.export_summary.isChecked()
        }
        
        # Create progress dialog with dark theme
        progress_dialog = QProgressDialog("Exporting data...", "Cancel", 0, 100, self)
        progress_dialog.setWindowTitle("Export Progress")
        progress_dialog.setModal(True)
        progress_dialog.setStyleSheet(f"""
            QProgressDialog {{
                background-color: {DarkTheme.COLORS['bg_dark']};
                color: {DarkTheme.COLORS['text_primary']};
            }}
            QLabel {{
                color: {DarkTheme.COLORS['text_primary']};
            }}
        """ + DarkTheme.STYLESHEETS['progress_bar'])
        progress_dialog.show()
        
        # Create and start worker thread
        self.worker = ExportWorker(self.frame_data, export_options, self.output_dir, self.analyzer)
        self.worker.progress.connect(progress_dialog.setValue)
        self.worker.finished.connect(lambda success, msg: self.export_finished(success, msg, progress_dialog))
        self.worker.start()
    
    def export_finished(self, success, message, progress_dialog):
        progress_dialog.close()
        
        if success:
            QMessageBox.information(self, "Export Complete", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Export Failed", message)


class VideoDisplay(QLabel):
    """Custom widget for video display with dark theme"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setText("No video loaded")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(DarkTheme.STYLESHEETS['video_display'])


class MetricsDisplay(QLabel):
    """Custom widget for metrics display with dark theme"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = QFont('Consolas', 9)
        self.setFont(font)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {DarkTheme.COLORS['bg_widget']};
                color: {DarkTheme.COLORS['text_primary']};
                border: 1px solid {DarkTheme.COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: Consolas, monospace;
            }}
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Analysis Tool - Professional Edition")
        
        # Apply dark theme to main window
        self.setStyleSheet(DarkTheme.STYLESHEETS['main_window'])

        # Start maximized to adapt to user's resolution but allow manual resize
        self.showMaximized()
        
        # Initialize analyzer
        self.analyzer = PoseAnalyzer()
        self.is_processing = False
        self.current_video_path = ""
        
        self.init_ui()
        self.init_menu()

    def init_ui(self):
        """Initialize the main UI with dark theme"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {DarkTheme.COLORS['border']};
            }}
            QSplitter::handle:hover {{
                background-color: {DarkTheme.COLORS['border_light']};
            }}
        """)
        
        # Left panel - video display and controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        # Video display
        self.video_display = VideoDisplay()
        left_layout.addWidget(self.video_display, 8)
        
        # Controls
        controls_layout = QHBoxLayout()
        self.btn_play = QPushButton("Start Analysis")
        self.btn_play.setStyleSheet(DarkTheme.STYLESHEETS['button_primary'])
        self.btn_play.clicked.connect(self.toggle_analysis)
        self.btn_play.setEnabled(False)
        
        self.btn_export = QPushButton("Export Data")
        self.btn_export.setStyleSheet(DarkTheme.STYLESHEETS['button_secondary'])
        self.btn_export.clicked.connect(self.show_export_dialog)
        self.btn_export.setEnabled(False)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.setStyleSheet(DarkTheme.STYLESHEETS['progress_bar'])
        
        controls_layout.addWidget(self.btn_play)
        controls_layout.addWidget(self.btn_export)
        controls_layout.addWidget(self.progress_bar, 1)
        controls_layout.addStretch()
        left_layout.addLayout(controls_layout)
        
        # Right panel - metrics and graphs
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Metrics display in a scroll area so content is always visible
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.StyledPanel)
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        
        self.metrics_label = MetricsDisplay()
        metrics_layout.addWidget(self.metrics_label)
        
        metrics_scroll = QScrollArea()
        metrics_scroll.setStyleSheet(DarkTheme.STYLESHEETS['scroll_area'])
        metrics_scroll.setWidget(metrics_frame)
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setMinimumHeight(180)
        right_layout.addWidget(metrics_scroll, 2)
        
        # Graph displays
        self.graph_display = VideoDisplay()
        self.graph_display.setMinimumHeight(240)
        self.graph_display.setText("Angular velocity graph")
        right_layout.addWidget(self.graph_display, 4)
        
        self.bilateral_display = VideoDisplay()
        self.bilateral_display.setMinimumHeight(200)
        self.bilateral_display.setText("Bilateral comparison")
        right_layout.addWidget(self.bilateral_display, 3)
        
        # Add to splitter and set stretch so layouts adapt dynamically
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(DarkTheme.STYLESHEETS['status_bar'])
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to load video")
        
        # Timer for video processing
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

    def create_toolbar(self):
        """Create the main toolbar with dark theme"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setStyleSheet(DarkTheme.STYLESHEETS['toolbar'])
        self.addToolBar(toolbar)
        
        # Load video action
        load_action = QAction("Load Video", self)
        load_action.triggered.connect(self.load_video)
        toolbar.addAction(load_action)
        
        toolbar.addSeparator()
        
        # Export action
        export_action = QAction("Export Data", self)
        export_action.triggered.connect(self.show_export_dialog)
        toolbar.addAction(export_action)

    def init_menu(self):
        """Initialize the menu bar with dark theme"""
        menubar = self.menuBar()
        menubar.setStyleSheet(f"""
            QMenuBar {{
                background-color: {DarkTheme.COLORS['bg_medium']};
                color: {DarkTheme.COLORS['text_primary']};
                border: none;
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 8px;
            }}
            QMenuBar::item:selected {{
                background-color: {DarkTheme.COLORS['hover']};
            }}
            QMenu {{
                background-color: {DarkTheme.COLORS['bg_medium']};
                color: {DarkTheme.COLORS['text_primary']};
                border: 1px solid {DarkTheme.COLORS['border']};
            }}
            QMenu::item {{
                padding: 4px 20px;
            }}
            QMenu::item:selected {{
                background-color: {DarkTheme.COLORS['hover']};
            }}
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Video", self)
        load_action.triggered.connect(self.load_video)
        file_menu.addAction(load_action)
        
        export_action = QAction("Export Data", self)
        export_action.triggered.connect(self.show_export_dialog)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def load_video(self):
        """Load video file for analysis"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if path:
            try:
                self.status_bar.showMessage("Loading video...")
                success = self.analyzer.load_video(path)
                
                if success:
                    self.current_video_path = path
                    self.btn_play.setEnabled(True)
                    self.btn_export.setEnabled(True)
                    
                    # Show first frame (safe call)
                    self.show_first_frame()
                    
                    self.status_bar.showMessage(f"Loaded: {os.path.basename(path)}")
                    
                    # Update progress bar
                    total_frames = max(1, int(getattr(self.analyzer, 'total_frames', 0)))
                    self.progress_bar.setMaximum(total_frames)
                    self.progress_bar.setValue(0)
                    self.progress_bar.setVisible(True)
                    
                else:
                    QMessageBox.critical(self, "Error", "Failed to load video file")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")

    def show_first_frame(self):
        """Display the first frame of the video"""
        success, frame, plot_img, bilateral_img, metrics, dominance = self.analyzer.process_next_frame()
        if success:
            self.update_video_display(frame)
            self.update_graph_display(plot_img)
            self.update_bilateral_display(bilateral_img)
            self.update_metrics_display(metrics, dominance)

    def toggle_analysis(self):
        """Start or stop the analysis"""
        if not self.is_processing:
            self.start_analysis()
        else:
            self.stop_analysis()

    def start_analysis(self):
        """Start video analysis"""
        self.is_processing = True
        self.btn_play.setText("Stop Analysis")
        # aim for roughly the video's fps but cap to a reasonable interval
        interval = int(max(1, 1000.0 / max(1.0, getattr(self.analyzer, 'fps', 25.0))))
        self.timer.start(interval)
        self.status_bar.showMessage("Analysis running...")

    def stop_analysis(self):
        """Stop video analysis"""
        self.is_processing = False
        self.btn_play.setText("Start Analysis")
        self.timer.stop()
        self.status_bar.showMessage("Analysis stopped")

    def process_frame(self):
        """Process a single video frame"""
        if not self.analyzer.cap or not self.analyzer.cap.isOpened():
            self.stop_analysis()
            return
        
        success, frame, plot_img, bilateral_img, metrics, dominance = self.analyzer.process_next_frame()
        
        if success:
            self.update_video_display(frame)
            self.update_graph_display(plot_img)
            self.update_bilateral_display(bilateral_img)
            self.update_metrics_display(metrics, dominance)
            
            # Update progress
            current_frame = int(getattr(self.analyzer, 'current_frame', 0))
            total_frames = max(1, int(getattr(self.analyzer, 'total_frames', 0)))
            try:
                self.progress_bar.setValue(min(current_frame, total_frames))
            except Exception:
                pass
            
            # Check if we've reached the end
            if current_frame >= total_frames and total_frames > 0:
                self.stop_analysis()
                self.status_bar.showMessage("Analysis completed")
        else:
            # If processing naturally ended or failed, stop
            self.stop_analysis()
            self.status_bar.showMessage("Analysis finished or error occurred")

    def update_video_display(self, frame):
        """Update the main video display"""
        if frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_display.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.video_display.width(), 
                self.video_display.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def update_graph_display(self, plot_img):
        """Update the graph display"""
        if plot_img is not None:
            plot_img_rgb = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
            h, w, ch = plot_img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(plot_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.graph_display.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.graph_display.width(),
                self.graph_display.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def update_bilateral_display(self, bilateral_img):
        """Update the bilateral comparison display"""
        if bilateral_img is not None:
            bilateral_img_rgb = cv2.cvtColor(bilateral_img, cv2.COLOR_BGR2RGB)
            h, w, ch = bilateral_img_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(bilateral_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.bilateral_display.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.bilateral_display.width(),
                self.bilateral_display.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))

    def update_metrics_display(self, metrics, dominance):
        """Update the metrics text display"""
        if metrics is None:
            return
            
        metrics_text = "CURRENT METRICS:\n"
        metrics_text += "=" * 60 + "\n"
        
        # Add joint angles and velocities
        for joint in self.analyzer.JOINT_DEFS.keys():
            angle = metrics.get(joint)
            velocity = metrics.get(f"{joint}_vel", 0.0)
            rom = float(self.analyzer.state.get(f"{joint}_rom", 0.0) or 0.0)
            
            if angle is not None:
                metrics_text += f"{joint.replace('_', ' ').title():<20}: {angle:7.1f}\u00b0 | {velocity:7.1f}\u00b0/s | ROM: {rom:7.1f}\u00b0\n"
        
        metrics_text += "\nDOMINANCE ANALYSIS:\n"
        metrics_text += "=" * 60 + "\n"
        
        for joint, dom in dominance.items():
            metrics_text += f"{joint.title():<12}: {dom}\n"
        
        self.metrics_label.setText(metrics_text)

    def show_export_dialog(self):
        """Show the export configuration dialog"""
        if not self.analyzer.frame_data:
            QMessageBox.warning(self, "No Data", "No analysis data available. Please run analysis first.")
            return
        
        dialog = ExportDialog(
            self.analyzer.frame_data,
            self.analyzer.get_joint_names(),
            self.analyzer,
            self
        )
        dialog.exec()

    def show_about(self):
        """Show about dialog with dark theme"""
        about_text = f"""
        <h2 style="color: {DarkTheme.COLORS['text_accent']};">Pose Analysis Tool</h2>
        <p style="color: {DarkTheme.COLORS['text_primary']};">Professional motion analysis software for biomechanical assessment.</p>
        <p style="color: {DarkTheme.COLORS['text_primary']};"><b>Features:</b></p>
        <ul style="color: {DarkTheme.COLORS['text_primary']};">
            <li>Real-time pose estimation using YOLOv8</li>
            <li>Joint angle and range of motion analysis</li>
            <li>Bilateral symmetry assessment</li>
            <li>Comprehensive data export capabilities</li>
            <li>Professional visualization tools</li>
        </ul>
        <p style="color: {DarkTheme.COLORS['text_primary']};"><b>Version:</b> 1.0.0</p>
        <p style="color: {DarkTheme.COLORS['text_primary']};"><b>Powered by:</b> PySide6, OpenCV, Ultralytics YOLO</p>
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("About Pose Analysis Tool")
        msg.setText(about_text)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: {DarkTheme.COLORS['bg_dark']};
                color: {DarkTheme.COLORS['text_primary']};
            }}
            QMessageBox QLabel {{
                color: {DarkTheme.COLORS['text_primary']};
            }}
            QPushButton {{
                {DarkTheme.STYLESHEETS['button_secondary']}
            }}
        """)
        msg.exec()

    def closeEvent(self, event):
        """Handle application closure"""
        if self.is_processing:
            self.stop_analysis()
        
        if hasattr(self.analyzer, 'cap') and self.analyzer.cap:
            try:
                self.analyzer.cap.release()
            except Exception:
                pass
        
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Pose Analysis Tool")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Biomechanics Lab")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()