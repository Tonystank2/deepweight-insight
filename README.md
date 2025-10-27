Barbell Tracker Pro — README

A polished GUI tool for visual weight/barbell tracking & analytics built with OpenCV (YOLO-friendly) and PySide6.
Use it to draw an ROI on a video, track the barbell (or object), visualize speed/acceleration/dwell heatmaps, select segments, and export an annotated video.


Quickstart

Clone the repo

cd deepweight-insight

Run the run.exe
























What it does (features)

GUI video player with zoomable main view + analytics view.

Draw bounding box (ROI) with left mouse to initialize tracking.

Right-click timeline to create selection ranges (labelled & colorized).

Per-selection stats: total distance, vertical ROM, horizontal sway, avg/max/min speed, avg/max acceleration, dwell time.

Visualizations: Speed, Acceleration, Dwell (heatmap color modes).

Record annotated output to barbell_tracked.mp4 (default).

Sensitivity controls for speed/acceleration/dwell thresholds.

Export/launch optional bodyui.py if included.
