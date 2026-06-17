"""
dashboard.py
────────────
A premium, dark-themed Tkinter dashboard for YOLOv11 Palm Cooking Oil Detection.
Provides image selection, interactive threshold tuning (Conf/IoU) via sliders,
and real-time inference visualization.
"""

import sys
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

try:
    import torch
    from ultralytics import YOLO
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}")
    print("Please make sure 'ultralytics' and 'torch' are installed.")
    sys.exit(1)

# Configure encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


class PalmOilDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Palm Cooking Oil — YOLOv11 Dashboard")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 720)

        # Style colors (Premium Dark Mode)
        self.bg_dark = "#111115"
        self.bg_panel = "#181822"
        self.bg_card = "#222230"
        self.accent_color = "#3b82f6"  # Electric Blue
        self.accent_green = "#10b981"  # Emerald
        self.accent_red = "#ef4444"    # Rose Red
        self.text_light = "#f4f4f7"
        self.text_muted = "#a1a1aa"

        # Apply root background
        self.root.configure(bg=self.bg_dark)

        # State Variables
        self.weights_path = "palm_oil_yolov11/train_v1/weights/best.pt"
        self.model = None
        self.current_image_path = None
        self.original_img = None
        self.processed_img = None
        self.debounce_job = None

        # Try to load YOLO model
        self.load_yolo_model()

        # Build UI layout
        self.setup_ui()

    def load_yolo_model(self):
        if not Path(self.weights_path).exists():
            # Fallback to general yolo11m.pt if custom weights not found yet
            self.weights_path = "yolo11m.pt"
            print(f"⚠️ Custom weights not found. Falling back to base model: {self.weights_path}")
        try:
            self.model = YOLO(self.weights_path)
            # Map GPU if available
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            print(f"✅ Loaded YOLO model weights from {self.weights_path} onto {device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def setup_ui(self):
        # Configure Styles for TTK widgets
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background=self.bg_panel, foreground=self.text_light)
        style.configure("TFrame", background=self.bg_panel)
        style.configure("TLabel", background=self.bg_panel, foreground=self.text_light)
        
        # Horizontal Paned Window for layout scaling
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Left Sidebar (Controls & Stats)
        sidebar = tk.Frame(main_pane, bg=self.bg_panel, width=340)
        sidebar.pack_propagate(False)
        main_pane.add(sidebar, weight=1)

        # Right Panel (Viewport Area)
        viewport = tk.Frame(main_pane, bg=self.bg_dark)
        main_pane.add(viewport, weight=4)

        # --- SIDEBAR CONTENTS ---
        # Header/Title Card
        header_card = tk.Frame(sidebar, bg=self.bg_card, bd=0, padx=15, pady=15)
        header_card.pack(fill=tk.X, padx=10, pady=10)

        title_lbl = tk.Label(
            header_card, text="PALM OIL DETECTOR",
            fg=self.text_light, bg=self.bg_card,
            font=("Segoe UI", 16, "bold")
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            header_card, text=f"Model: {Path(self.weights_path).name}",
            fg=self.text_muted, bg=self.bg_card,
            font=("Segoe UI", 9)
        )
        subtitle_lbl.pack(anchor="w", pady=(2, 0))

        # Controls Card
        controls_card = tk.Frame(sidebar, bg=self.bg_card, padx=15, pady=15)
        controls_card.pack(fill=tk.X, padx=10, pady=5)

        # Image Load Button
        self.btn_load = tk.Button(
            controls_card, text="📁 Choose Image",
            bg=self.accent_color, fg=self.text_light,
            activebackground="#2563eb", activeforeground=self.text_light,
            bd=0, font=("Segoe UI", 11, "bold"),
            padx=10, pady=8, cursor="hand2",
            command=self.choose_image
        )
        self.btn_load.pack(fill=tk.X, pady=(0, 15))

        # Confidence Slider
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_lbl_frame = tk.Frame(controls_card, bg=self.bg_card)
        conf_lbl_frame.pack(fill=tk.X, pady=(5, 2))
        
        tk.Label(conf_lbl_frame, text="Confidence Threshold", fg=self.text_light, bg=self.bg_card, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.conf_val_lbl = tk.Label(conf_lbl_frame, text="0.25", fg=self.accent_color, bg=self.bg_card, font=("Segoe UI", 10, "bold"))
        self.conf_val_lbl.pack(side=tk.RIGHT)

        self.conf_slider = tk.Scale(
            controls_card, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.conf_var,
            bg=self.bg_card, fg=self.text_light,
            activebackground=self.accent_color, troughcolor=self.bg_panel,
            bd=0, highlightthickness=0,
            command=self.on_slider_changed
        )
        self.conf_slider.pack(fill=tk.X, pady=(0, 10))

        # IoU Slider
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_lbl_frame = tk.Frame(controls_card, bg=self.bg_card)
        iou_lbl_frame.pack(fill=tk.X, pady=(5, 2))
        
        tk.Label(iou_lbl_frame, text="IoU Threshold (NMS)", fg=self.text_light, bg=self.bg_card, font=("Segoe UI", 10)).pack(side=tk.LEFT)
        self.iou_val_lbl = tk.Label(iou_lbl_frame, text="0.45", fg=self.accent_color, bg=self.bg_card, font=("Segoe UI", 10, "bold"))
        self.iou_val_lbl.pack(side=tk.RIGHT)

        self.iou_slider = tk.Scale(
            controls_card, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.iou_var,
            bg=self.bg_card, fg=self.text_light,
            activebackground=self.accent_color, troughcolor=self.bg_panel,
            bd=0, highlightthickness=0,
            command=self.on_slider_changed
        )
        self.iou_slider.pack(fill=tk.X, pady=(0, 5))

        # Results & Verdict Card
        self.results_card = tk.Frame(sidebar, bg=self.bg_card, padx=15, pady=15)
        self.results_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.verdict_header = tk.Label(
            self.results_card, text="DETECTION STATUS",
            fg=self.text_muted, bg=self.bg_card,
            font=("Segoe UI", 10, "bold")
        )
        self.verdict_header.pack(anchor="w")

        # Huge Verdict Indicator
        self.verdict_lbl = tk.Label(
            self.results_card, text="No Image Loaded",
            fg=self.text_muted, bg=self.bg_card,
            font=("Segoe UI", 16, "bold"),
            pady=10
        )
        self.verdict_lbl.pack(anchor="w")

        # Detailed Detections List/Text Box
        tk.Label(
            self.results_card, text="Detailed Detections:",
            fg=self.text_light, bg=self.bg_card,
            font=("Segoe UI", 10, "bold")
        )
        self.results_card.pack_propagate(False) # Keep fixed height/bounds
        
        self.details_box = tk.Text(
            self.results_card, bg=self.bg_panel, fg=self.text_light,
            insertbackground=self.text_light, bd=0, wrap=tk.WORD,
            font=("Consolas", 9), padx=10, pady=10
        )
        self.details_box.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self.details_box.config(state=tk.DISABLED)

        # --- VIEWPORT CONTENTS ---
        # Top banner info
        vp_header = tk.Frame(viewport, bg=self.bg_dark, pady=5)
        vp_header.pack(fill=tk.X)
        self.status_bar_lbl = tk.Label(
            vp_header, text="Select an image file above to begin...",
            fg=self.text_muted, bg=self.bg_dark,
            font=("Segoe UI", 10, "italic")
        )
        self.status_bar_lbl.pack(side=tk.LEFT)

        # Images Viewport Container (Original vs Detected Side-By-Side)
        self.images_container = tk.Frame(viewport, bg=self.bg_dark)
        self.images_container.pack(fill=tk.BOTH, expand=True)

        self.canvas_orig = tk.Canvas(self.images_container, bg=self.bg_panel, bd=0, highlightthickness=0)
        self.canvas_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        self.canvas_det = tk.Canvas(self.images_container, bg=self.bg_panel, bd=0, highlightthickness=0)
        self.canvas_det.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)

        # Bind resize events to scale images properly
        self.root.bind("<Configure>", self.on_window_resized)

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.JPG *.PNG")]
        )
        if file_path:
            self.current_image_path = file_path
            self.original_img = Image.open(file_path)
            self.processed_img = None
            self.status_bar_lbl.config(text=f"Loaded: {Path(file_path).name}", fg=self.text_light)
            self.run_inference()

    def on_slider_changed(self, event):
        # Live slider values text update
        self.conf_val_lbl.config(text=f"{self.conf_var.get():.2f}")
        self.iou_val_lbl.config(text=f"{self.iou_var.get():.2f}")

        # Debounce the re-inference run to keep slider interaction smooth
        if self.current_image_path:
            if self.debounce_job:
                self.root.after_cancel(self.debounce_job)
            self.debounce_job = self.root.after(200, self.run_inference)

    def run_inference(self):
        if not self.current_image_path or not self.model:
            return

        t_start = time.time()
        conf = self.conf_var.get()
        iou = self.iou_var.get()

        # Run model inference
        results = self.model.predict(
            source=self.current_image_path,
            conf=conf,
            iou=iou,
            save=False,
            show=False,
            verbose=False
        )
        latency = (time.time() - t_start) * 1000

        # Retrieve prediction rendering
        # YOLO results[0].plot() returns a numpy array in BGR format
        res_bgr = results[0].plot()
        # Convert BGR (OpenCV format) to RGB (Pillow format)
        res_rgb = res_bgr[..., ::-1]
        self.processed_img = Image.fromarray(res_rgb)

        # Populate Stats & Verdict
        boxes = results[0].boxes
        num_dets = len(boxes)
        detectable = num_dets > 0

        # Update Verdict Display
        if detectable:
            self.verdict_lbl.config(text="DETECTABLE (YES)", fg=self.accent_green)
        else:
            self.verdict_lbl.config(text="UNDETECTABLE (NO)", fg=self.accent_red)

        # Update details textbox
        self.details_box.config(state=tk.NORMAL)
        self.details_box.delete("1.0", tk.END)
        
        self.details_box.insert(tk.END, f"Inference Time: {latency:.1f} ms\n")
        self.details_box.insert(tk.END, f"Detections found: {num_dets}\n")
        self.details_box.insert(tk.END, "-"*30 + "\n")

        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            name = results[0].names[cls_id]
            score = float(box.conf[0])
            self.details_box.insert(tk.END, f"• {name} ({score:.2%})\n")
            class_counts[name] = class_counts.get(name, 0) + 1

        if class_counts:
            self.details_box.insert(tk.END, "\nSummary:\n")
            for name, count in class_counts.items():
                self.details_box.insert(tk.END, f"  {name}: {count}\n")

        self.details_box.config(state=tk.DISABLED)

        # Update image canvases
        self.render_canvases()

    def render_canvases(self):
        if not self.original_img:
            return

        # Render original image to left canvas
        self.draw_image_on_canvas(self.canvas_orig, self.original_img, "Original Image")

        # Render annotated image to right canvas (if exists)
        if self.processed_img:
            self.draw_image_on_canvas(self.canvas_det, self.processed_img, "Detected Detections")

    def draw_image_on_canvas(self, canvas, img, label_text):
        canvas.delete("all")
        
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return # Skip if window hasn't laid out yet

        # Fit image while keeping aspect ratio
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        rw = int(iw * scale)
        rh = int(ih * scale)

        if rw <= 0 or rh <= 0:
            return

        resized = img.resize((rw, rh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        # Save reference so it doesn't get garbage collected
        if canvas == self.canvas_orig:
            self._tk_img_orig = tk_img
        else:
            self._tk_img_det = tk_img

        # Center image in canvas
        x = (cw - rw) // 2
        y = (ch - rh) // 2
        canvas.create_image(x, y, anchor="nw", image=tk_img)

        # Add text label overlay at the top left corner of canvas
        canvas.create_rectangle(5, 5, 140, 28, fill="#000000", outline="", stipple="gray50")
        canvas.create_text(10, 10, anchor="nw", text=label_text, fill=self.text_light, font=("Segoe UI", 9, "bold"))

    def on_window_resized(self, event):
        # Trigger re-render of canvases on resize to keep it perfectly fit
        if self.original_img:
            self.render_canvases()


def main():
    root = tk.Tk()
    app = PalmOilDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
