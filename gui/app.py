import os
import sys
import time
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk

from core.face_engine import FaceEngine
from core.database import FaceDatabase
from core.attendance import AttendanceLogger

class ModernFaceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FaceEngine AI • Offline Face Recognition & Attendance System")
        self.root.geometry("1180x780")
        self.root.minsize(1020, 680)
        self.root.configure(bg="#0f111a")

        # Color Palette - Professional Deep Slate & Indigo Theme
        self.bg_root = "#0f111a"
        self.bg_sidebar = "#161925"
        self.bg_card = "#1c2030"
        self.bg_card_hover = "#242a3e"
        self.bg_accent = "#252b42"
        self.border_color = "#2d354d"
        
        self.primary = "#4f46e5"          # Indigo
        self.primary_hover = "#4338ca"
        self.success = "#10b981"          # Emerald
        self.success_hover = "#059669"
        self.danger = "#ef4444"           # Red
        self.danger_hover = "#dc2626"
        self.warning = "#f59e0b"          # Amber
        self.cyan = "#06b6d4"             # Cyan
        
        self.text_title = "#ffffff"
        self.text_main = "#f1f5f9"
        self.text_muted = "#94a3b8"
        self.text_dim = "#64748b"

        # Core Engines
        self.engine = FaceEngine(detector_score_threshold=0.75, cosine_threshold=0.38)
        self.database = FaceDatabase(engine=self.engine)
        self.attendance = AttendanceLogger()

        # Camera state
        self.cap = None
        self.is_camera_running = False
        self.camera_index = 0
        self.current_frame = None
        self.last_fps_time = time.time()
        self.fps = 0.0

        # Live Enrollment state
        self.is_enrolling = False
        self.enroll_target_name = ""
        self.enroll_samples = []
        self.enroll_target_count = 5
        self.enroll_countdown = 0

        # UI State
        self.status_var = tk.StringVar(value="System Ready. Zero external APIs • 100% Offline.")
        self.nav_buttons = {}
        self.active_page = None

        self._configure_styles()
        self._build_main_layout()
        self._refresh_db_stats()

        # Safe shutdown
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Modern Treeview Styling
        style.configure("Treeview",
                        background=self.bg_card,
                        foreground=self.text_main,
                        fieldbackground=self.bg_card,
                        rowheight=32,
                        font=("Segoe UI", 9),
                        borderwidth=0)
        style.configure("Treeview.Heading",
                        background=self.bg_accent,
                        foreground=self.text_title,
                        font=("Segoe UI", 10, "bold"),
                        relief="flat",
                        padding=[8, 6])
        style.map("Treeview",
                  background=[("selected", self.primary)],
                  foreground=[("selected", "#ffffff")])
        style.map("Treeview.Heading",
                  background=[("active", self.primary)])

        # Scrollbar
        style.configure("Vertical.TScrollbar",
                        background=self.bg_accent,
                        troughcolor=self.bg_sidebar,
                        borderwidth=0,
                        arrowsize=12)

    def _build_main_layout(self):
        # 1. Left Sidebar Navigation
        self.sidebar = tk.Frame(self.root, bg=self.bg_sidebar, width=240)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Brand / Logo Header
        brand_frame = tk.Frame(self.sidebar, bg=self.bg_sidebar, pady=22, padx=20)
        brand_frame.pack(fill=tk.X)

        brand_title = tk.Label(brand_frame, text="⚡ FACE ENGINE", font=("Segoe UI", 14, "bold"),
                               bg=self.bg_sidebar, fg="#818cf8")
        brand_title.pack(anchor="w")

        brand_sub = tk.Label(brand_frame, text="Open-Source • Local AI", font=("Segoe UI", 8),
                             bg=self.bg_sidebar, fg=self.text_dim)
        brand_sub.pack(anchor="w")

        sep = tk.Frame(self.sidebar, bg=self.border_color, height=1)
        sep.pack(fill=tk.X, padx=15, pady=(0, 15))

        # Nav Buttons Container
        self.nav_container = tk.Frame(self.sidebar, bg=self.bg_sidebar)
        self.nav_container.pack(fill=tk.X, padx=12)

        nav_items = [
            ("camera", "📹  Live Camera", self.show_camera_page),
            ("enroll", "➕  Enroll Person", self.show_enroll_page),
            ("inspect", "🖼️  Inspect Image", self.show_inspect_page),
            ("database", "👥  Identities DB", self.show_database_page),
            ("attendance", "📋  Attendance Log", self.show_attendance_page),
            ("settings", "⚙️  Tuning & Sliders", self.show_settings_page),
        ]

        for key, text, cmd in nav_items:
            btn = tk.Button(
                self.nav_container,
                text=text,
                font=("Segoe UI", 10, "bold"),
                anchor="w",
                padx=16,
                pady=10,
                bg=self.bg_sidebar,
                fg=self.text_muted,
                activebackground=self.bg_card,
                activeforeground="#ffffff",
                relief=tk.FLAT,
                cursor="hand2",
                command=cmd
            )
            btn.pack(fill=tk.X, pady=3)
            self.nav_buttons[key] = btn

        # Bottom System Info in Sidebar
        system_box = tk.Frame(self.sidebar, bg=self.bg_card, padx=14, pady=12)
        system_box.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=16)

        tk.Label(system_box, text="MODELS STATUS", font=("Segoe UI", 8, "bold"),
                 bg=self.bg_card, fg=self.cyan).pack(anchor="w")
        tk.Label(system_box, text="• Detector: YuNet CNN\n• Recognizer: SFace\n• Mode: 100% Offline",
                 font=("Segoe UI", 8), justify="left", bg=self.bg_card, fg=self.text_muted).pack(anchor="w", pady=(3, 0))

        # 2. Main Content Wrapper (Right Side)
        self.main_wrapper = tk.Frame(self.root, bg=self.bg_root)
        self.main_wrapper.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Top Header Bar
        self.top_bar = tk.Frame(self.main_wrapper, bg=self.bg_card, height=58, padx=24)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        self.top_bar.pack_propagate(False)

        self.page_title_lbl = tk.Label(self.top_bar, text="Live Camera Recognition",
                                       font=("Segoe UI", 13, "bold"), bg=self.bg_card, fg=self.text_title)
        self.page_title_lbl.pack(side=tk.LEFT, pady=14)

        # Top Badges on Right
        badges_frame = tk.Frame(self.top_bar, bg=self.bg_card)
        badges_frame.pack(side=tk.RIGHT, pady=12)

        self.badge_registered = tk.Label(badges_frame, text="Identities: 0", font=("Segoe UI", 9, "bold"),
                                         bg=self.bg_accent, fg=self.cyan, padx=10, pady=4)
        self.badge_registered.pack(side=tk.RIGHT, padx=6)

        self.badge_fps = tk.Label(badges_frame, text="FPS: 0", font=("Segoe UI", 9, "bold"),
                                  bg=self.bg_accent, fg=self.success, padx=10, pady=4)
        self.badge_fps.pack(side=tk.RIGHT, padx=6)

        # Content Area where pages are switched
        self.content_area = tk.Frame(self.main_wrapper, bg=self.bg_root, padx=18, pady=14)
        self.content_area.pack(fill=tk.BOTH, expand=True)

        # Bottom Global Status Bar
        self.status_bar = tk.Frame(self.main_wrapper, bg=self.bg_sidebar, height=28, padx=18)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar.pack_propagate(False)

        self.status_lbl = tk.Label(self.status_bar, textvariable=self.status_var, font=("Segoe UI", 9),
                                   bg=self.bg_sidebar, fg=self.text_muted)
        self.status_lbl.pack(side=tk.LEFT)

        # Initialize and build all pages
        self.pages = {}
        self._init_camera_page()
        self._init_enroll_page()
        self._init_inspect_page()
        self._init_database_page()
        self._init_attendance_page()
        self._init_settings_page()

        # Show initial page
        self.show_camera_page()

    def _set_active_nav(self, active_key: str, title: str):
        self.page_title_lbl.config(text=title)
        for key, btn in self.nav_buttons.items():
            if key == active_key:
                btn.config(bg=self.primary, fg="#ffffff")
            else:
                btn.config(bg=self.bg_sidebar, fg=self.text_muted)

        # Hide current page and show new page
        if self.active_page:
            self.active_page.pack_forget()

        page = self.pages.get(active_key)
        if page:
            page.pack(fill=tk.BOTH, expand=True)
            self.active_page = page

    # =========================================================================
    # PAGE 1: LIVE CAMERA RECOGNITION
    # =========================================================================
    def _init_camera_page(self):
        page = tk.Frame(self.content_area, bg=self.bg_root)
        self.pages["camera"] = page

        # Split: Left (Canvas) | Right (Action Panel)
        left_box = tk.Frame(page, bg=self.bg_card, highlightbackground=self.border_color, highlightthickness=1)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        self.video_canvas = tk.Canvas(left_box, bg="#08090f", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Right Panel
        right_panel = tk.Frame(page, bg=self.bg_card, width=300, padx=16, pady=16)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        tk.Label(right_panel, text="Camera Controls", font=("Segoe UI", 11, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(anchor="w", pady=(0, 10))

        # Main Start / Stop Button
        self.btn_toggle_cam = tk.Button(
            right_panel,
            text="▶  Start Camera",
            font=("Segoe UI", 11, "bold"),
            bg=self.success,
            fg="#ffffff",
            activebackground=self.success_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=10,
            command=self.toggle_camera
        )
        self.btn_toggle_cam.pack(fill=tk.X, pady=(0, 12))

        # Device Selection
        dev_frame = tk.Frame(right_panel, bg=self.bg_card)
        dev_frame.pack(fill=tk.X, pady=(0, 12))
        tk.Label(dev_frame, text="Video Device:", font=("Segoe UI", 9),
                 bg=self.bg_card, fg=self.text_muted).pack(anchor="w", pady=(0, 3))
        
        self.cam_combo = ttk.Combobox(dev_frame, values=["Device 0 (Default Webcam)", "Device 1", "Device 2"], state="readonly")
        self.cam_combo.current(0)
        self.cam_combo.pack(fill=tk.X)

        # Auto-Log Attendance Checkbox
        self.auto_log_var = tk.BooleanVar(value=True)
        chk = tk.Checkbutton(
            right_panel,
            text="Auto-Log to Attendance CSV",
            variable=self.auto_log_var,
            font=("Segoe UI", 9),
            bg=self.bg_card,
            fg=self.text_main,
            selectcolor=self.bg_card,
            activebackground=self.bg_card,
            activeforeground=self.text_main
        )
        chk.pack(anchor="w", pady=(0, 14))

        # Live Feed Metrics Card
        stat_card = tk.LabelFrame(right_panel, text="Live Telemetry", font=("Segoe UI", 9, "bold"),
                                  bg=self.bg_accent, fg=self.cyan, padx=12, pady=10, relief=tk.FLAT)
        stat_card.pack(fill=tk.X, pady=(0, 16))

        self.lbl_cam_fps = tk.Label(stat_card, text="Speed: 0 FPS", font=("Segoe UI", 9),
                                    bg=self.bg_accent, fg=self.text_main)
        self.lbl_cam_fps.pack(anchor="w")

        self.lbl_cam_faces = tk.Label(stat_card, text="Detected Faces: 0", font=("Segoe UI", 9),
                                      bg=self.bg_accent, fg=self.text_main)
        self.lbl_cam_faces.pack(anchor="w", pady=3)

        self.lbl_cam_last = tk.Label(stat_card, text="Last Verified: None", font=("Segoe UI", 9),
                                     bg=self.bg_accent, fg=self.success)
        self.lbl_cam_last.pack(anchor="w")

        # Quick Enroll from Camera Feed
        tk.Label(right_panel, text="Instant Enrollment", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(anchor="w", pady=(10, 4))
        tk.Label(right_panel, text="Register the person currently standing in front of the camera:",
                 font=("Segoe UI", 8), bg=self.bg_card, fg=self.text_muted, wraplength=260, justify="left").pack(anchor="w", pady=(0, 8))

        btn_quick = tk.Button(
            right_panel,
            text="⚡  Enroll Current Face",
            font=("Segoe UI", 9, "bold"),
            bg=self.primary,
            fg="#ffffff",
            activebackground=self.primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=7,
            command=self.open_quick_enroll_modal
        )
        btn_quick.pack(fill=tk.X)

    def show_camera_page(self):
        self._set_active_nav("camera", "Live Camera Recognition")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        idx = self.cam_combo.current()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Unable to open camera device index {idx}.\nPlease verify your webcam connection.")
            return

        self.is_camera_running = True
        self.btn_toggle_cam.config(text="⏹  Stop Camera", bg=self.danger, activebackground=self.danger_hover)
        self.status_var.set(f"Camera active on device {idx}. Live detection running.")
        self.last_fps_time = time.time()
        self._camera_update_loop()

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_toggle_cam.config(text="▶  Start Camera", bg=self.success, activebackground=self.success_hover)
        self.status_var.set("Camera stopped.")
        self.lbl_cam_fps.config(text="Speed: 0 FPS")
        self.lbl_cam_faces.config(text="Detected Faces: 0")
        self.badge_fps.config(text="FPS: 0")

    def _camera_update_loop(self):
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            # Mirror horizontally for natural view
            frame = cv2.flip(frame, 1)

            # Measure real FPS
            now = time.time()
            fps_val = 1.0 / max(1e-4, now - self.last_fps_time)
            self.last_fps_time = now
            self.fps = 0.9 * self.fps + 0.1 * fps_val if self.fps > 0 else fps_val

            # Run Face Recognition Pipeline
            known_embeddings = self.database.embeddings
            results = self.engine.process_frame(frame, known_embeddings, draw_annotations=True)

            fps_int = int(self.fps)
            self.lbl_cam_fps.config(text=f"Speed: {fps_int} FPS")
            self.lbl_cam_faces.config(text=f"Detected Faces: {len(results)}")
            self.badge_fps.config(text=f"FPS: {fps_int}")

            # Check for recognized faces and log attendance
            for r in results:
                if r["is_known"]:
                    name = r["name"]
                    conf = r["confidence_pct"]
                    self.lbl_cam_last.config(text=f"Last: {name} ({conf:.0f}%)")
                    if self.auto_log_var.get():
                        self.attendance.log(name, conf)

            # Handle live enrollment capture if active
            if self.is_enrolling:
                self._process_enrollment_frame(frame, results)

            # Draw FPS badge on frame
            cv2.putText(frame, f"{fps_int} FPS", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            self.current_frame = frame
            self._render_frame(self.video_canvas, frame)

        # Target ~30 FPS UI refresh
        self.root.after(20, self._camera_update_loop)

    def _render_frame(self, canvas: tk.Canvas, bgr_frame: np.ndarray):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 640, 480

        h, w = bgr_frame.shape[:2]
        ratio = min(cw / w, ch / h)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))

        resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))

        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=img_tk)
        canvas.image = img_tk

    # =========================================================================
    # PAGE 2: ENROLL PERSON
    # =========================================================================
    def _init_enroll_page(self):
        page = tk.Frame(self.content_area, bg=self.bg_root)
        self.pages["enroll"] = page

        # Centered Card Container
        card = tk.Frame(page, bg=self.bg_card, padx=32, pady=28,
                        highlightbackground=self.border_color, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True)

        tk.Label(card, text="Register New Identity", font=("Segoe UI", 15, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(anchor="w")
        tk.Label(card, text="Provide the person's name and choose between live camera sampling or photo uploads.",
                 font=("Segoe UI", 9), bg=self.bg_card, fg=self.text_muted).pack(anchor="w", pady=(2, 20))

        # Input Field: Name
        name_box = tk.Frame(card, bg=self.bg_card)
        name_box.pack(fill=tk.X, pady=(0, 24))
        tk.Label(name_box, text="Full Name / Identifier:", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_main).pack(anchor="w", pady=(0, 6))

        self.enroll_name_entry = tk.Entry(name_box, font=("Segoe UI", 11), bg="#111420", fg="#ffffff",
                                          insertbackground="#ffffff", relief=tk.FLAT, bd=8)
        self.enroll_name_entry.pack(fill=tk.X, ipady=4)

        # Split Card for 2 Options
        methods_frame = tk.Frame(card, bg=self.bg_card)
        methods_frame.pack(fill=tk.BOTH, expand=True)

        # Option A: Live Camera Sampling
        opt_cam = tk.LabelFrame(methods_frame, text="Method A: Live Webcam Sampling",
                                font=("Segoe UI", 10, "bold"), bg=self.bg_accent, fg=self.cyan,
                                padx=20, pady=18, relief=tk.FLAT)
        opt_cam.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        tk.Label(opt_cam, text="Captures 5 consecutive frames from your live webcam. A multi-shot mean embedding will be computed for maximum accuracy across head angles.",
                 font=("Segoe UI", 9), bg=self.bg_accent, fg=self.text_muted, wraplength=340, justify="left").pack(anchor="w", pady=(0, 16))

        self.btn_start_sampling = tk.Button(
            opt_cam,
            text="📸  Start Live Sampling (5 Shots)",
            font=("Segoe UI", 10, "bold"),
            bg=self.primary,
            fg="#ffffff",
            activebackground=self.primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=8,
            command=self.start_live_enrollment
        )
        self.btn_start_sampling.pack(fill=tk.X)

        self.enroll_feedback_lbl = tk.Label(opt_cam, text="", font=("Segoe UI", 9, "bold"),
                                            bg=self.bg_accent, fg=self.warning)
        self.enroll_feedback_lbl.pack(pady=12)

        # Option B: Upload Images
        opt_file = tk.LabelFrame(methods_frame, text="Method B: Select Existing Photos",
                                 font=("Segoe UI", 10, "bold"), bg=self.bg_accent, fg=self.cyan,
                                 padx=20, pady=18, relief=tk.FLAT)
        opt_file.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        tk.Label(opt_file, text="Choose one or multiple clear portrait photos of this person from your computer (.jpg, .png).",
                 font=("Segoe UI", 9), bg=self.bg_accent, fg=self.text_muted, wraplength=340, justify="left").pack(anchor="w", pady=(0, 16))

        btn_pick_files = tk.Button(
            opt_file,
            text="📁  Browse Photos from Disk...",
            font=("Segoe UI", 10, "bold"),
            bg="#334155",
            fg="#ffffff",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=8,
            command=self.enroll_from_files
        )
        btn_pick_files.pack(fill=tk.X)

    def show_enroll_page(self):
        self._set_active_nav("enroll", "Enroll New Identity")

    def start_live_enrollment(self):
        name = self.enroll_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Name Required", "Please enter a valid person name first.")
            return

        if not self.is_camera_running:
            self.start_camera()
            self.show_camera_page()

        self.is_enrolling = True
        self.enroll_target_name = name
        self.enroll_samples = []
        self.enroll_countdown = 15
        self.enroll_feedback_lbl.config(text=f"Preparing camera... Look straight at the lens.")
        self.status_var.set(f"Sampling face for '{name}'... Stand still and slowly tilt head slightly.")

    def _process_enrollment_frame(self, frame: np.ndarray, results: list):
        if self.enroll_countdown > 0:
            self.enroll_countdown -= 1
            cv2.putText(frame, f"GET READY: {self.enroll_countdown // 3 + 1}", (frame.shape[1]//2 - 110, frame.shape[0]//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
            return

        # Ensure only 1 face is present for high fidelity
        if len(results) == 1:
            self.enroll_samples.append(frame.copy())
            count = len(self.enroll_samples)
            self.enroll_feedback_lbl.config(text=f"Captured {count}/{self.enroll_target_count} samples...")
            self.status_var.set(f"Capturing samples: {count}/{self.enroll_target_count}")

            cv2.putText(frame, f"CAPTURED {count}/{self.enroll_target_count}", (20, 80),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            if count >= self.enroll_target_count:
                self.is_enrolling = False
                success, msg = self.database.add_person_from_frames(self.enroll_target_name, self.enroll_samples)
                self.enroll_feedback_lbl.config(text=msg)
                self.status_var.set(msg)
                self._refresh_db_stats()
                self._refresh_database_table()
                messagebox.showinfo("Enrollment Complete", msg)
        else:
            cv2.putText(frame, "Ensure only 1 face is visible in front of camera", (25, 75),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

    def enroll_from_files(self):
        name = self.enroll_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Name Required", "Please enter a valid person name first.")
            return

        files = filedialog.askopenfilenames(
            title=f"Select Photos for {name}",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All Files", "*.*")]
        )
        if not files:
            return

        self.status_var.set(f"Extracting features from {len(files)} image(s)...")
        self.root.update()

        success, msg = self.database.add_person_from_images(name, list(files))
        if success:
            self._refresh_db_stats()
            self._refresh_database_table()
            messagebox.showinfo("Success", msg)
            self.status_var.set(msg)
        else:
            messagebox.showerror("Enrollment Error", msg)
            self.status_var.set(msg)

    def open_quick_enroll_modal(self):
        modal = tk.Toplevel(self.root)
        modal.title("Quick Enroll")
        modal.geometry("380x180")
        modal.configure(bg=self.bg_card)
        modal.transient(self.root)
        modal.grab_set()

        tk.Label(modal, text="Enter Person's Name:", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(pady=(22, 6))

        entry = tk.Entry(modal, font=("Segoe UI", 11), bg="#111420", fg="#ffffff",
                         insertbackground="#ffffff", relief=tk.FLAT, bd=6)
        entry.pack(fill=tk.X, padx=30, ipady=3)
        entry.focus_set()

        def confirm():
            val = entry.get().strip()
            if val:
                self.enroll_name_entry.delete(0, tk.END)
                self.enroll_name_entry.insert(0, val)
                modal.destroy()
                self.start_live_enrollment()

        tk.Button(modal, text="Confirm & Capture", font=("Segoe UI", 10, "bold"),
                  bg=self.primary, fg="#ffffff", relief=tk.FLAT, pady=6, cursor="hand2",
                  command=confirm).pack(pady=16)

    # =========================================================================
    # PAGE 3: INSPECT IMAGE
    # =========================================================================
    def _init_inspect_page(self):
        page = tk.Frame(self.content_area, bg=self.bg_root)
        self.pages["inspect"] = page

        left_box = tk.Frame(page, bg=self.bg_card, highlightbackground=self.border_color, highlightthickness=1)
        left_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))

        self.inspect_canvas = tk.Canvas(left_box, bg="#08090f", highlightthickness=0)
        self.inspect_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Right Inspector Sidebar
        sidebar = tk.Frame(page, bg=self.bg_card, width=300, padx=16, pady=16)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Image Analysis", font=("Segoe UI", 11, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(anchor="w", pady=(0, 10))

        btn_browse = tk.Button(
            sidebar,
            text="📁  Browse Image...",
            font=("Segoe UI", 10, "bold"),
            bg=self.primary,
            fg="#ffffff",
            activebackground=self.primary_hover,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            pady=8,
            command=self.inspect_image_file
        )
        btn_browse.pack(fill=tk.X, pady=(0, 8))

        self.btn_export_annotated = tk.Button(
            sidebar,
            text="💾  Export Annotated Image",
            font=("Segoe UI", 9),
            bg="#334155",
            fg="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED,
            pady=6,
            command=self.export_annotated_image
        )
        self.btn_export_annotated.pack(fill=tk.X, pady=(0, 16))

        tk.Label(sidebar, text="Detections & Matches:", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(anchor="w", pady=(0, 6))

        self.txt_inspect_report = tk.Text(sidebar, bg=self.bg_accent, fg=self.text_main,
                                          font=("Segoe UI", 9), wrap=tk.WORD, height=16,
                                          relief=tk.FLAT, padx=10, pady=10)
        self.txt_inspect_report.pack(fill=tk.BOTH, expand=True)

        self.last_inspected_image = None

    def show_inspect_page(self):
        self._set_active_nav("inspect", "Inspect Image File")

    def inspect_image_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image to Analyze",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return

        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not load selected image.")
            return

        display_img = img.copy()
        results = self.engine.process_frame(display_img, self.database.embeddings, draw_annotations=True)

        self.last_inspected_image = display_img
        self.btn_export_annotated.config(state=tk.NORMAL)

        # Render preview
        self._render_frame(self.inspect_canvas, display_img)

        # Update text breakdown
        self.txt_inspect_report.delete("1.0", tk.END)
        self.txt_inspect_report.insert(tk.END, f"File: {Path(file_path).name}\n")
        self.txt_inspect_report.insert(tk.END, f"Total Faces Found: {len(results)}\n\n")

        if results:
            for i, r in enumerate(results, start=1):
                name = r["name"]
                conf = r["confidence_pct"]
                status = "VERIFIED" if r["is_known"] else "UNKNOWN"
                self.txt_inspect_report.insert(tk.END, f"Face #{i}:\n")
                self.txt_inspect_report.insert(tk.END, f" • Identity: {name}\n")
                self.txt_inspect_report.insert(tk.END, f" • Confidence: {conf:.1f}%\n")
                self.txt_inspect_report.insert(tk.END, f" • Status: {status}\n\n")
        else:
            self.txt_inspect_report.insert(tk.END, "No faces detected in this image.")

    def export_annotated_image(self):
        if self.last_inspected_image is None:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("PNG Image", "*.png")]
        )
        if dest:
            _, buf = cv2.imencode(Path(dest).suffix or ".jpg", self.last_inspected_image)
            buf.tofile(dest)
            messagebox.showinfo("Export Successful", f"Saved annotated image to:\n{dest}")

    # =========================================================================
    # PAGE 4: DATABASE MANAGER
    # =========================================================================
    def _init_database_page(self):
        page = tk.Frame(self.content_area, bg=self.bg_root)
        self.pages["database"] = page

        # Action Bar
        top_bar = tk.Frame(page, bg=self.bg_root)
        top_bar.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_bar, text="Registered Identities", font=("Segoe UI", 12, "bold"),
                 bg=self.bg_root, fg=self.text_title).pack(side=tk.LEFT)

        btn_rebuild = tk.Button(top_bar, text="🔄  Re-index Database", font=("Segoe UI", 9),
                                bg="#334155", fg="#ffffff", relief=tk.FLAT, cursor="hand2",
                                command=self.rebuild_database)
        btn_rebuild.pack(side=tk.RIGHT, padx=5)

        btn_open_folder = tk.Button(top_bar, text="📂  Open Faces Folder", font=("Segoe UI", 9),
                                    bg="#334155", fg="#ffffff", relief=tk.FLAT, cursor="hand2",
                                    command=self.open_faces_folder)
        btn_open_folder.pack(side=tk.RIGHT, padx=5)

        btn_delete = tk.Button(top_bar, text="🗑️  Delete Selected", font=("Segoe UI", 9, "bold"),
                               bg=self.danger, fg="#ffffff", relief=tk.FLAT, cursor="hand2",
                               command=self.delete_selected_person)
        btn_delete.pack(side=tk.RIGHT, padx=5)

        # Table
        cols = ("name", "photos", "status", "folder")
        self.tree_db = ttk.Treeview(page, columns=cols, show="headings", selectmode="browse")
        self.tree_db.heading("name", text="Person Name")
        self.tree_db.heading("photos", text="Photos Count")
        self.tree_db.heading("status", text="Embedding Cache")
        self.tree_db.heading("folder", text="Directory Path")

        self.tree_db.column("name", width=200)
        self.tree_db.column("photos", width=120, anchor="center")
        self.tree_db.column("status", width=160, anchor="center")
        self.tree_db.column("folder", width=420)

        sb = ttk.Scrollbar(page, orient=tk.VERTICAL, command=self.tree_db.yview)
        self.tree_db.configure(yscroll=sb.set)

        self.tree_db.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_database_table()

    def show_database_page(self):
        self._set_active_nav("database", "Identities Database")
        self._refresh_database_table()

    def _refresh_database_table(self):
        for item in self.tree_db.get_children():
            self.tree_db.delete(item)

        people = self.database.list_people()
        for p in people:
            status = "Active (Computed)" if p["is_active"] else "Missing"
            self.tree_db.insert("", tk.END, values=(p["name"], p["photo_count"], status, p["folder"]))

    def _refresh_db_stats(self):
        count = len(self.database.embeddings)
        self.badge_registered.config(text=f"Identities: {count}")

    def rebuild_database(self):
        self.status_var.set("Re-indexing database embeddings...")
        self.root.update()
        self.database.rebuild_all()
        self._refresh_db_stats()
        self._refresh_database_table()
        self.status_var.set("Database re-indexed successfully.")
        messagebox.showinfo("Done", "All face embeddings recomputed successfully.")

    def open_faces_folder(self):
        folder = str(self.database.faces_dir)
        if os.name == 'nt':
            os.startfile(folder)
        else:
            subprocess.run(["xdg-open", folder])

    def delete_selected_person(self):
        sel = self.tree_db.selection()
        if not sel:
            messagebox.showwarning("Selection Required", "Please select an identity from the table.")
            return

        item = self.tree_db.item(sel[0])
        name = item["values"][0]
        if messagebox.askyesno("Confirm Deletion", f"Permanently delete '{name}' and all associated photos?"):
            self.database.delete_person(name)
            self._refresh_db_stats()
            self._refresh_database_table()
            messagebox.showinfo("Deleted", f"'{name}' removed from database.")

    # =========================================================================
    # PAGE 5: ATTENDANCE LOGS
    # =========================================================================
    def _init_attendance_page(self):
        page = tk.Frame(self.content_area, bg=self.bg_root)
        self.pages["attendance"] = page

        top_bar = tk.Frame(page, bg=self.bg_root)
        top_bar.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_bar, text="Verified Attendance Logs", font=("Segoe UI", 12, "bold"),
                 bg=self.bg_root, fg=self.text_title).pack(side=tk.LEFT)

        btn_refresh = tk.Button(top_bar, text="🔄  Refresh Logs", font=("Segoe UI", 9),
                                bg="#334155", fg="#ffffff", relief=tk.FLAT, cursor="hand2",
                                command=self._refresh_attendance_table)
        btn_refresh.pack(side=tk.RIGHT, padx=5)

        btn_open_csv = tk.Button(top_bar, text="📊  Open in Excel / CSV", font=("Segoe UI", 9, "bold"),
                                 bg=self.success, fg="#ffffff", relief=tk.FLAT, cursor="hand2",
                                 command=self.open_attendance_csv)
        btn_open_csv.pack(side=tk.RIGHT, padx=5)

        cols = ("date", "time", "name", "confidence", "status")
        self.tree_attendance = ttk.Treeview(page, columns=cols, show="headings")
        self.tree_attendance.heading("date", text="Date")
        self.tree_attendance.heading("time", text="Time")
        self.tree_attendance.heading("name", text="Person Name")
        self.tree_attendance.heading("confidence", text="Confidence (%)")
        self.tree_attendance.heading("status", text="Verification Status")

        self.tree_attendance.column("date", width=120, anchor="center")
        self.tree_attendance.column("time", width=120, anchor="center")
        self.tree_attendance.column("name", width=220)
        self.tree_attendance.column("confidence", width=140, anchor="center")
        self.tree_attendance.column("status", width=140, anchor="center")

        sb = ttk.Scrollbar(page, orient=tk.VERTICAL, command=self.tree_attendance.yview)
        self.tree_attendance.configure(yscroll=sb.set)

        self.tree_attendance.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_attendance_table()

    def show_attendance_page(self):
        self._set_active_nav("attendance", "Attendance Logs")
        self._refresh_attendance_table()

    def _refresh_attendance_table(self):
        for item in self.tree_attendance.get_children():
            self.tree_attendance.delete(item)

        records = self.attendance.get_recent_records(limit=100)
        for r in records:
            if len(r) >= 5:
                self.tree_attendance.insert("", tk.END, values=(r[0], r[1], r[2], r[3], r[4]))

    def open_attendance_csv(self):
        f = str(self.attendance.log_file)
        if os.name == 'nt':
            os.startfile(f)
        else:
            subprocess.run(["xdg-open", f])

    # =========================================================================
    # PAGE 6: TUNING & SETTINGS
    # =========================================================================
    def _init_settings_page(self):
        page = tk.Frame(self.content_area, bg=self.bg_root)
        self.pages["settings"] = page

        card = tk.Frame(page, bg=self.bg_card, padx=32, pady=28,
                        highlightbackground=self.border_color, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True)

        tk.Label(card, text="Threshold & Algorithm Tuning", font=("Segoe UI", 15, "bold"),
                 bg=self.bg_card, fg=self.text_title).pack(anchor="w")
        tk.Label(card, text="Fine-tune deep learning confidence levels for detection and identity matching in real-time.",
                 font=("Segoe UI", 9), bg=self.bg_card, fg=self.text_muted).pack(anchor="w", pady=(2, 24))

        # Slider 1: Match Threshold (Cosine similarity)
        tk.Label(card, text="Match Similarity Threshold (Cosine):", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_main).pack(anchor="w")
        tk.Label(card, text="Default: 0.38 (Higher = stricter recognition, Lower = more lenient)",
                 font=("Segoe UI", 8), bg=self.bg_card, fg=self.text_dim).pack(anchor="w", pady=(0, 6))

        self.cosine_val_lbl = tk.Label(card, text=f"{self.engine.cosine_thresh:.2f}", font=("Segoe UI", 10, "bold"),
                                       bg=self.bg_card, fg=self.cyan)
        self.cosine_slider = ttk.Scale(card, from_=0.20, to=0.70, value=self.engine.cosine_thresh,
                                       command=self._on_cosine_slider_change)
        self.cosine_slider.pack(fill=tk.X, pady=(0, 4))
        self.cosine_val_lbl.pack(anchor="e", pady=(0, 20))

        # Slider 2: Detection Confidence
        tk.Label(card, text="Face Detection Confidence (YuNet):", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_main).pack(anchor="w")
        tk.Label(card, text="Default: 0.75 (Higher = eliminates false positive face boxes)",
                 font=("Segoe UI", 8), bg=self.bg_card, fg=self.text_dim).pack(anchor="w", pady=(0, 6))

        self.det_val_lbl = tk.Label(card, text=f"{self.engine.score_thresh:.2f}", font=("Segoe UI", 10, "bold"),
                                    bg=self.bg_card, fg=self.cyan)
        self.det_slider = ttk.Scale(card, from_=0.40, to=0.95, value=self.engine.score_thresh,
                                    command=self._on_detector_slider_change)
        self.det_slider.pack(fill=tk.X, pady=(0, 4))
        self.det_val_lbl.pack(anchor="e", pady=(0, 20))

        # Cooldown setting
        tk.Label(card, text="Attendance Log Cooldown (Seconds):", font=("Segoe UI", 10, "bold"),
                 bg=self.bg_card, fg=self.text_main).pack(anchor="w")
        tk.Label(card, text="Prevents logging the same verified person more than once per time window.",
                 font=("Segoe UI", 8), bg=self.bg_card, fg=self.text_dim).pack(anchor="w", pady=(0, 6))

        self.cooldown_val_lbl = tk.Label(card, text=f"{self.attendance.cooldown_seconds}s", font=("Segoe UI", 10, "bold"),
                                         bg=self.bg_card, fg=self.cyan)
        self.cooldown_slider = ttk.Scale(card, from_=10, to=300, value=self.attendance.cooldown_seconds,
                                         command=self._on_cooldown_slider_change)
        self.cooldown_slider.pack(fill=tk.X, pady=(0, 4))
        self.cooldown_val_lbl.pack(anchor="e", pady=(0, 20))

    def show_settings_page(self):
        self._set_active_nav("settings", "Algorithm Tuning")

    def _on_cosine_slider_change(self, val):
        f = float(val)
        self.engine.set_cosine_threshold(f)
        self.cosine_val_lbl.config(text=f"{f:.2f}")

    def _on_detector_slider_change(self, val):
        f = float(val)
        self.engine.set_detector_threshold(f)
        self.det_val_lbl.config(text=f"{f:.2f}")

    def _on_cooldown_slider_change(self, val):
        i = int(float(val))
        self.attendance.cooldown_seconds = i
        self.cooldown_val_lbl.config(text=f"{i}s")

    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    def on_close(self):
        self.stop_camera()
        self.root.destroy()

def start_gui():
    root = tk.Tk()
    app = ModernFaceApp(root)
    root.mainloop()

if __name__ == "__main__":
    start_gui()
