"""
FaceEngine AI — Simple, friendly Desktop GUI
Clean layout that anyone can use without technical knowledge.
"""

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


# ─────────────────────────────────────────────
#  COLOR THEME
# ─────────────────────────────────────────────
BG          = "#1a1a2e"   # dark navy background
PANEL       = "#16213e"   # sidebar / panel
CARD        = "#0f3460"   # card surface
BTN_BLUE    = "#4361ee"   # primary blue button
BTN_GREEN   = "#2dc653"   # start / confirm green
BTN_RED     = "#e63946"   # stop / delete red
BTN_GRAY    = "#3a3a5c"   # secondary gray button
ACCENT      = "#e94560"   # accent / highlight
WHITE       = "#ffffff"
TEXT        = "#e0e0e0"
MUTED       = "#8888aa"
SUCCESS_CLR = "#2dc653"
WARN_CLR    = "#f4a261"


def _btn(parent, text, color, command, width=None, pady=8):
    """Helper — creates a flat, rounded-looking button."""
    kw = dict(
        text=text,
        bg=color,
        fg=WHITE,
        font=("Segoe UI", 10, "bold"),
        relief=tk.FLAT,
        cursor="hand2",
        activebackground=color,
        activeforeground=WHITE,
        pady=pady,
        command=command,
    )
    if width:
        kw["width"] = width
    b = tk.Button(parent, **kw)
    b.bind("<Enter>", lambda e: b.config(bg=_darken(color)))
    b.bind("<Leave>", lambda e: b.config(bg=color))
    return b


def _darken(hex_color: str) -> str:
    """Return a slightly darker shade of hex_color."""
    c = hex_color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    r, g, b = max(0, r - 25), max(0, g - 25), max(0, b - 25)
    return f"#{r:02x}{g:02x}{b:02x}"


def _label(parent, text, size=10, bold=False, color=TEXT, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, font=("Segoe UI", size, weight),
                    bg=parent["bg"], fg=color, **kw)


def _sep(parent):
    tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, pady=10)


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────
class FaceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("FaceEngine AI  •  Face Recognition")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        # ── Core AI Engines ──────────────────────────────────────────
        self.engine     = FaceEngine(detector_score_threshold=0.75, cosine_threshold=0.38)
        self.database   = FaceDatabase(engine=self.engine)
        self.attendance = AttendanceLogger()

        # ── Camera state ─────────────────────────────────────────────
        self.cap               = None
        self.is_camera_running = False
        self.camera_index      = 0
        self.current_frame     = None
        self.last_fps_time     = time.time()
        self.fps               = 0.0

        # ── Enrollment state ─────────────────────────────────────────
        self.is_enrolling       = False
        self.enroll_name        = ""
        self.enroll_samples     = []
        self.enroll_need        = 5
        self.enroll_countdown   = 0

        # ── Status bar text ──────────────────────────────────────────
        self.status_var = tk.StringVar(value="✅  Ready — 100 % Offline. No internet or API keys needed.")

        self._build_ui()
        self._update_identity_count()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Left sidebar ─────────────────────────────────────────────
        sidebar = tk.Frame(self.root, bg=PANEL, width=200)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Logo
        logo_frame = tk.Frame(sidebar, bg=PANEL, pady=20)
        logo_frame.pack(fill=tk.X)
        tk.Label(logo_frame, text="🎭", font=("Segoe UI", 28), bg=PANEL, fg=ACCENT).pack()
        tk.Label(logo_frame, text="FaceEngine AI", font=("Segoe UI", 12, "bold"),
                 bg=PANEL, fg=WHITE).pack()
        tk.Label(logo_frame, text="100 % Offline  •  Free", font=("Segoe UI", 8),
                 bg=PANEL, fg=MUTED).pack()

        tk.Frame(sidebar, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=16, pady=8)

        # Navigation buttons
        self._nav_buttons = {}
        nav_items = [
            ("camera",     "📹   Live Camera",      self._show_camera),
            ("enroll",     "➕   Add New Person",   self._show_enroll),
            ("inspect",    "🔍   Analyze a Photo",  self._show_inspect),
            ("people",     "👥   People List",      self._show_people),
            ("attendance", "📋   Attendance Log",   self._show_attendance),
            ("settings",   "⚙️   Settings",         self._show_settings),
        ]
        nav_box = tk.Frame(sidebar, bg=PANEL)
        nav_box.pack(fill=tk.X, padx=10)

        for key, label, cmd in nav_items:
            b = tk.Button(
                nav_box, text=label,
                font=("Segoe UI", 10), anchor="w", padx=14, pady=9,
                bg=PANEL, fg=TEXT,
                activebackground=BTN_BLUE, activeforeground=WHITE,
                relief=tk.FLAT, cursor="hand2",
                command=lambda c=cmd, k=key: self._nav_click(k, c),
            )
            b.pack(fill=tk.X, pady=2)
            self._nav_buttons[key] = b

        # Sidebar status pill
        tk.Frame(sidebar, bg="#2a2a4a", height=1).pack(fill=tk.X, padx=16, pady=(0, 8))
        self._id_count_lbl = tk.Label(sidebar, text="People registered: 0",
                                      font=("Segoe UI", 9), bg=PANEL, fg=MUTED)
        self._id_count_lbl.pack(side=tk.BOTTOM, pady=(0, 20))

        # ── Right content area ────────────────────────────────────────
        right = tk.Frame(self.root, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Page title bar
        self._title_bar = tk.Frame(right, bg=CARD, height=52)
        self._title_bar.pack(fill=tk.X)
        self._title_bar.pack_propagate(False)

        self._page_title = tk.Label(self._title_bar, text="",
                                    font=("Segoe UI", 13, "bold"),
                                    bg=CARD, fg=WHITE)
        self._page_title.pack(side=tk.LEFT, padx=20, pady=12)

        self._fps_badge = tk.Label(self._title_bar, text="",
                                   font=("Segoe UI", 9, "bold"),
                                   bg=CARD, fg=SUCCESS_CLR)
        self._fps_badge.pack(side=tk.RIGHT, padx=16)

        # Content switcher
        self._content = tk.Frame(right, bg=BG)
        self._content.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

        # Global status bar
        bar = tk.Frame(right, bg=PANEL, height=26)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        bar.pack_propagate(False)
        tk.Label(bar, textvariable=self.status_var,
                 font=("Segoe UI", 8), bg=PANEL, fg=MUTED).pack(side=tk.LEFT, padx=12)

        # Build all pages
        self._pages = {}
        self._build_camera_page()
        self._build_enroll_page()
        self._build_inspect_page()
        self._build_people_page()
        self._build_attendance_page()
        self._build_settings_page()

        self._active_page = None
        self._show_camera()

    # ── Navigation helper ────────────────────────────────────────────
    def _nav_click(self, key: str, cmd):
        for k, b in self._nav_buttons.items():
            b.config(bg=BTN_BLUE if k == key else PANEL,
                     fg=WHITE if k == key else TEXT)
        cmd()

    def _switch_page(self, key: str, title: str):
        if self._active_page:
            self._active_page.pack_forget()
        page = self._pages[key]
        page.pack(fill=tk.BOTH, expand=True)
        self._active_page = page
        self._page_title.config(text=title)

    # ══════════════════════════════════════════════════════════════════
    #  PAGE 1 — LIVE CAMERA
    # ══════════════════════════════════════════════════════════════════
    def _build_camera_page(self):
        page = tk.Frame(self._content, bg=BG)
        self._pages["camera"] = page

        # ── Video canvas (left, large) ────────────────────────────────
        cam_frame = tk.Frame(page, bg="#000000",
                             highlightbackground="#2a2a4a", highlightthickness=1)
        cam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self._video_canvas = tk.Canvas(cam_frame, bg="#0a0a14", highlightthickness=0)
        self._video_canvas.pack(fill=tk.BOTH, expand=True)

        # Placeholder text on canvas
        self._video_canvas.create_text(
            300, 240,
            text="📷  Camera is off\n\nPress  ▶ Start Camera  to begin",
            font=("Segoe UI", 14), fill=MUTED, justify=tk.CENTER,
            tags="placeholder"
        )

        # ── Right control panel ───────────────────────────────────────
        ctrl = tk.Frame(page, bg=PANEL, width=230, padx=16, pady=16)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl.pack_propagate(False)

        _label(ctrl, "Camera Controls", 11, bold=True, color=WHITE).pack(anchor="w", pady=(0, 12))

        # Big start / stop button
        self._cam_btn = _btn(ctrl, "▶  Start Camera", BTN_GREEN, self._toggle_camera, pady=12)
        self._cam_btn.pack(fill=tk.X, pady=(0, 10))

        # Camera selector
        _label(ctrl, "Select Camera:", 9, color=MUTED).pack(anchor="w")
        self._cam_combo = ttk.Combobox(ctrl,
            values=["Camera 0  (default)", "Camera 1", "Camera 2"],
            state="readonly", font=("Segoe UI", 9))
        self._cam_combo.current(0)
        self._cam_combo.pack(fill=tk.X, pady=(2, 12))

        # Auto-log checkbox
        self._auto_log = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Auto-save attendance log",
                       variable=self._auto_log,
                       font=("Segoe UI", 9), bg=PANEL, fg=TEXT,
                       selectcolor=PANEL,
                       activebackground=PANEL).pack(anchor="w", pady=(0, 14))

        _sep(ctrl)

        # Live stats
        _label(ctrl, "Live Stats", 10, bold=True, color=WHITE).pack(anchor="w", pady=(0, 8))
        self._lbl_fps   = _label(ctrl, "Speed:  —", 9, color=TEXT)
        self._lbl_faces = _label(ctrl, "Faces detected:  0", 9, color=TEXT)
        self._lbl_last  = _label(ctrl, "Last recognized:  —", 9, color=SUCCESS_CLR)
        self._lbl_fps.pack(anchor="w")
        self._lbl_faces.pack(anchor="w", pady=4)
        self._lbl_last.pack(anchor="w")

        _sep(ctrl)

        # Quick enroll from camera
        _label(ctrl, "Enroll while camera is on", 10, bold=True, color=WHITE).pack(anchor="w", pady=(0, 6))
        _label(ctrl, "Stand in front of the camera,\nthen press the button below.",
               9, color=MUTED, justify="left", wraplength=190).pack(anchor="w", pady=(0, 8))
        _btn(ctrl, "⚡  Enroll Current Face", BTN_BLUE, self._quick_enroll_modal).pack(fill=tk.X)

    def _show_camera(self):
        self._switch_page("camera", "📹   Live Camera  —  Real-time Face Recognition")

    def _toggle_camera(self):
        if self.is_camera_running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        idx = self._cam_combo.current()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Not Found",
                f"Could not open camera #{idx}.\n\n"
                "Make sure your webcam is connected and not used by another app.")
            return

        # Remove placeholder
        self._video_canvas.delete("placeholder")

        self.is_camera_running = True
        self._cam_btn.config(text="⏹  Stop Camera", bg=BTN_RED,
                              activebackground=BTN_RED)
        self.last_fps_time = time.time()
        self.status_var.set(f"✅  Camera #{idx} is running — recognizing faces in real time.")
        self._camera_loop()

    def _stop_camera(self):
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._cam_btn.config(text="▶  Start Camera", bg=BTN_GREEN,
                              activebackground=BTN_GREEN)
        self._lbl_fps.config(text="Speed:  —")
        self._lbl_faces.config(text="Faces detected:  0")
        self._fps_badge.config(text="")
        self.status_var.set("Camera stopped.")

        # Restore placeholder
        w = self._video_canvas.winfo_width() or 600
        h = self._video_canvas.winfo_height() or 400
        self._video_canvas.delete("all")
        self._video_canvas.create_text(
            w // 2, h // 2,
            text="📷  Camera is off\n\nPress  ▶ Start Camera  to begin",
            font=("Segoe UI", 14), fill=MUTED, justify=tk.CENTER,
            tags="placeholder"
        )

    def _camera_loop(self):
        if not self.is_camera_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            frame = cv2.flip(frame, 1)

            now = time.time()
            fps_raw = 1.0 / max(1e-4, now - self.last_fps_time)
            self.last_fps_time = now
            self.fps = 0.9 * self.fps + 0.1 * fps_raw if self.fps > 0 else fps_raw

            results = self.engine.process_frame(frame, self.database.embeddings, draw_annotations=True)

            fps_int = int(self.fps)
            self._lbl_fps.config(text=f"Speed:  {fps_int} FPS")
            self._lbl_faces.config(text=f"Faces detected:  {len(results)}")
            self._fps_badge.config(text=f"⚡ {fps_int} FPS")

            for r in results:
                if r["is_known"]:
                    name = r["name"]
                    conf = r["confidence_pct"]
                    self._lbl_last.config(text=f"Last recognized:  {name}  ({conf:.0f}%)")
                    if self._auto_log.get():
                        self.attendance.log(name, conf)

            if self.is_enrolling:
                self._run_enrollment_frame(frame, results)

            cv2.putText(frame, f"{fps_int} FPS", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)

            self.current_frame = frame
            self._draw_frame(self._video_canvas, frame)

        self.root.after(20, self._camera_loop)

    def _draw_frame(self, canvas: tk.Canvas, bgr: np.ndarray):
        cw = canvas.winfo_width() or 640
        ch = canvas.winfo_height() or 480
        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR),
                           cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=img_tk)
        canvas.image = img_tk

    # ── Quick enroll from camera ──────────────────────────────────────
    def _quick_enroll_modal(self):
        modal = tk.Toplevel(self.root)
        modal.title("Enroll Current Face")
        modal.geometry("360x200")
        modal.configure(bg=PANEL)
        modal.transient(self.root)
        modal.grab_set()
        modal.resizable(False, False)

        _label(modal, "What is this person's name?", 11, bold=True, color=WHITE).pack(pady=(24, 8))

        entry = tk.Entry(modal, font=("Segoe UI", 12), bg="#0f3460", fg=WHITE,
                         insertbackground=WHITE, relief=tk.FLAT, bd=8, justify="center")
        entry.pack(fill=tk.X, padx=30, ipady=5)
        entry.focus_set()

        def _confirm():
            name = entry.get().strip()
            if not name:
                return
            modal.destroy()
            self._enroll_name_entry.delete(0, tk.END)
            self._enroll_name_entry.insert(0, name)
            self._start_live_enrollment()

        entry.bind("<Return>", lambda e: _confirm())
        _btn(modal, "✅  Start Capturing", BTN_GREEN, _confirm).pack(pady=16)

    # ═══════════════════════════════════════════════════════════════════
    #  PAGE 2 — ENROLL / ADD NEW PERSON
    # ═══════════════════════════════════════════════════════════════════
    def _build_enroll_page(self):
        page = tk.Frame(self._content, bg=BG)
        self._pages["enroll"] = page

        # ── Header ────────────────────────────────────────────────────
        hdr = tk.Frame(page, bg=CARD, padx=24, pady=20,
                       highlightbackground="#2a2a4a", highlightthickness=1)
        hdr.pack(fill=tk.X, pady=(0, 16))

        _label(hdr, "➕   Add a New Person", 14, bold=True, color=WHITE).pack(anchor="w")
        _label(hdr, "Type their name, then choose how to add their photo.",
               9, color=MUTED).pack(anchor="w", pady=(4, 0))

        # ── Name input ────────────────────────────────────────────────
        name_card = tk.Frame(page, bg=CARD, padx=24, pady=18,
                             highlightbackground="#2a2a4a", highlightthickness=1)
        name_card.pack(fill=tk.X, pady=(0, 16))

        _label(name_card, "Step 1 — Enter the person's name:", 10, bold=True, color=WHITE).pack(anchor="w", pady=(0, 8))

        self._enroll_name_entry = tk.Entry(
            name_card, font=("Segoe UI", 13),
            bg="#0f3460", fg=WHITE,
            insertbackground=WHITE, relief=tk.FLAT, bd=10,
        )
        self._enroll_name_entry.pack(fill=tk.X, ipady=5)
        _label(name_card, "Example: Ahmed Mohamed", 8, color=MUTED).pack(anchor="w", pady=(4, 0))

        # ── Two method cards ─────────────────────────────────────────
        methods = tk.Frame(page, bg=BG)
        methods.pack(fill=tk.BOTH, expand=True)
        methods.columnconfigure(0, weight=1)
        methods.columnconfigure(1, weight=1)

        # Method A — Live camera
        card_a = tk.Frame(methods, bg=CARD, padx=22, pady=20,
                          highlightbackground="#2a2a4a", highlightthickness=1)
        card_a.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        _label(card_a, "📸  Method A  —  Use Camera", 11, bold=True, color=WHITE).pack(anchor="w", pady=(0, 10))
        _label(card_a,
               "Stand in front of the webcam.\n"
               "The app will take 5 quick photos of your face automatically.\n\n"
               "Best for: accurate recognition in all lighting conditions.",
               9, color=MUTED, justify="left", wraplength=320).pack(anchor="w", pady=(0, 16))

        self._btn_live = _btn(card_a, "📸  Start Camera Capture", BTN_BLUE,
                              self._start_live_enrollment, pady=10)
        self._btn_live.pack(fill=tk.X)

        self._enroll_status_lbl = _label(card_a, "", 9, color=WARN_CLR)
        self._enroll_status_lbl.pack(pady=(10, 0))

        # Method B — Upload photos
        card_b = tk.Frame(methods, bg=CARD, padx=22, pady=20,
                          highlightbackground="#2a2a4a", highlightthickness=1)
        card_b.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        _label(card_b, "📁  Method B  —  Upload Photos", 11, bold=True, color=WHITE).pack(anchor="w", pady=(0, 10))
        _label(card_b,
               "Select one or more clear portrait photos from your computer.\n\n"
               "Accepted formats: JPG, PNG, WEBP, BMP\n\n"
               "Best for: people who are not present right now.",
               9, color=MUTED, justify="left", wraplength=320).pack(anchor="w", pady=(0, 16))

        _btn(card_b, "📂  Choose Photos from Computer", BTN_GRAY,
             self._enroll_from_files, pady=10).pack(fill=tk.X)

    def _show_enroll(self):
        self._switch_page("enroll", "➕   Add New Person")

    def _start_live_enrollment(self):
        name = self._enroll_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Name Missing",
                "Please type the person's name first (Step 1).")
            return
        if not self.is_camera_running:
            self._start_camera()
            self._show_camera()
        self.is_enrolling     = True
        self.enroll_name      = name
        self.enroll_samples   = []
        self.enroll_countdown = 18  # ~0.5 sec warm-up at 30fps
        self._enroll_status_lbl.config(
            text=f"Camera starting… Look straight at the lens.")
        self.status_var.set(f"Getting ready to capture {name}'s face…")

    def _run_enrollment_frame(self, frame, results):
        if self.enroll_countdown > 0:
            self.enroll_countdown -= 1
            cv2.putText(frame, "GET READY...", (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
            return

        if len(results) == 1:
            # ── Duplicate check — runs only on the very first sample ─────────
            if len(self.enroll_samples) == 0:
                face_emb = results[0].get("embedding")
                if face_emb is not None:
                    matched_name = self.database.find_matching_identity(face_emb)
                    if matched_name is not None:
                        # This face is already registered — reject enrollment
                        self.is_enrolling = False
                        self._enroll_status_lbl.config(
                            text=f"⚠️  Already registered as '{matched_name}'")
                        self.status_var.set(
                            f"Enrollment stopped — face already registered as '{matched_name}'.")
                        messagebox.showwarning(
                            "Already Registered",
                            f"This person is already in the system as:\n\n"
                            f"       ➜  {matched_name}\n\n"
                            "Enrollment has been cancelled.\n"
                            "If you want to update their photos, please remove them first "
                            "from the People List and re-enroll."
                        )
                        return

            # ── Collect sample ────────────────────────────────────────────────
            self.enroll_samples.append(frame.copy())
            count = len(self.enroll_samples)
            self._enroll_status_lbl.config(text=f"📸  Captured {count} / {self.enroll_need} …")
            cv2.putText(frame, f"Photo {count}/{self.enroll_need}",
                        (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            if count >= self.enroll_need:
                self.is_enrolling = False
                ok, msg = self.database.add_person_from_frames(self.enroll_name, self.enroll_samples)
                self._enroll_status_lbl.config(text=msg)
                self.status_var.set(msg)
                self._update_identity_count()
                self._refresh_people_table()
                if ok:
                    messagebox.showinfo("✅  Person Added!",
                        f"'{self.enroll_name}' has been successfully registered.\n\n"
                        "They will now be recognized by the camera.")
                else:
                    messagebox.showerror("Enrollment Failed", msg)
        else:
            cv2.putText(frame, "⚠ Only one face please", (20, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 60, 255), 2, cv2.LINE_AA)

    def _enroll_from_files(self):
        name = self._enroll_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Name Missing",
                "Please type the person's name first (Step 1).")
            return
        files = filedialog.askopenfilenames(
            title=f"Select photos for  {name}",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if not files:
            return

        # ── Duplicate check: scan the first photo before saving anything ──
        self.status_var.set("Checking if this person is already registered…")
        self.root.update()
        try:
            import cv2, numpy as np
            first_img = cv2.imdecode(np.fromfile(files[0], dtype=np.uint8), cv2.IMREAD_COLOR)
            if first_img is not None:
                faces = self.engine.detect_faces(first_img)
                if faces is not None and len(faces) > 0:
                    best_face = max(faces, key=lambda f: f[2] * f[3])
                    emb = self.engine.extract_embedding(first_img, best_face)
                    matched = self.database.find_matching_identity(emb)
                    if matched is not None:
                        messagebox.showwarning(
                            "Already Registered",
                            f"The face in the selected photo is already registered as:\n\n"
                            f"       ➜  {matched}\n\n"
                            "Enrollment has been cancelled.\n"
                            "If you want to update their photos, remove them first "
                            "from the People List and re-enroll."
                        )
                        self.status_var.set(
                            f"Enrollment cancelled — face already registered as '{matched}'.")
                        return
        except Exception:
            pass  # If check fails for any reason, let the normal flow continue

        # ── No duplicate found — proceed with enrollment ───────────────
        self.status_var.set(f"Processing {len(files)} photo(s) for {name}…")
        self.root.update()
        ok, msg = self.database.add_person_from_images(name, list(files))
        self._update_identity_count()
        self._refresh_people_table()
        if ok:
            messagebox.showinfo("✅  Person Added!", msg)
        else:
            messagebox.showerror("Could not add person", msg)
        self.status_var.set(msg)

    # ═══════════════════════════════════════════════════════════════════
    #  PAGE 3 — ANALYZE A PHOTO
    # ═══════════════════════════════════════════════════════════════════
    def _build_inspect_page(self):
        page = tk.Frame(self._content, bg=BG)
        self._pages["inspect"] = page

        # Left — image preview
        left = tk.Frame(page, bg="#000000",
                        highlightbackground="#2a2a4a", highlightthickness=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        self._inspect_canvas = tk.Canvas(left, bg="#0a0a14", highlightthickness=0)
        self._inspect_canvas.pack(fill=tk.BOTH, expand=True)
        self._inspect_canvas.create_text(
            300, 220,
            text="🖼️  No photo selected yet\n\nPress  📂 Open Photo  to get started",
            font=("Segoe UI", 13), fill=MUTED, justify=tk.CENTER,
            tags="placeholder"
        )

        # Right — results
        right = tk.Frame(page, bg=PANEL, width=250, padx=16, pady=16)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        _label(right, "Photo Analysis", 11, bold=True, color=WHITE).pack(anchor="w", pady=(0, 12))
        _btn(right, "📂  Open Photo", BTN_BLUE, self._inspect_file, pady=9).pack(fill=tk.X, pady=(0, 8))

        self._btn_export = _btn(right, "💾  Save Annotated Copy", BTN_GRAY,
                                self._export_annotated, pady=7)
        self._btn_export.config(state=tk.DISABLED)
        self._btn_export.pack(fill=tk.X, pady=(0, 16))

        _sep(right)
        _label(right, "Results:", 10, bold=True, color=WHITE).pack(anchor="w", pady=(0, 6))

        self._inspect_text = tk.Text(right, bg=CARD, fg=TEXT, font=("Segoe UI", 9),
                                     wrap=tk.WORD, relief=tk.FLAT, padx=10, pady=10, height=18)
        self._inspect_text.pack(fill=tk.BOTH, expand=True)
        self._inspect_text.insert("1.0", "Results will appear here after you open a photo.")
        self._inspect_text.config(state=tk.DISABLED)

        self._last_annotated = None

    def _show_inspect(self):
        self._switch_page("inspect", "🔍   Analyze a Photo  —  Find Faces in any Image")

    def _inspect_file(self):
        path = filedialog.askopenfilename(
            title="Select a photo to analyze",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected file.")
            return

        display = img.copy()
        results = self.engine.process_frame(display, self.database.embeddings, draw_annotations=True)
        self._last_annotated = display
        self._btn_export.config(state=tk.NORMAL)

        self._inspect_canvas.delete("all")
        self._draw_frame(self._inspect_canvas, display)

        # Build readable report
        lines = []
        lines.append(f"📄  File: {Path(path).name}\n")
        lines.append(f"👤  Faces found: {len(results)}\n")
        lines.append("─" * 28 + "\n")
        if results:
            for i, r in enumerate(results, 1):
                status = "✅ Recognized" if r["is_known"] else "❓ Unknown"
                lines.append(f"Face #{i}\n")
                lines.append(f"  Name:       {r['name']}\n")
                lines.append(f"  Confidence: {r['confidence_pct']:.1f}%\n")
                lines.append(f"  Status:     {status}\n\n")
        else:
            lines.append("No faces were detected in this photo.\n\n"
                         "Try a clearer, well-lit portrait image.")

        self._inspect_text.config(state=tk.NORMAL)
        self._inspect_text.delete("1.0", tk.END)
        self._inspect_text.insert("1.0", "".join(lines))
        self._inspect_text.config(state=tk.DISABLED)

    def _export_annotated(self):
        if self._last_annotated is None:
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if dest:
            ext = Path(dest).suffix or ".jpg"
            _, buf = cv2.imencode(ext, self._last_annotated)
            buf.tofile(dest)
            messagebox.showinfo("Saved!", f"Annotated photo saved to:\n{dest}")

    # ═══════════════════════════════════════════════════════════════════
    #  PAGE 4 — PEOPLE LIST
    # ═══════════════════════════════════════════════════════════════════
    def _build_people_page(self):
        page = tk.Frame(self._content, bg=BG)
        self._pages["people"] = page

        # Top bar
        top = tk.Frame(page, bg=BG)
        top.pack(fill=tk.X, pady=(0, 10))

        _label(top, "Registered People", 12, bold=True, color=WHITE).pack(side=tk.LEFT)

        _btn(top, "🗑️  Remove Selected", BTN_RED,
             self._delete_person, pady=5).pack(side=tk.RIGHT, padx=4)
        _btn(top, "📂  Open Faces Folder", BTN_GRAY,
             self._open_faces_folder, pady=5).pack(side=tk.RIGHT, padx=4)
        _btn(top, "🔄  Refresh List", BTN_GRAY,
             self._refresh_people_table, pady=5).pack(side=tk.RIGHT, padx=4)

        # Helper text
        _label(page, "These are all the people the system currently knows and can recognize.",
               9, color=MUTED).pack(anchor="w", pady=(0, 8))

        # Table
        style = ttk.Style()
        style.configure("People.Treeview",
                         background=CARD, foreground=TEXT,
                         fieldbackground=CARD, rowheight=34,
                         font=("Segoe UI", 10))
        style.configure("People.Treeview.Heading",
                         background=PANEL, foreground=WHITE,
                         font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("People.Treeview",
                  background=[("selected", BTN_BLUE)],
                  foreground=[("selected", WHITE)])

        cols = ("name", "photos", "status")
        self._people_tree = ttk.Treeview(page, columns=cols, show="headings",
                                          selectmode="browse", style="People.Treeview")
        self._people_tree.heading("name",   text="  Name")
        self._people_tree.heading("photos", text="Photos")
        self._people_tree.heading("status", text="Recognition Status")
        self._people_tree.column("name",   width=280, anchor="w")
        self._people_tree.column("photos", width=100, anchor="center")
        self._people_tree.column("status", width=200, anchor="center")

        sb = ttk.Scrollbar(page, orient=tk.VERTICAL, command=self._people_tree.yview)
        self._people_tree.configure(yscroll=sb.set)
        self._people_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_people_table()

    def _show_people(self):
        self._switch_page("people", "👥   People List  —  Everyone the System Knows")
        self._refresh_people_table()

    def _refresh_people_table(self):
        for item in self._people_tree.get_children():
            self._people_tree.delete(item)
        for p in self.database.list_people():
            status = "✅ Active" if p["is_active"] else "⚠️ Needs refresh"
            self._people_tree.insert("", tk.END,
                values=(f"  {p['name']}", p["photo_count"], status))

    def _update_identity_count(self):
        count = len(self.database.embeddings)
        self._id_count_lbl.config(text=f"People registered: {count}")

    def _delete_person(self):
        sel = self._people_tree.selection()
        if not sel:
            messagebox.showwarning("Nothing Selected",
                "Please click on a person's name first.")
            return
        name = self._people_tree.item(sel[0])["values"][0].strip()
        if messagebox.askyesno("Remove Person",
            f"Remove  '{name}'  from the system?\n\n"
            "This will delete all their photos and recognition data."):
            self.database.delete_person(name)
            self._refresh_people_table()
            self._update_identity_count()
            messagebox.showinfo("Removed", f"'{name}' has been removed.")

    def _open_faces_folder(self):
        folder = str(self.database.faces_dir)
        if os.name == "nt":
            os.startfile(folder)
        else:
            subprocess.run(["xdg-open", folder])

    # ═══════════════════════════════════════════════════════════════════
    #  PAGE 5 — ATTENDANCE LOG
    # ═══════════════════════════════════════════════════════════════════
    def _build_attendance_page(self):
        page = tk.Frame(self._content, bg=BG)
        self._pages["attendance"] = page

        # Top bar
        top = tk.Frame(page, bg=BG)
        top.pack(fill=tk.X, pady=(0, 10))

        _label(top, "Attendance Log", 12, bold=True, color=WHITE).pack(side=tk.LEFT)

        _btn(top, "📊  Open in Excel", BTN_GREEN,
             self._open_csv, pady=5).pack(side=tk.RIGHT, padx=4)
        _btn(top, "🔄  Refresh", BTN_GRAY,
             self._refresh_attendance, pady=5).pack(side=tk.RIGHT, padx=4)

        _label(page,
               "Every time someone is recognized by the camera, it is automatically recorded here.",
               9, color=MUTED).pack(anchor="w", pady=(0, 8))

        # Table
        style = ttk.Style()
        style.configure("Att.Treeview",
                         background=CARD, foreground=TEXT,
                         fieldbackground=CARD, rowheight=32,
                         font=("Segoe UI", 10))
        style.configure("Att.Treeview.Heading",
                         background=PANEL, foreground=WHITE,
                         font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("Att.Treeview",
                  background=[("selected", BTN_BLUE)],
                  foreground=[("selected", WHITE)])

        cols = ("date", "time", "name", "confidence", "status")
        self._att_tree = ttk.Treeview(page, columns=cols, show="headings",
                                       style="Att.Treeview")
        self._att_tree.heading("date",       text="Date")
        self._att_tree.heading("time",       text="Time")
        self._att_tree.heading("name",       text="Person")
        self._att_tree.heading("confidence", text="Confidence")
        self._att_tree.heading("status",     text="Status")
        self._att_tree.column("date",       width=110, anchor="center")
        self._att_tree.column("time",       width=110, anchor="center")
        self._att_tree.column("name",       width=230)
        self._att_tree.column("confidence", width=120, anchor="center")
        self._att_tree.column("status",     width=130, anchor="center")

        sb = ttk.Scrollbar(page, orient=tk.VERTICAL, command=self._att_tree.yview)
        self._att_tree.configure(yscroll=sb.set)
        self._att_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._refresh_attendance()

    def _show_attendance(self):
        self._switch_page("attendance", "📋   Attendance Log  —  Who Was Recognized & When")
        self._refresh_attendance()

    def _refresh_attendance(self):
        for item in self._att_tree.get_children():
            self._att_tree.delete(item)
        for r in self.attendance.get_recent_records(limit=200):
            if len(r) >= 5:
                self._att_tree.insert("", tk.END, values=r[:5])

    def _open_csv(self):
        f = str(self.attendance.log_file)
        if os.name == "nt":
            os.startfile(f)
        else:
            subprocess.run(["xdg-open", f])

    # ═══════════════════════════════════════════════════════════════════
    #  PAGE 6 — SETTINGS
    # ═══════════════════════════════════════════════════════════════════
    def _build_settings_page(self):
        page = tk.Frame(self._content, bg=BG)
        self._pages["settings"] = page

        card = tk.Frame(page, bg=CARD, padx=30, pady=26,
                        highlightbackground="#2a2a4a", highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True)

        _label(card, "⚙️   Settings", 14, bold=True, color=WHITE).pack(anchor="w")
        _label(card, "Adjust how sensitive the recognition is. Most users can leave these at the defaults.",
               9, color=MUTED).pack(anchor="w", pady=(4, 20))

        # ── Slider helper ─────────────────────────────────────────────
        def slider_section(parent, title, hint, from_, to_, default, on_change):
            tk.Frame(parent, bg="#2a2a4a", height=1).pack(fill=tk.X, pady=(0, 14))
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill=tk.X, pady=(0, 4))
            _label(row, title, 10, bold=True, color=WHITE).pack(side=tk.LEFT, anchor="w")
            val_lbl = _label(row, f"{default}", 10, bold=True, color=ACCENT)
            val_lbl.pack(side=tk.RIGHT)
            _label(parent, hint, 8, color=MUTED).pack(anchor="w", pady=(0, 6))
            sl = ttk.Scale(parent, from_=from_, to=to_, value=default,
                           command=lambda v, lbl=val_lbl: on_change(v, lbl))
            sl.pack(fill=tk.X, pady=(0, 16))
            return sl

        # Cosine threshold
        def _on_cosine(v, lbl):
            f = float(v)
            self.engine.set_cosine_threshold(f)
            lbl.config(text=f"{f:.2f}")

        slider_section(
            card,
            "Recognition Sensitivity",
            "Default: 0.38  |  Lower = recognizes more easily (may confuse similar faces)  |  Higher = stricter match",
            0.20, 0.70, self.engine.cosine_thresh, _on_cosine
        )

        # Detection threshold
        def _on_detect(v, lbl):
            f = float(v)
            self.engine.set_detector_threshold(f)
            lbl.config(text=f"{f:.2f}")

        slider_section(
            card,
            "Face Detection Confidence",
            "Default: 0.75  |  Lower = detects more faces  |  Higher = only detects clearly visible faces",
            0.40, 0.95, self.engine.score_thresh, _on_detect
        )

        # Attendance cooldown
        def _on_cool(v, lbl):
            i = int(float(v))
            self.attendance.cooldown_seconds = i
            lbl.config(text=f"{i}s")

        slider_section(
            card,
            "Attendance Log Cooldown",
            "Default: 60 s  |  Prevents logging the same person over and over within this time window.",
            10, 300, self.attendance.cooldown_seconds, _on_cool
        )

        # Reset tip
        _label(card,
               "💡  Tip: If the camera is not recognizing someone correctly,\n"
               "    go to 'Add New Person' and add more photos of them.",
               9, color=MUTED).pack(anchor="w", pady=(10, 0))

    def _show_settings(self):
        self._switch_page("settings", "⚙️   Settings  —  Adjust Recognition Sensitivity")

    # ═══════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
def start_gui():
    root = tk.Tk()
    FaceApp(root)
    root.mainloop()


if __name__ == "__main__":
    start_gui()
