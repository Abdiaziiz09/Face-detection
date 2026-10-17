@echo off
title FaceEngine AI - Face Recognition & Attendance
echo ========================================================
echo  Launching Face Recognition System (100%% Local & Free)...
echo ========================================================
python run_gui.py
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit.
    pause >nul
)
