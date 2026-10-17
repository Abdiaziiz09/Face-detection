import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class AttendanceLogger:
    """
    Logs recognized faces into an attendance CSV file with timestamp and confidence.
    Includes a cooldown period to avoid duplicate logs for the same person.
    """

    def __init__(self, log_dir: str = "logs", cooldown_seconds: int = 60):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "attendance.csv"
        self.cooldown_seconds = cooldown_seconds
        self.last_logged: Dict[str, float] = {}

        self._ensure_header()

    def _ensure_header(self):
        if not self.log_file.exists() or self.log_file.stat().st_size == 0:
            with open(self.log_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Person Name", "Confidence (%)", "Status"])

    def log(self, person_name: str, confidence_pct: float) -> bool:
        """
        Logs a detection if person is not in cooldown.
        Returns True if newly logged, False if skipped due to cooldown.
        """
        if person_name.lower() == "unknown":
            return False

        now = time.time()
        last_time = self.last_logged.get(person_name, 0.0)

        if now - last_time < self.cooldown_seconds:
            return False  # Cooldown active

        self.last_logged[person_name] = now
        now_dt = datetime.now()
        date_str = now_dt.strftime("%Y-%m-%d")
        time_str = now_dt.strftime("%H:%M:%S")

        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([date_str, time_str, person_name, f"{confidence_pct:.1f}%", "Verified"])

        print(f"[Attendance Logged] {person_name} ({confidence_pct:.1f}%) at {time_str}")
        return True

    def get_recent_records(self, limit: int = 100) -> List[List[str]]:
        """Reads recent attendance entries from CSV."""
        if not self.log_file.exists():
            return []

        records = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) > 1:
                    records = rows[1:]  # skip header
                    records.reverse()    # newest first
                    return records[:limit]
        except Exception as e:
            print(f"[Attendance Read Error] {e}")
        return []
