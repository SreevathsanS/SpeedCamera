# database_manager.py

import sqlite3
import logging

class DatabaseManager:
    """
    Manages all database operations for the speed camera application.
    This class is responsible for connecting to the database,
    creating the necessary table, and logging overspeeding events.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_table()

    def _connect(self):
        """
        Establishes a connection to the SQLite database.
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logging.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")
            raise

    def _create_table(self):
        """
        Creates the 'overspeeding_log' table if it doesn't already exist.
        The table stores details of vehicles that have exceeded the speed limit.
        """
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS overspeeding_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    vehicle_id INTEGER NOT NULL,
                    speed_kmh REAL NOT NULL,
                    plate_number TEXT
                )
            ''')
            self.conn.commit()
            logging.info("Ensured 'overspeeding_log' table exists.")
        except sqlite3.Error as e:
            logging.error(f"Database table creation error: {e}")
            raise

    def log_overspeeding_event(self, timestamp: str, camera_id: str, vehicle_id: int, speed_kmh: float, plate_number: str = "UNKNOWN"):
        """
        Inserts an overspeeding event record into the database.

        Args:
            timestamp (str): The timestamp of the event.
            camera_id (str): The ID of the camera that detected the event.
            vehicle_id (int): The unique tracking ID of the vehicle.
            speed_kmh (float): The detected speed in kilometers per hour.
            plate_number (str): The detected license plate number, defaults to "UNKNOWN".
        """
        try:
            self.cursor.execute(
                "INSERT INTO overspeeding_log (timestamp, camera_id, vehicle_id, speed_kmh, plate_number) VALUES (?, ?, ?, ?, ?)",
                (timestamp, camera_id, vehicle_id, speed_kmh, plate_number)
            )
            self.conn.commit()
            logging.warning(f"Overspeeding event logged for Cam {camera_id}, ID {vehicle_id} at {int(speed_kmh)} km/h.")
        except sqlite3.Error as e:
            logging.error(f"Failed to log event to database: {e}")

    def close(self):
        """
        Closes the database connection.
        """
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")