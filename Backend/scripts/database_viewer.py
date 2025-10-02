import sqlite3
import pandas as pd   # optional, for nicer display

db_path = "Backend/outputs/overspeeding_log.db"
conn = sqlite3.connect(db_path)

# Using pandas for a quick table view
df = pd.read_sql_query("SELECT * FROM overspeeding_log;", conn)
print(df)

conn.close()
