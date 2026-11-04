import sqlite3
import datetime

import pandas as pd

DB = "arena.db"


def init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        model TEXT,
        mode TEXT,
        seconds REAL,
        tokens INTEGER,
        ok INTEGER,
        fallback INTEGER,
        reason TEXT,
        prompt_len INTEGER)""")
    con.commit()
    con.close()


def log(model, mode, seconds, tokens, ok=True, fallback=False, reason="", prompt_len=0):
    con = sqlite3.connect(DB)
    con.execute(
        "INSERT INTO runs (ts, model, mode, seconds, tokens, ok, fallback, reason, prompt_len) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (datetime.datetime.now().isoformat(timespec="seconds"), model, mode,
         seconds, tokens, int(ok), int(fallback), reason, prompt_len),
    )
    con.commit()
    con.close()


def read_all():
    con = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM runs", con)
    con.close()
    return df