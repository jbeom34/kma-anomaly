"""asos_hourly_clean에서 행 자체가 없는 시각을 찾아 연속 구간으로 묶어 보여준다."""
import sqlite3

import pandas as pd

with sqlite3.connect("data/kma.db") as conn:
    df = pd.read_sql("SELECT STN, TM FROM asos_hourly_clean", conn)

for stn, g in df.groupby("STN"):
    tm = pd.to_datetime(g["TM"])
    full = pd.date_range(tm.min(), tm.max(), freq="h")
    missing = full.difference(tm)
    print(f"지점 {stn}: {tm.min()} ~ {tm.max()}")
    print(f"  기대 {len(full)}행, 실제 {len(tm)}행, 빠진 시각 {len(missing)}개")
    if len(missing) == 0:
        continue
    s = pd.Series(missing)
    group_id = (s.diff() != pd.Timedelta(hours=1)).cumsum()
    for _, part in s.groupby(group_id):
        print(f"  {part.iloc[0]} ~ {part.iloc[-1]} ({len(part)}시간)")
