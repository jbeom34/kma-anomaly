"""의심 값 점검: 기온 -9.0, 기압변화 -9.0, 강수 '불명' 구간의 성격."""
import sqlite3

import pandas as pd

pd.set_option("display.width", 200)
with sqlite3.connect("data/kma.db") as conn:
    raw = pd.read_sql("SELECT * FROM asos_hourly", conn)
    cln = pd.read_sql("SELECT * FROM asos_hourly_clean", conn)

raw["TM"] = pd.to_datetime(raw["TM"])
raw = raw.set_index("TM").sort_index()
raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="h"))
ta = raw["TA"].where(raw["TA"] > -90)

print("=== 1. 기온이 -9.0인 행: 앞뒤 시각 기온과 비교 ===")
m = ta == -9.0
print(pd.DataFrame({"이전 시각": ta.shift(1), "기온": ta, "다음 시각": ta.shift(-1),
                    "습도": raw["HM"], "이슬점": raw["TD"]})[m].to_string())

print("\n=== 2. PR이 -9.0인데 PT 값이 있는 행: 현지기압(PA) 3시간 변화와 비교 ===")
pa_diff = (raw["PA"] - raw["PA"].shift(3)).round(1)
m = (raw["PR"] == -9.0) & raw["PT"].notna()
print(pd.DataFrame({"PR": raw["PR"], "PT": raw["PT"], "PA 3시간 변화": pa_diff})[m].to_string())

print("\n=== 3. 강수 상태별 성격 (불명이 무강수와 비슷한지, 비 온 시간과 비슷한지) ===")
g = cln.groupby("RN_STATUS").agg(
    행수=("TM", "size"),
    습도중앙값=("HM", "median"),
    습도90이상비율=("HM", lambda s: round((s >= 90).mean() * 100, 1)),
    시정중앙값=("VS", "median"),
    운량중앙값=("CA_TOT", "median"),
)
print(g.to_string())

print("\n=== 4. IR=3(무강수)인데 RN>0인 행의 강수량 분포 ===")
x = raw[(raw["IR"] == 3) & (raw["RN"] > 0)]["RN"]
print(x.describe().round(2).to_string())
print(x.value_counts().head(8).to_string())

print(f"\n=== 5. 정리본 기온 범위: {cln['TA'].where(cln['TA'] > -90).min()} ~ {cln['TA'].max()}")
