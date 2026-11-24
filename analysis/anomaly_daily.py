"""일 단위 이상탐지: 같은 날짜 앞뒤 7일, 다른 해들의 분포(중앙값·MAD)와 비교한다."""
import sqlite3

import numpy as np
import pandas as pd

STN = 159
WINDOW = 7             # 같은 날짜 앞뒤 며칠을 기준 구간으로 쓸지
MIN_VALID_HOURS = 20   # 하루 중 유효 시간이 이보다 적으면 그날은 제외
MIN_BASE = 30          # 기준 표본이 이보다 적으면 점수를 내지 않음
Z_LIMIT = 3.0          # |z|가 이 값 이상이면 이상
LOG_ELEMENTS = {"WS_mean"}  # 오른쪽으로 치우친 분포라 로그를 취해서 비교
SPECS = {
    "TA_mean": ("TA", "mean"), "TA_max": ("TA", "max"), "TA_min": ("TA", "min"),
    "HM_mean": ("HM", "mean"), "WS_mean": ("WS", "mean"), "PS_mean": ("PS", "mean"),
}
# 대표 사례로 기억하는 날짜입니다. 실제 기록과 직접 대조해 보세요.
CHECK_DAYS = ["2018-01-25", "2018-08-01", "2020-09-03", "2020-09-07", "2022-09-06"]


def load_hourly(stn):
    with sqlite3.connect("data/kma.db") as conn:
        df = pd.read_sql(
            "SELECT TM, TA, HM, WS, PS FROM asos_hourly_clean WHERE STN = ?",
            conn, params=(stn,),
        )
    df["TM"] = pd.to_datetime(df["TM"])
    return df.set_index("TM").sort_index()


def to_daily(h):
    g = h.groupby(h.index.date)
    out = {}
    for name, (col, fn) in SPECS.items():
        s = g[col].agg(fn)
        s[g[col].count() < MIN_VALID_HOURS] = np.nan
        out[name] = s
    d = pd.DataFrame(out)
    d.index = pd.to_datetime(d.index)
    return d


def day_of_year_365(idx):
    """윤년 3월 이후를 하루 당겨 1~365 체계로 맞춘다."""
    doy = idx.day_of_year.to_numpy()
    return np.where(idx.is_leap_year & (idx.month > 2), doy - 1, doy)


def score(d):
    years = d.index.year.to_numpy()
    doy = day_of_year_365(d.index)
    rows = []
    for name in d.columns:
        x = d[name].to_numpy()
        valid = ~np.isnan(x)
        xt = np.log(np.clip(x, 0.1, None)) if name in LOG_ELEMENTS else x
        for i in np.where(valid)[0]:
            diff = np.abs(doy - doy[i])
            near = np.minimum(diff, 365 - diff) <= WINDOW   # 연말·연초도 이어서 계산
            base = x[near & valid & (years != years[i])]
            if len(base) < MIN_BASE:
                continue
            med = np.median(base)                      # 표시용(원래 단위)
            bt = xt[near & valid & (years != years[i])]
            medt = np.median(bt)
            mad = np.median(np.abs(bt - medt)) * 1.4826
            if mad == 0:
                continue
            rows.append((d.index[i].strftime("%Y-%m-%d"), name, x[i], med,
                         (xt[i] - medt) / mad, (base < x[i]).mean() * 100))
    res = pd.DataFrame(rows, columns=["date", "element", "value", "median", "z", "pct"])
    res["flag"] = (res["z"].abs() >= Z_LIMIT).astype(int)
    return res.round({"value": 2, "median": 2, "z": 2, "pct": 1})


def main():
    d = to_daily(load_hourly(STN))
    print(f"일 자료 {len(d)}일 (유효 시간 {MIN_VALID_HOURS}시간 미만인 날은 해당 요소에서 제외)")
    res = score(d)
    res.insert(0, "STN", STN)
    with sqlite3.connect("data/kma.db") as conn:
        res.to_sql("anomaly_daily", conn, if_exists="replace", index=False)

    print(f"점수 계산 {len(res)}건, 이상(|z|>={Z_LIMIT}) {int(res['flag'].sum())}건")

    print("\n=== 요소별 이상 건수 ===")
    t = res.groupby("element")["flag"].agg(["sum", "count"])
    t["비율(%)"] = (t["sum"] / t["count"] * 100).round(1)
    print(t.to_string())

    print("\n=== |z|가 큰 상위 20건 ===")
    top = res.reindex(res["z"].abs().sort_values(ascending=False).index).head(20)
    print(top[["date", "element", "value", "median", "z", "pct"]].to_string(index=False))

    print("\n=== 확인용 날짜 ===")
    chk = res[res["date"].isin(CHECK_DAYS)].sort_values(["date", "element"])
    print(chk[["date", "element", "value", "median", "z", "pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
