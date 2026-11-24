"""풍속계 건강도: 7일 이동 풍속 0 비율로 센서 이상 징후 구간을 찾는다."""
import sqlite3

import pandas as pd

STN = 159
WINDOW = 168        # 7일(시간 단위)
MIN_VALID = 120     # 창 안의 유효 시간이 이보다 적으면 계산하지 않음
RATIO_LIMIT = 8.0   # 7일 풍속 0 비율(%) 기준
MIN_DAYS = 8        # 단발 1회는 7일 창에서 최대 7일만 지속하므로, 8일 이상 이어질 때만 징후로 본다


def load(stn):
    with sqlite3.connect("data/kma.db") as conn:
        h = pd.read_sql("SELECT TM, WS FROM asos_hourly_clean WHERE STN = ?",
                        conn, params=(stn,))
    h["TM"] = pd.to_datetime(h["TM"])
    h = h.set_index("TM").sort_index()
    return h.reindex(pd.date_range(h.index.min(), h.index.max(), freq="h"))


def daily_health(h):
    valid = h["WS"].notna().astype(float)
    zero = (h["WS"] == 0).astype(float)
    n = valid.rolling(WINDOW, min_periods=1).sum()
    z = zero.rolling(WINDOW, min_periods=1).sum()
    ratio = (z / n * 100).where(n >= MIN_VALID)
    mean_ws = h["WS"].rolling(WINDOW, min_periods=MIN_VALID).mean()
    d = pd.DataFrame({"zero_ratio_7d": ratio, "mean_ws_7d": mean_ws})
    d = d[d.index.hour == 23].copy()          # 매일 23시 값을 그날의 값으로 사용
    d.index = d.index.normalize()
    d["over_limit"] = (d["zero_ratio_7d"] >= RATIO_LIMIT).astype(int)
    return d.round(2)


def episodes(d):
    flag = d["over_limit"] == 1
    grp = (flag != flag.shift()).cumsum()
    rows = []
    for _, g in d[flag].groupby(grp[flag]):
        rows.append((g.index[0].date(), g.index[-1].date(), len(g),
                     g["zero_ratio_7d"].max(), g["mean_ws_7d"].min()))
    ep = pd.DataFrame(rows, columns=["시작", "끝", "일수", "최대_정온비율(%)", "최저_7일평균풍속"])
    ep["센서이상징후"] = ep["일수"] >= MIN_DAYS
    return ep


def main():
    d = daily_health(load(STN))
    ep = episodes(d)

    alert_days = set()
    for _, r in ep[ep["센서이상징후"]].iterrows():
        alert_days.update(pd.date_range(r["시작"], r["끝"]).strftime("%Y-%m-%d"))

    out = d.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["alert"] = out["date"].isin(alert_days).astype(int)
    out.insert(0, "STN", STN)
    with sqlite3.connect("data/kma.db") as conn:
        out.to_sql("sensor_health_daily", conn, if_exists="replace", index=False)

    print(f"일 단위 건강도 {len(out)}일 저장 (7일 정온 비율 {RATIO_LIMIT}% 이상: {int(d['over_limit'].sum())}일)")
    print(f"\n=== 기준 이상이 이어진 구간 (센서이상징후 = {MIN_DAYS}일 이상 연속) ===")
    print(ep.to_string(index=False))
    print(f"\n센서 이상 징후 판정 구간: {int(ep['센서이상징후'].sum())}개")


if __name__ == "__main__":
    main()
