"""KMA 이상 패턴 조회 API. 분석 결과(data/kma.db)를 읽기 전용으로 제공한다."""
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "kma.db"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
ELEMENTS = {"TA_mean", "TA_max", "TA_min", "HM_mean", "WS_mean", "PS_mean"}
CATEGORIES = {"기상 이상", "관측 의심", "저풍속(정상 범위)", "정상"}

app = FastAPI(title="KMA Anomaly", description="기상청 관측자료 이상 패턴 탐지")


def query(sql, params=()):
    """DB를 읽기 전용으로 열어 조회한 뒤 dict 목록으로 돌려준다."""
    if not DB_PATH.exists():
        raise HTTPException(503, "DB 파일이 없습니다. collector 스크립트를 먼저 실행하세요.")
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    return {
        "period": query("SELECT MIN(date) AS start, MAX(date) AS end FROM anomaly_daily")[0],
        "categories": query(
            "SELECT category, COUNT(*) AS n FROM anomaly_daily GROUP BY category ORDER BY n DESC"),
        "sensor_alert_days": query(
            "SELECT COUNT(*) AS n FROM sensor_health_daily WHERE alert = 1")[0]["n"],
    }


@app.get("/api/anomalies")
def anomalies(
    start: Optional[str] = Query(None, pattern=DATE_PATTERN, description="시작일 YYYY-MM-DD"),
    end: Optional[str] = Query(None, pattern=DATE_PATTERN, description="종료일 YYYY-MM-DD"),
    element: Optional[str] = Query(None, description="TA_mean, TA_max, TA_min, HM_mean, WS_mean, PS_mean"),
    category: str = Query("기상 이상", description="기상 이상, 관측 의심, 저풍속(정상 범위), 정상"),
    limit: int = Query(100, ge=1, le=1000),
):
    if element and element not in ELEMENTS:
        raise HTTPException(422, f"element는 {sorted(ELEMENTS)} 중 하나여야 합니다.")
    if category not in CATEGORIES:
        raise HTTPException(422, f"category는 {sorted(CATEGORIES)} 중 하나여야 합니다.")

    sql = ("SELECT date, element, value, median, z, pct, category, ws_suspect_hours "
           "FROM anomaly_daily WHERE category = ?")
    params = [category]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    if element:
        sql += " AND element = ?"
        params.append(element)
    sql += " ORDER BY ABS(z) DESC LIMIT ?"
    params.append(limit)
    return query(sql, params)


@app.get("/api/anomalies/{date}")
def anomalies_of_day(date: str):
    rows = query(
        "SELECT date, element, value, median, z, pct, category, ws_suspect_hours "
        "FROM anomaly_daily WHERE date = ? ORDER BY element", (date,))
    if not rows:
        raise HTTPException(404, f"{date} 자료가 없습니다. (유효 시간이 부족한 날은 제외됩니다)")
    return rows


@app.get("/api/sensor-health")
def sensor_health(
    start: Optional[str] = Query(None, pattern=DATE_PATTERN),
    end: Optional[str] = Query(None, pattern=DATE_PATTERN),
    only_alert: bool = Query(False, description="센서 이상 징후 판정일만 보기"),
    limit: int = Query(366, ge=1, le=5000),
):
    sql = ("SELECT date, zero_ratio_7d, mean_ws_7d, over_limit, alert "
           "FROM sensor_health_daily WHERE 1 = 1")
    params = []
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    if only_alert:
        sql += " AND alert = 1"
    sql += " ORDER BY date LIMIT ?"
    params.append(limit)
    return query(sql, params)
