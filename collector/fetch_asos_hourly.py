"""ASOS 시간자료를 기간별로 나눠 받아 data/raw에 저장한다."""
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("KMA_AUTH_KEY")

URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm3.php"
STN = 159                    # ASOS 지점번호 (159 = 부산)
START = date(2024, 1, 1)
END = date(2024, 12, 31)
CHUNK_DAYS = 30              # 한 번에 요청 가능한 최대 기간


def chunks(start, end, days):
    """start~end를 days일 단위로 나눠 (시작일, 종료일)을 하나씩 돌려준다."""
    cur = start
    while cur <= end:
        last = min(cur + timedelta(days=days - 1), end)
        yield cur, last
        cur = last + timedelta(days=1)


def fetch(tm1, tm2):
    params = {"tm1": tm1, "tm2": tm2, "stn": STN, "help": 1, "authKey": KEY}
    r = requests.get(URL, params=params, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.text


def main():
    if not KEY:
        raise SystemExit(".env 파일에 KMA_AUTH_KEY가 없습니다.")

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"asos_hourly_{STN}_{START:%Y%m%d}_{END:%Y%m%d}.txt"

    header_done = False
    total = 0
    with path.open("w", encoding="utf-8") as f:
        for a, b in chunks(START, END, CHUNK_DAYS):
            text = fetch(f"{a:%Y%m%d}0000", f"{b:%Y%m%d}2300")
            lines = text.splitlines()
            data = [x for x in lines if x.strip() and not x.startswith("#")]

            if not data:  # 데이터가 없으면 서버가 보낸 원문을 보여준다
                print(f"[경고] {a}~{b} 자료 없음. 응답 앞부분:\n{text[:300]}")
                continue

            if not header_done:  # 컬럼 설명(# 줄)은 처음 한 번만 저장
                f.write("\n".join(x for x in lines if x.startswith("#")) + "\n")
                header_done = True

            f.write("\n".join(data) + "\n")
            total += len(data)
            print(f"{a} ~ {b}: {len(data)}행")
            time.sleep(0.5)  # 서버 부담을 줄이기 위한 대기

    print(f"완료: 총 {total}행 -> {path}")


if __name__ == "__main__":
    main()
