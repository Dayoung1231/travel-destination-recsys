"""
TourAPI(KorService2)에서 수도권 관광지 목록을 받아 data/raw/tourapi/area_list.csv로 저장해요.

실행 (프로젝트 최상위 폴더에서):
    python -m src.preprocessing.collect_tourapi

.env에 TOURAPI_KEY(공공데이터포털 일반 인증키)가 있어야 해요.
"""
import time

import pandas as pd
import requests
from dotenv import dotenv_values

from src.config import ROOT, TOURAPI_DIR

BASE = "https://apis.data.go.kr/B551011/KorService2"
REGIONS = {"11": "서울", "28": "인천", "41": "경기"}  # 법정동 시·도 코드
TYPES = {"12": "관광지", "14": "문화시설", "15": "축제공연행사", "28": "레포츠"}  # 관광 타입
ROWS = 1000  # 한 페이지에 받을 개수
IMAGE_COLS = ["firstimage", "firstimage2", "cpyrhtDivCd"]  # 이미지마다 이용 조건이 달라서 저장하지 않음


def load_key():
    """.env에서 TourAPI 인증키를 읽는다."""
    key = dotenv_values(ROOT / ".env").get("TOURAPI_KEY", "").strip()
    if not key:
        raise SystemExit("TOURAPI_KEY가 없어요. .env.example을 복사해 .env를 만들고 인증키를 적어 주세요.")
    return key


def fetch_all(common, region, ctype):
    """한 지역 × 한 타입의 장소를 모든 페이지에 걸쳐 받아 리스트로 돌려준다."""
    rows, page = [], 1
    while True:
        params = {**common, "lDongRegnCd": region, "contentTypeId": ctype,
                  "arrange": "A", "numOfRows": ROWS, "pageNo": page}
        body = requests.get(f"{BASE}/areaBasedList2", params=params, timeout=30).json()["response"]["body"]
        items = body["items"]["item"] if body["items"] else []
        if isinstance(items, dict):  # 결과가 1건이면 리스트가 아니라 딕셔너리로 온다
            items = [items]
        rows.extend(items)
        if len(rows) >= body["totalCount"] or not items:
            return rows
        page += 1
        time.sleep(0.2)  # 서버에 부담을 주지 않게 잠깐 쉬기


def main():
    common = {"serviceKey": load_key(), "MobileOS": "ETC", "MobileApp": "TravelRecsys", "_type": "json"}

    all_rows = []
    for region, rname in REGIONS.items():
        for ctype, tname in TYPES.items():
            rows = fetch_all(common, region, ctype)
            print(f"{rname} {tname}: {len(rows)}곳")
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop(columns=IMAGE_COLS, errors="ignore")
    df = df.drop_duplicates("contentid")

    TOURAPI_DIR.mkdir(parents=True, exist_ok=True)
    out = TOURAPI_DIR / "area_list.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"총 {len(df)}곳 저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
