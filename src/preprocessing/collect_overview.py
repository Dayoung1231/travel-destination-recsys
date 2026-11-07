"""
TourAPI(KorService2)에서 매칭된 장소의 설명(overview)을 받아 data/raw/tourapi/overview.csv로 저장해요.

전처리를 먼저 돌려서 data/processed/items.csv에 TOUR_ID가 있어야 해요.
.env에 TOURAPI_KEY(공공데이터포털 일반 인증키)가 있어야 해요.

실행 (프로젝트 최상위 폴더에서):
    python -m src.preprocessing.collect_overview --limit 3   # 먼저 3곳만 시험
    python -m src.preprocessing.collect_overview             # 전체 수집

이미 받은 장소는 건너뛰어요. 중간에 멈춰도(호출 한도 초과 등) 다시 실행하면 이어서 받아요.
설명 원문을 그대로 저장하고, HTML 태그 제거 같은 정리는 토큰을 만들 때 해요.
"""
import argparse
import time

import pandas as pd
import requests

from src.config import PROCESSED_DIR, ROOT, TOURAPI_DIR
from src.preprocessing.collect_tourapi import BASE, load_key

OUT = TOURAPI_DIR / "overview.csv"
SLEEP = 0.2       # 서버에 부담을 주지 않게 호출 사이에 잠깐 쉬기
SAVE_EVERY = 50   # 이만큼 받을 때마다 저장


def target_ids():
    """items.csv에서 매칭된 TourAPI contentid 목록 (중복 제거, 정렬)."""
    ids = pd.read_csv(PROCESSED_DIR / "items.csv", usecols=["TOUR_ID"], dtype=str)["TOUR_ID"]
    return sorted(ids.dropna().unique())


def load_done():
    """이미 받은 표. 설명이 빈 장소도 다시 호출하지 않게 빈칸을 그대로 둔다."""
    if not OUT.exists():
        return pd.DataFrame(columns=["contentid", "overview"], dtype=str)
    return pd.read_csv(OUT, dtype=str, keep_default_na=False)


def save(done):
    TOURAPI_DIR.mkdir(parents=True, exist_ok=True)
    done.to_csv(OUT, index=False, encoding="utf-8-sig")


def fetch_overview(common, cid):
    """장소 하나의 설명. 설명이 없으면 빈 문자열, API 오류면 RuntimeError (메시지에 인증키를 넣지 않는다)."""
    res = requests.get(f"{BASE}/detailCommon2", params={**common, "contentId": cid}, timeout=30)
    try:
        data = res.json()
    except ValueError:  # 인증키 오류나 호출 한도 초과는 JSON이 아닌 응답으로 올 수 있다
        raise RuntimeError(f"JSON이 아닌 응답 (HTTP {res.status_code})") from None
    body = data.get("response", {})
    head = body.get("header", data)
    if head.get("resultCode") != "0000":
        raise RuntimeError(f"결과 코드 {head.get('resultCode')}: {head.get('resultMsg')}")
    items = body.get("body", {}).get("items") or {}
    item = items.get("item") or {}
    if isinstance(item, list):
        item = item[0] if item else {}
    return (item.get("overview") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="이번에 받을 장소 수 (시험할 때)")
    args = ap.parse_args()

    common = {"serviceKey": load_key(), "MobileOS": "ETC", "MobileApp": "TravelRecsys", "_type": "json"}
    done = load_done()
    finished = set(done["contentid"])
    todo = [c for c in target_ids() if c not in finished]
    print(f"대상 {len(todo) + len(finished)}곳, 이미 받음 {len(finished)}곳, 남음 {len(todo)}곳")
    if args.limit:
        todo = todo[:args.limit]

    rows = []
    try:
        for n, cid in enumerate(todo, 1):
            rows.append({"contentid": cid, "overview": fetch_overview(common, cid)})
            if n % SAVE_EVERY == 0:
                print(f"  {n} / {len(todo)}")
                done, rows = pd.concat([done, pd.DataFrame(rows)], ignore_index=True), []
                save(done)
            time.sleep(SLEEP)
    except requests.RequestException as e:  # 오류 문구의 URL에 인증키가 들어 있어서 종류만 출력한다
        print(f"네트워크 오류로 멈췄어요: {type(e).__name__}. 다시 실행하면 이어서 받아요.")
    except RuntimeError as e:
        print(f"API 오류로 멈췄어요: {e}. 원인을 해결한 뒤 다시 실행하면 이어서 받아요.")
    finally:
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        save(done)

    empty = (done["overview"].str.len() == 0).sum()
    print(f"저장 {len(done)}곳 (설명 없음 {empty}곳) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
