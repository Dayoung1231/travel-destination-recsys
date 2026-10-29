"""
AI Hub 여행 로그 원본(TL, VL)을 읽어 하나로 합친다.

압축 프로그램에 따라 파일명의 한글 부분이 깨질 수 있어서, 파일은 영문 앞부분
(tn_visit_area_info 등)으로 찾는다. 폴더 위치는 src/config.py의 TL_DIR, VL_DIR.
"""
import pandas as pd

from src.config import TL_DIR, VL_DIR

# 표 이름 -> 파일명 앞부분
TABLES = {
    "visit": "tn_visit_area_info_",       # 방문지정보
    "travel": "tn_travel_",               # 여행
    "traveller": "tn_traveller_master_",  # 여행객 Master
    "photo": "tn_tour_photo_",            # 관광사진
    "activity": "tn_activity_his_",       # 활동내역
    "move": "tn_move_his_",               # 이동내역
    "codeb": "tc_codeb_",                 # 코드B
    "companion": "tn_companion_info_",   # 동반자정보
}


def find_file(folder, table):
    """folder 안에서 table에 해당하는 CSV 파일 하나를 찾는다."""
    prefix = TABLES[table]
    found = sorted(folder.glob(f"{prefix}*.csv"))
    if not found:
        raise SystemExit(f"{folder}에 {prefix}*.csv가 없어요. README 2-1대로 TL_csv.zip, VL_csv.zip을 풀어 넣어 주세요.")
    return found[0]


def read(table):
    """TL과 VL의 같은 표를 합쳐 읽는다. SOURCE 열에 원래 폴더(TL / VL)를 남긴다."""
    parts = []
    for source, folder in (("TL", TL_DIR), ("VL", VL_DIR)):
        df = pd.read_csv(find_file(folder, table), encoding="utf-8-sig")
        df["SOURCE"] = source
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def read_code(table="codeb"):
    """코드표는 TL과 VL이 같아서 TL 것만 읽는다."""
    return pd.read_csv(find_file(TL_DIR, table), encoding="utf-8-sig")
