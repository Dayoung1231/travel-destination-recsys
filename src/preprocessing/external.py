"""
여행 로그 외의 두 데이터(TourAPI 장소 목록, ASOS 날씨 일자료)를 장소·방문에 붙인다.

- TourAPI: 우리 장소를 TourAPI 관광지와 이름 + 거리로 매칭한다 (장소 병합과 같은 기준).
  매칭되면 공식 분류(lclsSystm)와 contentid를 가져오고, 매칭 여부는 '알려진 관광지' 신호로 쓴다.
- ASOS: 방문 날짜 x 가장 가까운 관측소의 기온·강수량을 붙인다.
"""
import numpy as np
import pandas as pd

from src.config import ASOS_DIR, TOURAPI_DIR
from src.preprocessing.places import dist_m, norm_name, region_of, similar

TOUR_NEAR_M = 300    # 이름이 비슷하면 이 거리 안에서 매칭
TOUR_PART_M = 1000   # 이름이 같거나 한쪽이 다른 쪽을 품으면 이 거리까지 (테마파크·공원은 넓다)
TOUR_EVENT_TYPE = "15"  # TourAPI 관광 타입: 축제공연행사
FESTIVAL_TYPE = 8       # 우리 방문지 유형: 지역 축제

# ASOS 관측소 위치 (기상청 기상자료개방포털 > ASOS 관측지점정보, 2023년 기준 위치. 경도·위도)
ASOS_STATIONS = {
    98: ("동두천", 127.0607, 37.9019),
    99: ("파주", 126.7665, 37.8859),
    102: ("백령도", 124.7124, 37.9740),
    108: ("서울", 126.9658, 37.5714),
    112: ("인천", 126.6249, 37.4777),
    119: ("수원", 126.9830, 37.2575),
    201: ("강화", 126.4463, 37.7074),
    202: ("양평", 127.4945, 37.4886),
    203: ("이천", 127.4842, 37.2640),
}
# 좌표를 모를 때 시·도 대표 관측소
SIDO_STATION = {"서울": 108, "인천": 112, "경기": 119}
RAINY_MM = 1.0  # 일강수량 1mm 이상이면 비 온 날


# ---------------------------------------------------------------- TourAPI

def load_tourapi():
    t = pd.read_csv(TOURAPI_DIR / "area_list.csv", dtype={"contentid": str, "contenttypeid": str})
    t["NORM"] = t["title"].map(norm_name)
    t["SIDO"] = t["addr1"].map(lambda a: region_of(a)[0])
    return t


def name_rank(rep, names, title, near):
    """TourAPI 이름 title이 우리 장소 이름과 얼마나 잘 맞는지. 작을수록 좋고, 맞지 않으면 None.
    rep: 대표 이름, names: 여행객이 쓴 이름들, near: TOUR_NEAR_M 안에 있는지 (모두 norm_name 결과).
      0 대표 이름과 같음                  '경복궁' / '경복궁'
      1 여행객이 쓴 다른 이름과 같음
      2 TourAPI 쪽이 상위 장소            '석촌호수 동호' / '석촌호수'
      3 TourAPI 쪽이 세부 시설·행사        '서울랜드' / '서울랜드 눈썰매장'
      4 이름이 비슷함 (near일 때만)        '서울로 717' / '서울로 7017'
    """
    if title == rep:
        return 0
    if title in names:
        return 1
    if len(title) >= 4 and any(len(n) >= 4 and title in n for n in names):
        return 2
    if len(title) >= 4 and any(len(n) >= 4 and n in title for n in names):
        return 3
    if near and any(similar(n, title) for n in names):
        return 4
    return None


def match_tourapi(items, v):
    """장소마다 가장 잘 맞는 TourAPI 관광지를 찾는다.

    장소 하나에 여행객이 쓴 이름이 여러 개라서(병합 결과), 그 이름들 중 하나라도 맞으면 된다.
      - TOUR_NEAR_M 안이고 이름이 비슷함 (places.similar)
      - TOUR_PART_M 안이고 이름이 같거나 한쪽이 다른 쪽을 품음 (4글자 이상)
      - 장소 좌표가 없으면: 이름이 같고 시·도가 같음
    후보가 여럿이면 이름이 더 잘 맞는 것(name_rank), 같으면 가장 가까운 것.
    거리만 보면 '경복궁'이 같은 좌표의 '경복궁 생과방'에 매칭된다.
    행사(TOUR_EVENT_TYPE)는 이름이 같을 때와 우리 장소가 지역 축제일 때만 후보로 둔다
    ('시흥갯골생태공원' / '시흥갯골축제'처럼 상설 장소에 행사가 붙지 않게).
    """
    t = load_tourapi()
    tx, ty = t["mapx"].to_numpy(), t["mapy"].to_numpy()
    names = v.groupby("ITEM_ID")["VISIT_AREA_NM"].agg(lambda s: {norm_name(x) for x in s})
    by_name = t.groupby("NORM").indices
    is_event = (t["contenttypeid"] == TOUR_EVENT_TYPE).to_numpy()

    rows = []
    for it in items.itertuples():
        my = names[it.ITEM_ID]
        rep = norm_name(it.ITEM_NM)
        best, best_d, best_key = None, np.inf, (np.inf, np.inf)
        if pd.notna(it.X):
            d = dist_m(tx, ty, it.X, it.Y)
            for k in np.flatnonzero(d <= TOUR_PART_M):
                rank = name_rank(rep, my, t.at[k, "NORM"], d[k] <= TOUR_NEAR_M)
                if rank is None or (is_event[k] and rank > 1 and it.TYPE_CD != FESTIVAL_TYPE):
                    continue
                if (rank, d[k]) < best_key:
                    best, best_d, best_key = k, d[k], (rank, d[k])
        else:
            for n in my:
                for k in by_name.get(n, []):
                    if t.at[k, "SIDO"] == it.SIDO:
                        best, best_d = k, np.nan
        if best is not None:
            r = t.loc[best]
            rows.append({"ITEM_ID": it.ITEM_ID, "TOUR_ID": r["contentid"], "TOUR_TITLE": r["title"],
                         "TOUR_TYPE": r["contenttypeid"], "TOUR_CAT1": r["lclsSystm1"],
                         "TOUR_CAT2": r["lclsSystm2"], "TOUR_CAT3": r["lclsSystm3"],
                         "TOUR_DIST_M": round(best_d) if pd.notna(best_d) else np.nan})
    cols = ["ITEM_ID", "TOUR_ID", "TOUR_TITLE", "TOUR_TYPE", "TOUR_CAT1", "TOUR_CAT2", "TOUR_CAT3", "TOUR_DIST_M"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- ASOS

def load_asos():
    a = pd.read_csv(ASOS_DIR / "asos_daily.csv", encoding="cp949")
    a.columns = ["STN", "STN_NM", "DATE", "TA_AVG", "TA_MIN", "TA_MAX", "RN_DAY"]
    # ASOS 일자료에서 일강수량이 비어 있으면 비가 오지 않은 날이다
    a["RN_DAY"] = a["RN_DAY"].fillna(0.0)
    a["DATE"] = pd.to_datetime(a["DATE"]).dt.strftime("%Y-%m-%d")
    return a


def nearest_station(x, y):
    stn = list(ASOS_STATIONS)
    sx = np.array([ASOS_STATIONS[s][1] for s in stn])
    sy = np.array([ASOS_STATIONS[s][2] for s in stn])
    return stn[int(np.argmin(dist_m(sx, sy, x, y)))]


def attach_weather(v, items):
    """방문마다 날짜 x 가장 가까운 관측소의 날씨를 붙인다.
    장소 좌표가 없으면 같은 여행의 다른 방문 좌표 중앙값, 그것도 없으면 시·도 대표 관측소."""
    xy = items.set_index("ITEM_ID")[["X", "Y"]]
    w = v[["TRAVEL_ID", "VISIT_AREA_ID", "ITEM_ID", "VISIT_START_YMD", "SIDO"]].copy()
    w["X"] = w["ITEM_ID"].map(xy["X"])
    w["Y"] = w["ITEM_ID"].map(xy["Y"])
    w["X"] = w["X"].fillna(w.groupby("TRAVEL_ID")["X"].transform("median"))
    w["Y"] = w["Y"].fillna(w.groupby("TRAVEL_ID")["Y"].transform("median"))
    w["STN"] = [nearest_station(x, y) if pd.notna(x) else SIDO_STATION.get(s, 108)
                for x, y, s in zip(w["X"], w["Y"], w["SIDO"])]
    w["DATE"] = pd.to_datetime(w["VISIT_START_YMD"]).dt.strftime("%Y-%m-%d")
    w = w.merge(load_asos().drop(columns="STN_NM"), on=["STN", "DATE"], how="left")
    w["RAINY"] = (w["RN_DAY"] >= RAINY_MM).astype(int)
    return w[["TRAVEL_ID", "VISIT_AREA_ID", "ITEM_ID", "DATE", "STN", "TA_AVG", "TA_MIN", "TA_MAX", "RN_DAY", "RAINY"]]
