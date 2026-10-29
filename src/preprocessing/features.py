"""
모델 입력용 숫자 특성 행렬 (CBF, 사용자 프로필).

모든 값은 0~1. 특성마다 범위가 다르면 큰 숫자가 유사도를 좌우해서 스케일을 맞춘다.
통계로 만든 특성(활동 비율, 인기도, 동반자 비율)은 train 여행객으로만 계산한 값이다.
"""
import numpy as np
import pandas as pd

from src.config import COMPANION_GROUPS, REGIONS, TOURIST_TYPES

# TRAVEL_STATUS_ACCOMPANY -> 동반자 그룹.
# TCR 코드: 1 배우자, 2 자녀, 3 부모, 4 조부모, 5 형제/자매, 6 친인척, 7 친구, 8 연인, 9 동료, 10 모임, 11 기타
FAMILY_ADULTS = {3, 4, 5, 6}

def companion_group(rels, n):
    """한 여행의 동반자 관계 코드 집합(rels)과 동반자 수(n) -> 동반자 그룹. 위에서부터 먼저 맞는 그룹."""
    if not rels:
        return "solo"
    if 2 in rels:
        return "family_kids"
    if n == 1 and rels <= {1, 8}:   # 연인 또는 배우자와 단둘이
        return "couple"
    if rels & FAMILY_ADULTS:
        return "family_adults"
    return "friends"

# TourAPI 분류체계 대분류(lclsSystm1) 코드
TOUR_CATS = ("AC", "EV", "EX", "HS", "LS", "VE")


def build_item_features(items, v, users):
    """장소 특성 행렬 (장소 1행). TYPE_<코드>, SIDO_<시도>, ACT_<활동>, STAY, POPULARITY,
    COMP_<동반자 그룹>, IN_TOURAPI, TCAT_<TourAPI 대분류>."""
    it = items.set_index("ITEM_ID")
    f = pd.DataFrame(index=it.index)
    for cd in TOURIST_TYPES:
        f[f"TYPE_{cd}"] = (it["TYPE_CD"] == int(cd)).astype(int)
    for sd in REGIONS:
        f[f"SIDO_{sd}"] = (it["SIDO"] == sd).astype(int)
    f["SIDO_other"] = (~it["SIDO"].isin(REGIONS) & it["SIDO"].notna()).astype(int)
    for c in [c for c in it.columns if c.startswith("ACT_")]:
        f[c] = it[c]
    # 테마파크처럼 오래 머무는 곳과 잠깐 들르는 곳. log를 씌워 극단값 영향을 줄이고, 모르면 0
    stay = np.log1p(it["MEDIAN_STAY_MIN"].fillna(0))
    f["STAY"] = (stay / stay.max()).round(4)
    f["POPULARITY"] = it["POPULARITY"]

    # 그 장소에 누가 가나: train 방문자 중 동반자 그룹별 비율 (동반자 보정용)
    tr = v[v["SPLIT"] == "train"][["TRAVELER_ID", "ITEM_ID"]].drop_duplicates()
    tr = tr.merge(users[["TRAVELER_ID", "COMPANION_GROUP"]], on="TRAVELER_ID")
    comp = (pd.crosstab(tr["ITEM_ID"], tr["COMPANION_GROUP"], normalize="index")
          .reindex(columns=COMPANION_GROUPS, fill_value=0).round(4))
    comp.columns = [f"COMP_{c}" for c in comp.columns]
    f = f.join(comp)
    f[comp.columns] = f[comp.columns].fillna(0)

    # TourAPI에 있는 관광지인지, 있으면 공식 대분류
    f["IN_TOURAPI"] = it["TOUR_ID"].notna().astype(int)
    for c in TOUR_CATS:
        f[f"TCAT_{c}"] = (it["TOUR_CAT1"] == c).astype(int)
    return f.reset_index()


def build_users(traveller, travel, companion, split_of):
    """여행객 표. 동반자 그룹, 여행 일수, 여행 미션·동기 one-hot을 붙인다."""
    u = traveller.drop(columns="SOURCE").merge(travel.drop(columns="SOURCE"), on="TRAVELER_ID", how="left")
    comp = (companion.groupby("TRAVEL_ID")
            .agg(RELS=("REL_CD", lambda s: frozenset(s.dropna().astype(int))),
                 N_COMP=("REL_CD", "size")))          # N_COMP는 행 수 (len(RELS) 아님)
    u = u.merge(comp, on="TRAVEL_ID", how="left")
    u["RELS"] = u["RELS"].apply(lambda x: x if isinstance(x, frozenset) else frozenset())
    u["N_COMP"] = u["N_COMP"].fillna(0).astype(int)
    u["COMPANION_GROUP"] = [companion_group(r, n) for r, n in zip(u["RELS"], u["N_COMP"])]
    u["SPLIT"] = u["TRAVELER_ID"].map(split_of)
    u["GENDER"] = u["GENDER"].map({"남": "M", "여": "F"})
    u["TRAVEL_DAYS"] = (pd.to_datetime(u["TRAVEL_END_YMD"]) - pd.to_datetime(u["TRAVEL_START_YMD"])).dt.days + 1

    # 여행 미션 (복수 선택, '3;4;11') -> MIS_<코드>
    mis = u["TRAVEL_MISSION_CHECK"].fillna("").astype(str).str.get_dummies(sep=";")
    mis.columns = [f"MIS_{c}" for c in mis.columns]
    # 여행 동기 (최대 3개) -> MOT_<코드>
    mot = pd.get_dummies(u[["TRAVEL_MOTIVE_1", "TRAVEL_MOTIVE_2", "TRAVEL_MOTIVE_3"]]
                         .stack().dropna().astype(int)).groupby(level=0).max().astype(int)
    mot.columns = [f"MOT_{c}" for c in mot.columns]

    keep = (["TRAVELER_ID", "TRAVEL_ID", "SPLIT", "GENDER", "AGE_GRP", "COMPANION_GROUP",
             "TRAVEL_COMPANIONS_NUM", "TRAVEL_DAYS", "TRAVEL_START_YMD", "TRAVEL_END_YMD"]
            + [f"TRAVEL_STYL_{i}" for i in range(1, 9)])
    return pd.concat([u[keep], mis, mot.reindex(u.index, fill_value=0)], axis=1)


def build_user_features(users):
    """사용자 특성 행렬 (여행객 1행)."""
    u = users.set_index("TRAVELER_ID")
    f = pd.DataFrame(index=u.index)
    f["GENDER_F"] = (u["GENDER"] == "F").astype(int)
    f["AGE"] = ((u["AGE_GRP"] - 20) / 40).clip(0, 1).round(3)  # 20대 -> 0, 60대 -> 1
    for g in COMPANION_GROUPS:                       # 기존: sorted(set(COMPANION_GROUP.values()))
        f[f"COMP_{g}"] = (u["COMPANION_GROUP"] == g).astype(int)
    # 여행 스타일은 두 극 사이의 1~7 척도 (자연 1 <-> 도시 7 등)
    for i in range(1, 9):
        f[f"STYL_{i}"] = ((u[f"TRAVEL_STYL_{i}"] - 1) / 6).round(3)
    f["DAYS"] = ((u["TRAVEL_DAYS"] - 1) / 3).clip(0, 1).round(3)
    for c in sorted([c for c in u.columns if c.startswith(("MIS_", "MOT_"))], key=lambda c: (c[:3], int(c[4:]))):
        f[c] = u[c]
    return f.reset_index()
