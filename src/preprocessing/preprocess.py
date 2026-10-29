"""
전처리 전체 실행: 원본(TL+VL, TourAPI, ASOS) -> data/processed/의 모델 입력 CSV.

실행 (프로젝트 최상위 폴더에서):
    python -m src.preprocessing.preprocess

순서
  1. 여행지 방문만 남기기 (config.TOURIST_TYPES)
  2. 좌표 채우기 (사진 GPS), 같은 장소 합치기 (places.py)
  3. 여행객 단위 train / val / test 분할 (config.SPLIT_RATIO)
  4. 장소 표: 고정 속성 + train으로만 계산한 통계 + TourAPI 매칭 + 비 온 날 방문 비율
  5. 평점: val·test 여행객은 여행 뒤쪽 HIDE_FRAC 장소를 가림 (known / hidden)
  6. k-fold 파일, 코스 추천용 데이터(순서·전이·집합·이동 거리), 방문별 날씨
  7. 특성 행렬 (features.py)
결과 숫자는 results/metrics/preprocess_summary.txt에도 남긴다.
"""
import numpy as np
import pandas as pd

from src.config import (HIDE_FRAC, METRIC_DIR, MIN_ITEM_TRAVELERS, N_FOLDS, PROCESSED_DIR, SEED,
                        SPLIT_RATIO, TOURIST_TYPES)
from src.preprocessing.external import attach_weather, match_tourapi
from src.preprocessing.features import build_item_features, build_user_features, build_users
from src.preprocessing.load_data import read, read_code
from src.preprocessing.places import assign_places, fill_xy_from_photos, norm_name, region_of

PLACE_TYPES = {int(k) for k in TOURIST_TYPES}
# 여행을 서로 다른 나들이로 끊는 방문 유형 (집, 친지집, 사무실, 숙소)
BREAK_TYPES = {21, 22, 23, 24}


# ---------------------------------------------------------------- 방문 / 분할

def build_visits(visit, travel, photo):
    """여행지 방문만 남기고, 시·도를 뽑고, 좌표를 채우고, 같은 장소를 합친다."""
    v = visit.merge(travel[["TRAVEL_ID", "TRAVELER_ID"]], on="TRAVEL_ID", how="left")
    v = v[v["VISIT_AREA_TYPE_CD"].isin(PLACE_TYPES)].copy()
    addr = v["ROAD_NM_ADDR"].fillna(v["LOTNO_ADDR"])
    v[["SIDO", "SIGUNGU"]] = pd.DataFrame(addr.map(region_of).tolist(), index=v.index)
    v = fill_xy_from_photos(v, photo)
    v = assign_places(v)
    return v.sort_values(["TRAVEL_ID", "VISIT_ORDER"]).reset_index(drop=True)


def split_travelers(traveler_ids):
    """여행객을 SPLIT_RATIO대로 무작위로 나눈다 (SEED 고정). TRAVELER_ID -> 'train' / 'val' / 'test'"""
    ids = np.array(sorted(traveler_ids))
    ids = ids[np.random.default_rng(SEED).permutation(len(ids))]
    cut = np.cumsum([round(SPLIT_RATIO[s] * len(ids)) for s in ("train", "val")])
    labels = np.where(np.arange(len(ids)) < cut[0], "train", np.where(np.arange(len(ids)) < cut[1], "val", "test"))
    return dict(zip(ids, labels))


# ---------------------------------------------------------------- 장소

def build_items(v, activity):
    """장소 표. 이름·유형·위치는 모든 방문에서, 평점·방문 수에 따른 통계는 train 여행객으로만 계산한다
    (val·test 정보가 특성에 섞이면 점수가 실제보다 좋게 나온다)."""
    g = v.groupby("ITEM_ID")
    items = pd.DataFrame({
        "ITEM_NM": g["ITEM_NM"].first(),
        "REP_POI_ID": g["REP_POI_ID"].first(),
        "TYPE_CD": g["VISIT_AREA_TYPE_CD"].agg(lambda s: s.mode().iloc[0]),
        # 한 방문의 좌표가 없으면 같은 장소의 다른 방문 좌표로 채워진다
        "X": g["X_COORD"].median(),
        "Y": g["Y_COORD"].median(),
        "SIDO": g["SIDO"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
        "SIGUNGU": g["SIGUNGU"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
    })
    items["TYPE_NM"] = items["TYPE_CD"].astype(str).map(TOURIST_TYPES)

    tr = v[v["SPLIT"] == "train"]
    gt = tr.groupby("ITEM_ID")
    stats = pd.DataFrame({
        "N_VISITS": gt.size(),
        "N_TRAVELERS": gt["TRAVELER_ID"].nunique(),
        "MEAN_DGSTFN": gt["DGSTFN"].mean().round(3),
        "MEAN_REVISIT": gt["REVISIT_INTENTION"].mean().round(3),
        "MEAN_RCMDTN": gt["RCMDTN_INTENTION"].mean().round(3),
        "MEDIAN_STAY_MIN": gt["RESIDENCE_TIME_MIN"].median(),
    })
    items = items.join(stats)
    items[["N_VISITS", "N_TRAVELERS"]] = items[["N_VISITS", "N_TRAVELERS"]].fillna(0).astype(int)

    # 활동 비율: train 방문 중 그 활동(ACTIVITY_TYPE_CD 1~7)이 있었던 비율
    a = activity.merge(tr[["TRAVEL_ID", "VISIT_AREA_ID", "ITEM_ID"]], on=["TRAVEL_ID", "VISIT_AREA_ID"])
    a = a[a["ACTIVITY_TYPE_CD"].between(1, 7)]
    prof = (a.drop_duplicates(["TRAVEL_ID", "VISIT_AREA_ID", "ACTIVITY_TYPE_CD"])
             .pivot_table(index="ITEM_ID", columns="ACTIVITY_TYPE_CD", values="VISIT_AREA_ID",
                          aggfunc="count", fill_value=0))
    prof = prof.div(items.loc[prof.index, "N_VISITS"], axis=0).clip(upper=1).round(3)
    prof.columns = [f"ACT_{c}" for c in prof.columns]
    items = items.join(prof)
    items[prof.columns] = items[prof.columns].fillna(0)

    # 인기도 0~1 (train 방문자 수에 log)
    items["POPULARITY"] = (np.log1p(items["N_TRAVELERS"]) / np.log1p(items["N_TRAVELERS"].max())).round(4)
    items["IN_CF"] = items["N_TRAVELERS"] >= MIN_ITEM_TRAVELERS
    return items.reset_index()


# ---------------------------------------------------------------- 평점

def build_ratings(v, cf_items):
    """여행객 x 장소 평점 (만족도 DGSTFN). 같은 곳을 여러 번 평가했으면 평균."""
    return (v[v["ITEM_ID"].isin(cf_items)]
            .groupby(["TRAVELER_ID", "ITEM_ID", "SPLIT"], as_index=False)
            .agg(RATING=("DGSTFN", "mean"),
                 REVISIT=("REVISIT_INTENTION", "mean"),
                 RCMDTN=("RCMDTN_INTENTION", "mean"),
                 FIRST_ORDER=("VISIT_ORDER", "min")))


def hide_last(r):
    """평점이 2개 이상인 여행객은 여행 뒤쪽 max(1, round(HIDE_FRAC*n))개를 가린다.

    val·test 여행객은 학습에 없어서 MF가 바로 예측할 수 없다. 앞부분(known)을 학습에 넣고
    뒷부분(hidden)을 맞히게 한다. 여행객 1명당 여행이 1번뿐이라 무작위로 가리면 나중에 간 곳을 보고
    먼저 간 곳을 맞히는 누수가 생겨서 방문 순서대로 가린다. 평점이 1개뿐이면 전부 known."""
    hidden = []
    for _, grp in r.groupby("TRAVELER_ID"):
        n = len(grp)
        if n >= 2:
            k = max(1, int(round(HIDE_FRAC * n)))
            hidden.extend(grp.sort_values("FIRST_ORDER").index[-k:])
    return r.drop(index=hidden), r.loc[hidden]


def build_holdout_table(v, split_of):
    """전체 여행객 k-fold 파일.
    user_folds       TRAVELER_ID -> FOLD (0 ~ N_FOLDS-1), SEED 고정
    holdout_by_order 여행객 x 장소마다 방문 순위, 여행 뒤쪽 HIDE_FRAC이면 HOLDOUT=1
    어느 fold를 평가에 쓰고 어떻게 평균낼지는 모델링 쪽에서 정한다."""
    first = v.groupby(["TRAVELER_ID", "ITEM_ID"], as_index=False)["VISIT_ORDER"].min()
    first = first.sort_values(["TRAVELER_ID", "VISIT_ORDER"])
    first["ORDER_RANK"] = first.groupby("TRAVELER_ID").cumcount() + 1
    n = first.groupby("TRAVELER_ID")["ITEM_ID"].transform("size")
    k = np.maximum(1, np.round(HIDE_FRAC * n)).astype(int)
    first["HOLDOUT"] = ((n >= 2) & (first["ORDER_RANK"] > n - k)).astype(int)
    first = first.rename(columns={"VISIT_ORDER": "FIRST_VISIT_ORDER"})

    users = np.array(sorted(split_of))
    perm = np.random.default_rng(SEED).permutation(len(users))
    folds = pd.DataFrame({"TRAVELER_ID": users[perm], "FOLD": np.arange(len(users)) % N_FOLDS})
    return first, folds.sort_values("TRAVELER_ID").reset_index(drop=True)


# ---------------------------------------------------------------- 코스 추천용

def build_sequences(v):
    """여행별 방문 순서 (마르코프 체인 입력). 같은 장소 연속 방문은 한 번으로 합친다."""
    rows = []
    for tid, grp in v.groupby("TRAVEL_ID", sort=False):
        seq, tseq = [], []
        for it, ty in zip(grp["ITEM_ID"], grp["VISIT_AREA_TYPE_CD"]):
            if not seq or seq[-1] != it:
                seq.append(it)
                tseq.append(ty)
        rows.append({"TRAVEL_ID": tid, "TRAVELER_ID": grp["TRAVELER_ID"].iloc[0],
                     "SPLIT": grp["SPLIT"].iloc[0], "LENGTH": len(seq),
                     "ITEM_SEQ": "|".join(seq), "TYPE_SEQ": "|".join(map(str, tseq))})
    return pd.DataFrame(rows)


def build_transitions(seqs):
    """장소 -> 다음 장소 횟수와 확률 (train 여행만). 장소 단위는 대부분 1번씩이라 유형 단위도 같이 만든다."""
    rows = []
    for level, col in (("item", "ITEM_SEQ"), ("type", "TYPE_SEQ")):
        for s in seqs.loc[seqs["SPLIT"] == "train", col]:
            tok = s.split("|")
            rows += [(level, a, b) for a, b in zip(tok, tok[1:])]
    tr = pd.DataFrame(rows, columns=["LEVEL", "FROM", "TO"])
    tr = tr.groupby(["LEVEL", "FROM", "TO"]).size().rename("COUNT").reset_index()
    tr["PROB"] = (tr["COUNT"] / tr.groupby(["LEVEL", "FROM"])["COUNT"].transform("sum")).round(4)
    return tr


def build_transactions(seqs):
    """여행별 장소 집합 / 유형 집합 (Apriori 입력)."""
    tx = seqs[["TRAVEL_ID", "SPLIT", "LENGTH"]].copy()
    tx["ITEMS"] = seqs["ITEM_SEQ"].str.split("|").map(lambda s: "|".join(sorted(set(s))))
    tx["TYPES"] = seqs["TYPE_SEQ"].str.split("|").map(lambda s: "|".join(sorted(set(s), key=int)))
    return tx


def build_place_legs(visit_all, v, move, items, split_of):
    """여행지 A 다음에 간 여행지 B, 거리, 이동수단.

    이동내역에는 식당·역·집으로 가는 구간도 다 들어 있어서 '여행지 -> 다음 여행지'로 다시 묶는다.
      - 중간에 들른 식당·카페·역은 건너뛰고 유형을 VIA_TYPES에 남긴다
      - 중간에 집·숙소·사무실·친지집을 거치면 다른 나들이(대부분 다른 날)라 넣지 않는다
    이동시간(분)은 넣지 않는다. 원본의 도착시각과 체류시간이 30분 단위로만 기록돼 있어서
    500m 도보와 3km 차량 이동이 둘 다 30분으로 찍힌다. 이동 비용은 좌표로 계산한 DIST_KM을 쓴다.
    Returns
      place_legs   여행마다 관측된 A -> B 한 줄씩
      place_pairs  (A, B) 쌍별 집계, train 여행만: 횟수, 거리, 바로 이동한 비율, 주 이동수단
    """
    cb = read_code()
    cb = cb[cb["cd_a"] == "MOV"]
    mode_nm = dict(zip(cb["cd_b"].astype(int), cb["cd_nm"]))
    leg_mode = {(r.TRAVEL_ID, int(r.END_VISIT_AREA_ID)): r.MVMN_CD_1
                for r in move[move["END_VISIT_AREA_ID"].notna()].itertuples()}
    place = v.set_index(["TRAVEL_ID", "VISIT_AREA_ID"])["ITEM_ID"]
    place = place[~place.index.duplicated()].to_dict()
    traveler = v.drop_duplicates("TRAVEL_ID").set_index("TRAVEL_ID")["TRAVELER_ID"]
    xy = items.set_index("ITEM_ID")[["X", "Y"]]

    rows = []
    for tid, grp in visit_all.sort_values(["TRAVEL_ID", "VISIT_ORDER"]).groupby("TRAVEL_ID", sort=False):
        cur, via, modes, broken = None, [], [], False
        for r in grp.itertuples():
            item = place.get((tid, r.VISIT_AREA_ID))
            if cur is not None:
                mode = leg_mode.get((tid, r.VISIT_AREA_ID), np.nan)
                if pd.notna(mode):
                    modes.append(int(mode))
            if item is not None:
                if cur is not None and not broken and item != cur:
                    rows.append({"TRAVEL_ID": tid, "FROM_ITEM": cur, "TO_ITEM": item,
                                 "N_VIA": len(via), "VIA_TYPES": "|".join(map(str, via)),
                                 "MVMN_CD": max(set(modes), key=modes.count) if modes else np.nan})
                cur, via, modes, broken = item, [], [], False
            elif cur is not None:
                if r.VISIT_AREA_TYPE_CD in BREAK_TYPES:
                    broken = True
                else:
                    via.append(int(r.VISIT_AREA_TYPE_CD))
    pl = pd.DataFrame(rows)
    pl.insert(1, "SPLIT", pl["TRAVEL_ID"].map(traveler).map(split_of))
    a = xy.reindex(pl["FROM_ITEM"]).to_numpy()
    b = xy.reindex(pl["TO_ITEM"]).to_numpy()
    pl["DIST_KM"] = (np.hypot((a[:, 0] - b[:, 0]) * 88, (a[:, 1] - b[:, 1]) * 111)).round(2)
    pl["MVMN_NM"] = pl["MVMN_CD"].map(mode_nm)

    g = pl[pl["SPLIT"] == "train"].groupby(["FROM_ITEM", "TO_ITEM"])
    pair = pd.DataFrame({
        "N_TRIPS": g.size(),
        "DIST_KM": g["DIST_KM"].first(),
        "DIRECT_SHARE": g["N_VIA"].agg(lambda s: (s == 0).mean()).round(3),
        "MAIN_MVMN_CD": g["MVMN_CD"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan),
    }).reset_index()
    pair["MAIN_MVMN_NM"] = pair["MAIN_MVMN_CD"].map(mode_nm)
    return pl, pair


# ---------------------------------------------------------------- 실행

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    def say(msg):
        print(msg)
        log.append(msg)

    visit_all, travel = read("visit"), read("travel")
    traveller, photo, activity, move = read("traveller"), read("photo"), read("activity"), read("move")
    companion = read("companion")
    say(f"원본 방문 기록: {len(visit_all)} (TL {sum(visit_all.SOURCE == 'TL')}, VL {sum(visit_all.SOURCE == 'VL')}), "
        f"여행객 {traveller.TRAVELER_ID.nunique()}")

    v = build_visits(visit_all, travel, photo)
    say(f"여행지 방문 (유형 {sorted(PLACE_TYPES)}): {len(v)}, 여행객 {v.TRAVELER_ID.nunique()}")
    src = v["XY_SOURCE"].value_counts()
    say(f"  좌표: 방문 기록 {src.get('visit', 0)}, 사진 GPS로 채움 {src.get('photo', 0)}, 없음 {src.get('none', 0)}")
    n_before = v["POI_ID"].fillna("N_" + v["VISIT_AREA_NM"].map(norm_name)).nunique()
    say(f"  장소 수: 병합 전(POI_ID 또는 이름) {n_before} -> 병합 후 {v.ITEM_ID.nunique()}")

    split_of = split_travelers(traveller["TRAVELER_ID"].unique())
    v["SPLIT"] = v["TRAVELER_ID"].map(split_of)
    n_split = pd.Series(split_of).value_counts()
    say("여행객 분할: " + ", ".join(f"{s} {n_split.get(s, 0)}" for s in SPLIT_RATIO))

    items = build_items(v, activity)
    tour = match_tourapi(items, v)
    items = items.merge(tour, on="ITEM_ID", how="left")
    weather = attach_weather(v, items)
    # 날씨 보정용: train 방문 중 비 온 날 방문의 비율
    rainy = weather.loc[v["SPLIT"].to_numpy() == "train"].groupby("ITEM_ID")["RAINY"].mean().round(3)
    items["RAINY_SHARE"] = items["ITEM_ID"].map(rainy)
    say(f"장소: {len(items)}곳, MF용(train 방문자 {MIN_ITEM_TRAVELERS}명 이상) {items.IN_CF.sum()}곳, "
        f"좌표 없음 {items.X.isna().sum()}곳")
    say(f"  TourAPI 매칭: {items.TOUR_ID.notna().sum()}곳 ({items.TOUR_ID.notna().mean():.0%}), "
        f"MF용 장소 중 {items.loc[items.IN_CF, 'TOUR_ID'].notna().mean():.0%}")
    say(f"  날씨: 방문 {weather.TA_AVG.notna().sum()} / {len(weather)}건에 붙음, 비 온 날 방문 {weather.RAINY.mean():.0%}")

    users = build_users(traveller, travel, companion, split_of)
    say("  동반자 그룹: " + ", ".join(f"{k} {n}" for k, n in users.COMPANION_GROUP.value_counts().items()))

    r = build_ratings(v, set(items.loc[items.IN_CF, "ITEM_ID"]))
    train = r[r.SPLIT == "train"]
    parts = {"train": train}
    for s in ("val", "test"):
        sub = r[(r.SPLIT == s) & r.ITEM_ID.isin(train.ITEM_ID)]
        parts[f"{s}_known"], parts[f"{s}_hidden"] = hide_last(sub)
    n_u, n_i = train.TRAVELER_ID.nunique(), train.ITEM_ID.nunique()
    say(f"train 평점: {len(train)}, 여행객 {n_u}, 장소 {n_i}, 밀도 {len(train) / (n_u * n_i):.4f}")
    for s in ("val", "test"):
        h = parts[f"{s}_hidden"]
        say(f"  {s}: known {len(parts[f'{s}_known'])}, hidden {len(h)} (채점 여행객 {h.TRAVELER_ID.nunique()}명)")

    holdout, folds = build_holdout_table(v, split_of)
    say(f"k-fold: 여행객 {len(folds)}명 {N_FOLDS}개 fold, 여행객 x 장소 {len(holdout)}쌍 중 holdout {holdout.HOLDOUT.sum()}")

    seqs = build_sequences(v)
    trans = build_transitions(seqs)
    tx = build_transactions(seqs)
    place_legs, place_pairs = build_place_legs(visit_all, v, move, items, split_of)
    say(f"여행 순서: {len(seqs)}개, 길이 2 이상 {(seqs.LENGTH >= 2).sum()}; 전이(train) 장소 단위 "
        f"{(trans.LEVEL == 'item').sum()}쌍, 유형 단위 {(trans.LEVEL == 'type').sum()}쌍")
    say(f"여행지 간 이동: {len(place_legs)}구간 (바로 이동 {(place_legs.N_VIA == 0).mean():.0%}, "
        f"거리 중앙값 {place_legs.DIST_KM.median():.1f}km), 장소 쌍(train) {len(place_pairs)}")

    item_f = build_item_features(items, v, users)
    user_f = build_user_features(users)
    say(f"특성 행렬: 장소 {item_f.shape[0]} x {item_f.shape[1] - 1}, 여행객 {user_f.shape[0]} x {user_f.shape[1] - 1}")

    # MF 행렬 인덱스 (train 여행객 먼저, 그다음 val·test known)
    cf_items = items.loc[items.IN_CF, "ITEM_ID"].sort_values().reset_index(drop=True)
    cf_users = pd.concat([parts["train"], parts["val_known"], parts["test_known"]])["TRAVELER_ID"]
    cf_users = cf_users.drop_duplicates().reset_index(drop=True)

    v_cols = ["TRAVELER_ID", "TRAVEL_ID", "SPLIT", "SOURCE", "VISIT_AREA_ID", "VISIT_ORDER", "ITEM_ID", "ITEM_NM",
              "VISIT_AREA_NM", "POI_ID", "REP_POI_ID", "VISIT_AREA_TYPE_CD", "VISIT_START_YMD", "SIDO", "SIGUNGU",
              "X_COORD", "Y_COORD", "XY_SOURCE", "RESIDENCE_TIME_MIN", "DGSTFN", "REVISIT_INTENTION",
              "RCMDTN_INTENTION", "REVISIT_YN"]
    out = {
        "visits": v[v_cols], "items": items, "users": users,
        "item_features": item_f, "user_features": user_f,
        "item_index": pd.DataFrame({"ITEM_IDX": cf_items.index, "ITEM_ID": cf_items}),
        "user_index": pd.DataFrame({"USER_IDX": cf_users.index, "TRAVELER_ID": cf_users}),
        **{f"ratings_{k}": df for k, df in parts.items()},
        "user_folds": folds, "holdout_by_order": holdout,
        "sequences": seqs, "transitions": trans, "transactions": tx,
        "place_legs": place_legs, "place_pairs": place_pairs, "weather": weather,
    }
    for name, df in out.items():
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    (METRIC_DIR / "preprocess_summary.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    say(f"저장: {PROCESSED_DIR} ({len(out)}개 파일)")


if __name__ == "__main__":
    main()
