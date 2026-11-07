"""
추천 모델 공용 평가. CBF, MF 등 모든 모델이 같은 함수로 채점받는다.

터미널에서 사용자가 입력하는 두 상황을 각각 흉내 낸다.
  - cold: 설문만 입력 (가 본 곳 없음). 그 사람이 간 모든 장소가 정답.
  - warm: 설문 + 가 본 곳 입력. 여행 앞쪽(known)을 가 본 곳으로 주고, 뒤쪽(hidden)을 맞히는지 본다.
          known / hidden은 holdout_by_order의 HOLDOUT 열 (평점 파일과 달리 MF용이 아닌 장소도 포함).

후보 장소(catalog)는 train 여행객이 한 번이라도 간 수도권 장소로 제한한다.
  - 전체 장소에는 val·test 여행객만 간 곳이 섞여 있어서, 그대로 두면 'train 방문자 0명'이라는
    특징만으로 정답을 골라내는 누수가 생긴다.
  - 여행 로그에 수도권 밖 방문(강원 등)도 섞여 있지만 우리 시스템은 수도권 추천이라 뺀다.
  - 지역 축제는 사용자의 여행 날에 열린다는 보장이 없어서 뺀다. 학습 데이터에는 취향 정보로 남긴다.

    from src.modeling.evaluate import evaluate
    mean, per_user = evaluate(model.score, "val", "warm")

여러 설정을 빠르게 비교할 때는 여행객 x 후보 점수 행렬을 한 번에 채점하는 evaluate_matrix를 쓴다
(evaluate와 결과가 같고, 행 순서는 split_matrix의 여행객 순서).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REGIONS, SEED, TOP_K
from src.preprocessing.external import FESTIVAL_TYPE

MODES = ("cold", "warm")


@lru_cache(maxsize=None)
def load_table(name):
    """data/processed/<name>.csv. 여러 번 읽지 않게 캐시한다 (받은 표를 직접 수정하지 말 것).
    TOUR_CAT1의 NA는 자연관광 코드라서 빈칸만 결측값으로 읽는다."""
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", keep_default_na=False, na_values=[""])


def catalog():
    """추천 후보 장소 ID (정렬): train 여행객이 한 번이라도 간 수도권 장소 중 지역 축제가 아닌 곳.
    주소가 없어 시·도를 모르는 장소는 수도권 여행 로그라 남기고, 수도권 밖으로 확인된 곳만 뺀다."""
    items = load_table("items")
    ok = ((items["N_TRAVELERS"] >= 1) & (items["SIDO"].isin(REGIONS) | items["SIDO"].isna())
          & (items["TYPE_CD"] != FESTIVAL_TYPE))
    return np.sort(items.loc[ok, "ITEM_ID"].to_numpy())


@lru_cache(maxsize=None)
def load_split(split, mode="warm"):
    """split('val' / 'test') 여행객의 가 본 곳(known), 정답(hidden), 후보 장소.
    cold면 known은 모두 빈 목록이고 간 장소 전부가 정답이다.
    정답 중 후보에 있는 장소가 하나도 없는 여행객은 채점할 수 없어서 뺀다.
    Returns known {TRAVELER_ID: [ITEM_ID]}, hidden {TRAVELER_ID: [ITEM_ID]}, candidates (ITEM_ID 배열)"""
    if mode not in MODES:
        raise ValueError(f"mode는 {MODES} 중 하나: {mode}")
    candidates = catalog()
    users = load_table("users")
    h = load_table("holdout_by_order")
    h = h[h["TRAVELER_ID"].isin(users.loc[users["SPLIT"] == split, "TRAVELER_ID"])]
    if mode == "cold":
        hid = h[h["ITEM_ID"].isin(candidates)]
        hidden = hid.groupby("TRAVELER_ID")["ITEM_ID"].agg(list).to_dict()
        return {t: [] for t in hidden}, hidden, candidates
    hid = h[(h["HOLDOUT"] == 1) & h["ITEM_ID"].isin(candidates)]
    hidden = hid.groupby("TRAVELER_ID")["ITEM_ID"].agg(list).to_dict()
    known = h[h["HOLDOUT"] == 0].groupby("TRAVELER_ID")["ITEM_ID"].agg(list).to_dict()
    return {t: known.get(t, []) for t in hidden}, hidden, candidates


def evaluate(score_fn, split="val", mode="warm", k_list=TOP_K):
    """score_fn(traveler_id, known_items, candidates) -> 후보마다 점수 (클수록 먼저 추천).

    known 장소는 후보에서 빼고 상위 K개를 뽑아 여행객마다 Recall@K, NDCG@K, Hit@K를 계산한다.
    점수가 같은 장소는 SEED로 고정한 무작위 순서로 정한다 (ID 순으로 두면 앞 번호 장소가 유리하다).
    Returns 여행객 평균 (Series), 여행객별 점수 (DataFrame, 신뢰구간·t-test용)"""
    known, hidden, cand = load_split(split, mode)
    pos = {it: i for i, it in enumerate(cand)}
    tiebreak = np.random.default_rng(SEED).random(len(cand))
    discount = 1 / np.log2(np.arange(2, max(k_list) + 2))
    rows = []
    for tid, truth in hidden.items():
        s = np.array(score_fn(tid, known[tid], cand), dtype=float)
        s[[pos[i] for i in known[tid] if i in pos]] = -np.inf
        top = cand[np.lexsort((tiebreak, -s))[:max(k_list)]]
        hit = np.isin(top, truth)
        row = {"TRAVELER_ID": tid}
        for k in k_list:
            row[f"Recall@{k}"] = hit[:k].sum() / len(truth)
            row[f"NDCG@{k}"] = (hit[:k] * discount[:k]).sum() / discount[:min(len(truth), k)].sum()
            row[f"Hit@{k}"] = float(hit[:k].any())
        rows.append(row)
    per_user = pd.DataFrame(rows)
    return per_user.drop(columns="TRAVELER_ID").mean(), per_user


@lru_cache(maxsize=None)
def split_matrix(split, mode="warm"):
    """load_split을 행렬로. Returns 여행객 ID 목록 (행 순서), known 위치 (여행객 x 후보, bool), 정답 위치 (같은 모양)"""
    known, hidden, cand = load_split(split, mode)
    tids = list(hidden)
    pos = {it: i for i, it in enumerate(cand)}
    K = np.zeros((len(tids), len(cand)), dtype=bool)
    H = np.zeros_like(K)
    for r, t in enumerate(tids):
        K[r, [pos[i] for i in known[t] if i in pos]] = True
        H[r, [pos[i] for i in hidden[t]]] = True
    return tids, K, H


def evaluate_matrix(S, split="val", mode="warm", k_list=TOP_K):
    """evaluate()의 행렬 버전. S: 여행객 x 후보 점수 행렬 (행은 split_matrix 순서, 열은 catalog() 순서).
    여행객마다 반복하지 않고 한 번에 정렬·채점해서 빠르다. 동점 처리까지 evaluate()와 같다."""
    tids, K, H = split_matrix(split, mode)
    tiebreak = np.broadcast_to(np.random.default_rng(SEED).random(S.shape[1]), S.shape)
    discount = 1 / np.log2(np.arange(2, max(k_list) + 2))
    top = np.lexsort((tiebreak, -np.where(K, -np.inf, S)), axis=1)[:, :max(k_list)]
    hit = np.take_along_axis(H, top, axis=1)
    n_truth = H.sum(axis=1)
    per_user = pd.DataFrame({"TRAVELER_ID": tids})
    for k in k_list:
        idcg = np.array([discount[:min(n, k)].sum() for n in n_truth])
        per_user[f"Recall@{k}"] = hit[:, :k].sum(axis=1) / n_truth
        per_user[f"NDCG@{k}"] = (hit[:, :k] * discount[:k]).sum(axis=1) / idcg
        per_user[f"Hit@{k}"] = hit[:, :k].any(axis=1).astype(float)
    return per_user.drop(columns="TRAVELER_ID").mean(), per_user


def popularity_scorer():
    """베이스라인: 모든 사람에게 train 방문자 수가 많은 순서로 추천."""
    n = load_table("items").set_index("ITEM_ID")["N_TRAVELERS"]
    return lambda tid, known, cand: n.reindex(cand).to_numpy()
