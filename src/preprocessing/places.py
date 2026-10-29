"""
같은 장소 합치기 (장소 ID 통일).

원본의 POI_ID를 그대로 쓰면 같은 장소가 여러 개로 쪼개진다.
- 같은 장소에 POI_ID가 여러 개 (좌표가 같은 경기장에 ID 4개)
- POI_NM이 엉뚱한 장소로 찍힌 경우가 있음 -> 여행객이 쓴 VISIT_AREA_NM을 쓴다
- POI_ID가 없는 방문이 약 8%
- 넓은 장소가 세부 이름으로 나뉨 (석촌호수 / 석촌호수 동호 / 석촌호수 서호)
그래서 아래 조건 중 하나라도 맞는 방문끼리 연결하고, 연결된 묶음을 장소 하나로 본다 (union-find).
"""
import os
import re

import numpy as np
import pandas as pd

# 기준값은 사람이 검수한 장소 쌍(같은 여행지인가)으로 정밀도·재현율을 비교해서 정했다
MERGE_DIST_M = 300      # 이름이 비슷하고 이 거리 안이면 같은 장소
PART_DIST_M = 2000      # '석촌호수' -> '석촌호수 동호'처럼 상위-세부 관계면 이 거리까지 (호수·산·공원은 넓다)
SIBLING_DIST_M = 150    # '흥국사 삼성각' / '흥국사 약사전'처럼 앞부분이 같은 세부 장소끼리
POI_MAX_DIST_M = 5000   # POI_ID가 같아도 이보다 멀면 다른 장소 (19km 떨어진 두 '신흥시장')
NAME_SIM = 0.5          # 글자 bigram Dice 유사도 기준
PHOTO_MAX_SPREAD_M = 300  # 사진 GPS는 그 방문의 사진끼리 이 거리 안에서 일치할 때만 사용

SIDO_ALIAS = {"서울특별시": "서울", "경기도": "경기", "인천광역시": "인천",
              "강원도": "강원", "강원특별자치도": "강원", "충청남도": "충남", "충청북도": "충북"}
VALID_SIDO = {"서울", "경기", "인천", "강원", "충남", "충북", "대전", "세종", "전북", "전남",
              "경북", "경남", "대구", "부산", "울산", "광주", "제주"}


def norm_name(s):
    """이름 정규화: 괄호 안, 공백, 특수문자를 지우고 소문자로. '둘리의 거리(송내역 로데오거리)' -> '둘리의거리'"""
    s = re.sub(r"\(.*?\)|\[.*?\]", "", str(s))
    return re.sub(r"[\s\W_]+", "", s).lower()


def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def similar(a, b):
    """정규화한 두 이름이 같은 장소를 가리키는지.
    같거나, 한쪽이 다른 쪽을 포함하거나(짧은 쪽 4글자 이상, '계곡'이 모든 계곡에 붙지 않게),
    앞 두 글자가 같고 bigram 유사도가 NAME_SIM 이상 ('플래그십' / '플래그쉽' 같은 표기 차이)."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 4 and short in long_:
        return True
    # 끝말만 같은 다른 가게('...아트센터')가 붙지 않게 앞 두 글자도 같아야 한다
    if a[:2] != b[:2]:
        return False
    ba, bb = bigrams(a), bigrams(b)
    return 2 * len(ba & bb) / (len(ba) + len(bb)) >= NAME_SIM


def region_of(addr):
    """'경기 수원시 팔달구 ...' -> ('경기', '수원시'). 알 수 없으면 (nan, nan)."""
    if not isinstance(addr, str) or not addr.strip():
        return np.nan, np.nan
    tok = addr.split()
    sido = SIDO_ALIAS.get(tok[0], tok[0])
    if sido not in VALID_SIDO:  # 시·도가 빠진 주소 ('북한강로 123')
        return np.nan, np.nan
    # 시·군·구가 빠진 주소('경기 214-2')도 있어서 그때는 비워 둔다
    sgg = tok[1] if len(tok) > 1 and re.search(r"[시군구]$", tok[1]) else np.nan
    return sido, sgg


def dist_m(x1, y1, x2, y2):
    """경위도 두 점 사이의 대략적인 거리(m). 위도 37도 부근 기준, 좌표가 없으면 nan."""
    return np.hypot((x1 - x2) * 88_000, (y1 - y2) * 111_000)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def fill_xy_from_photos(v, photo):
    """좌표가 없는 방문은 그 방문에서 찍은 관광사진 GPS의 중앙값으로 채운다.
    사진끼리 PHOTO_MAX_SPREAD_M보다 어긋나면(이동 중에 찍은 사진 등) 믿지 않고 비워 둔다.
    XY_SOURCE 열에 좌표 출처(visit / photo / none)를 남긴다."""
    ph = photo[photo["PHOTO_FILE_X_COORD"].between(124, 132) & photo["PHOTO_FILE_Y_COORD"].between(33, 39)]
    g = ph.groupby(["TRAVEL_ID", "VISIT_AREA_ID"])
    gps = g[["PHOTO_FILE_X_COORD", "PHOTO_FILE_Y_COORD"]].median()
    spread = dist_m(g["PHOTO_FILE_X_COORD"].max(), g["PHOTO_FILE_Y_COORD"].max(),
                    g["PHOTO_FILE_X_COORD"].min(), g["PHOTO_FILE_Y_COORD"].min())
    gps = gps[spread <= PHOTO_MAX_SPREAD_M]
    key = pd.MultiIndex.from_frame(v[["TRAVEL_ID", "VISIT_AREA_ID"]])
    px = gps["PHOTO_FILE_X_COORD"].reindex(key).to_numpy()
    py = gps["PHOTO_FILE_Y_COORD"].reindex(key).to_numpy()
    v["XY_SOURCE"] = np.where(v["X_COORD"].notna(), "visit", np.where(~np.isnan(px), "photo", "none"))
    v["X_COORD"] = v["X_COORD"].fillna(pd.Series(px, index=v.index))
    v["Y_COORD"] = v["Y_COORD"].fillna(pd.Series(py, index=v.index))
    return v


def assign_places(v):
    """같은 장소인 방문끼리 묶어 ITEM_ID를 붙인다. 아래 중 하나라도 맞으면 연결한다.
      1) POI_ID가 같고, 이름이 비슷하고, POI_MAX_DIST_M 이내
      2) MERGE_DIST_M 이내이고 이름이 비슷함
      3) PART_DIST_M 이내이고 이름이 같거나 상위-세부 관계 ('석촌호수' / '석촌호수 동호')
      4) SIBLING_DIST_M 이내이고 이름 앞 3글자 이상이 같음 ('흥국사 삼성각' / '흥국사 약사전')
      5) 좌표가 없는 방문: 이름이 같거나 상위-세부 관계이고, 시·도가 서로 다르지 않음
    """
    v = v.reset_index(drop=True)
    names = v["VISIT_AREA_NM"].map(norm_name).tolist()
    X, Y = v["X_COORD"].to_numpy(), v["Y_COORD"].to_numpy()
    uf = UnionFind(len(v))

    def d(a, b):
        return dist_m(X[a], Y[a], X[b], Y[b])

    def part_of(a, b):
        """b가 a의 일부: '석촌호수' -> '석촌호수 서호', '관악산' -> '관악산 연주대'"""
        return len(names[a]) >= 3 and names[b].startswith(names[a])

    # 1) 같은 POI_ID
    for idx in v.groupby("POI_ID").groups.values():
        idx = list(idx)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a, b = idx[i], idx[j]
                if similar(names[a], names[b]) and not d(a, b) > POI_MAX_DIST_M:
                    uf.union(a, b)

    # 2) ~ 4) 가까운 방문끼리: 약 1km 격자로 나눠 이웃 칸끼리만 비교
    has_xy = v["X_COORD"].notna() & v["Y_COORD"].notna()
    cell = {}
    for i in v.index[has_xy]:
        cell.setdefault((int(X[i] / 0.012), int(Y[i] / 0.012)), []).append(i)
    for (cx, cy), members in cell.items():
        near = [k for dx in (-1, 0, 1) for dy in (-1, 0, 1) for k in cell.get((cx + dx, cy + dy), [])]
        for a in members:
            for b in near:
                if b <= a or uf.find(a) == uf.find(b):
                    continue
                dab = d(a, b)
                if dab <= MERGE_DIST_M and similar(names[a], names[b]):
                    uf.union(a, b)
                elif dab <= PART_DIST_M and (names[a] == names[b] or part_of(a, b) or part_of(b, a)):
                    uf.union(a, b)
                elif dab <= SIBLING_DIST_M and len(os.path.commonprefix([names[a], names[b]])) >= 3:
                    uf.union(a, b)

    # 5) 좌표 없는 방문 ('석촌호수'만 쓰고 주소가 없는 기록)
    sido = v["SIDO"].to_numpy()
    for i in v.index[~has_xy]:
        if len(names[i]) < 3:
            continue
        for j in v.index:
            if i == j or uf.find(i) == uf.find(j):
                continue
            if isinstance(sido[i], str) and isinstance(sido[j], str) and sido[i] != sido[j]:
                continue
            if names[i] == names[j] or part_of(i, j) or part_of(j, i):
                uf.union(i, j)

    root = pd.Series([uf.find(i) for i in v.index], index=v.index)
    # POI_ID를 그대로 ID로 쓰면 두 묶음이 같은 POI_ID를 가질 수 있어서 새 ID(P00000)를 붙이고
    # 묶음에서 가장 많이 나온 POI_ID는 REP_POI_ID로 남긴다
    order = {r: k for k, r in enumerate(sorted(root.unique()))}
    v["ITEM_ID"] = root.map(lambda r: f"P{order[r]:05d}")
    v["REP_POI_ID"] = v.groupby("ITEM_ID")["POI_ID"].transform(
        lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    # 표시 이름: 여행객이 가장 많이 쓴 이름 (POI_NM은 틀린 경우가 있음)
    v["ITEM_NM"] = v.groupby("ITEM_ID")["VISIT_AREA_NM"].transform(lambda s: s.mode().iloc[0])
    return v
