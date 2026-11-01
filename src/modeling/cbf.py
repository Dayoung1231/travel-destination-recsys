"""
콘텐츠 기반 필터링 (CBF).

장소 벡터: item_features의 내용 열 (방문지 유형, TourAPI 대분류, 활동 비율, 체류 시간).
  인기도(POPULARITY)와 동반자 비율(COMP_)은 제안서에서 '점수 보정' 요소라 여기서는 빼고 나중에 따로 더한다.
사용자 벡터 (장소 벡터와 같은 공간):
  - 이력 프로필: known에서 간 장소 벡터의 가중 평균. 만족도가 높을수록 크게 (5점 3, 4점 2, 3점 1, 2점 이하 0)
  - 설문 프로필: 설문(성별, 연령, 동반자, 여행 스타일, 미션, 동기)을 장소 벡터 공간으로 옮긴 것.
    train 여행객의 (설문, 간 장소 벡터 평균) 쌍으로 릿지 회귀를 풀어 옮기는 행렬 W를 배운다.
    방문 기록이 없는 새 사용자(cold start)도 추천할 수 있다.
    설문 항목은 SURVEY_COLS만 쓴다 (빼도 val 점수가 떨어지지 않은 항목은 터미널 설문에서도 뺐다).
  - 혼합: known이 n개면 이력 n / (n + SHRINK), 설문 나머지. 기록이 적을수록 설문을 더 믿는다.
유사도: 코사인 (피어슨, 가중 자카드와 비교).
점수 보정 (제안서 요인 ①, ②): 최종 점수 = 유사도 + 인기도 보정 + 동반자 보정
  - 인기도: (POP + POP_STYLE x (Q6 - 0.5)) x 장소 인기도.
    Q6(덜 알려진 곳 0 ~ 알려진 곳 1)에 따라 세기가 달라져서, 덜 알려진 곳을 원하면 약해지거나 감점이 된다.
  - 동반자: COMP x log(장소 방문자 중 내 동반자 그룹 비율 / 전체 중 그 그룹 비율).
    내 그룹이 평소보다 많이 가는 곳은 더하고 적게 가는 곳은 뺀다. 방문자가 적은 장소는 비율이 들쭉날쭉해서
    전체 비율 쪽으로 당겨서 쓴다 (COMP_PRIOR명분의 가상 방문자).
  보정 세기(POP, POP_STYLE, COMP)는 val로 고른다.

평가는 두 상황으로 나눈다 (evaluate.py).
  - cold: 설문만 입력 -> 설문 프로필만 쓸 수 있다
  - warm: 설문 + 가 본 곳 입력 -> 이력·설문·혼합 프로필 비교

실행 (프로젝트 최상위 폴더에서):
    python -m src.modeling.cbf            # val로 설정 비교 + 점수 보정 세기 고르기

    (x)
    python -m src.modeling.cbf --test     # 위에 더해 DEFAULT_MODEL의 보정 전·후를 test로 채점 (마지막에 한 번만)
    python -m src.modeling.cbf --check    # 빠른 평가가 evaluate와 같은 결과인지 확인
"""
import argparse

import numpy as np
import pandas as pd

from src.config import COMPANION_GROUPS, METRIC_DIR, TOP_K
from src.modeling.evaluate import (catalog, evaluate, evaluate_matrix, load_split, load_table, popularity_scorer,
                                   split_matrix)

# 설문 프로필에 쓰는 설문 항목 (터미널 설문 문항과 같아야 함).
# 하나씩 빼 보며 val 점수를 비교해서, 빼도 떨어지지 않은 항목은 뺐다 (47개 -> 27개, 차이 유의하지 않음):
# 여행 기간, 여행 스타일 Q2·3·4·7·8 (숙소·익숙함·계획성·사진), 여행지와 관련이 약한 미션·동기 보기
SURVEY_STYLES = (1, 5, 6)   # 자연-도시, 휴양-체험, 덜 알려진 곳-알려진 곳 (6은 인기도 보정에도 쓸 예정)
SURVEY_MISSIONS = (1, 2, 3, 5, 6, 8, 9, 10, 11, 21, 22, 24, 28)
SURVEY_MOTIVES = (2, 3, 7, 8)
SURVEY_COLS = (["GENDER_F", "AGE"] + [f"COMP_{g}" for g in COMPANION_GROUPS]
               + [f"STYL_{i}" for i in SURVEY_STYLES]
               + [f"MIS_{c}" for c in SURVEY_MISSIONS] + [f"MOT_{c}" for c in SURVEY_MOTIVES])

# 비교할 장소 벡터 구성 (열 이름 앞부분)
FEATURE_SETS = {
    "content": ("TYPE_", "TCAT_", "ACT_", "STAY"),
    "content+region": ("TYPE_", "TCAT_", "ACT_", "STAY", "SIDO_"),
}
PROFILES = ("history", "survey", "hybrid")
SIMILARITIES = ("cosine", "pearson", "jaccard")
SHRINK = 2.0   # 혼합 비율: known 2개면 이력과 설문을 반반
RIDGE = 10.0   # 설문 -> 장소 벡터 회귀의 L2 규제 세기
MAIN_METRIC = "NDCG@10"  # 설정을 고르는 기준 (TOP_K에 10이 있어야 함)
# 터미널 추천에 쓰는 설정 (val 결과에서 cold·warm 모두 1등). 가 본 곳이 없으면 hybrid도 설문 프로필만 쓴다
DEFAULT_MODEL = "cbf/content+region/hybrid/jaccard"

# 점수 보정 세기 후보 (val 그리드). 0이면 보정 없음
POP_GRID = (0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5)
POP_STYLE_GRID = (0, 0.2, 0.4, 0.8)
COMP_GRID = (0, 0.02, 0.05, 0.1)
COMP_PRIOR = 5  # 동반자 비율을 전체 비율 쪽으로 당길 때의 가상 방문자 수
# 터미널 추천에 쓰는 보정 세기 (val 그리드에서 cold·warm NDCG@10 평균 1등).
# 인기도 0.5는 그리드 안쪽의 최고점. 개인화(pop_style)와 동반자(comp)는 더해도 차이가 작다
DEFAULT_ADJUST = {"pop": 0.5, "pop_style": 0.4, "comp": 0.02}


def similarity(p, X, method="cosine"):
    """프로필 p (d,)와 장소 행렬 X (n, d)의 유사도 (n,). 영벡터와의 유사도는 0."""
    if method == "pearson":  # 각 벡터에서 자기 평균을 뺀 뒤의 코사인
        p, X = p - p.mean(), X - X.mean(axis=1, keepdims=True)
        method = "cosine"
    if method == "cosine":
        norm = np.linalg.norm(X, axis=1) * np.linalg.norm(p)
        return np.divide(X @ p, norm, out=np.zeros(len(X)), where=norm > 0)
    if method == "jaccard":  # 0~1 실수값용 가중 자카드: sum(min) / sum(max)
        p = np.clip(p, 0, None)  # 회귀로 만든 설문 프로필은 음수가 나올 수 있다
        num, den = np.minimum(X, p).sum(axis=1), np.maximum(X, p).sum(axis=1)
        return np.divide(num, den, out=np.zeros(len(X)), where=den > 0)
    raise ValueError(f"모르는 유사도: {method}")


class CBF:
    def __init__(self, features="content", profile="hybrid", sim="cosine", shrink=SHRINK, ridge=RIDGE,
                 pop=0, pop_style=0, comp=0):
        self.features, self.profile_type, self.sim = features, profile, sim
        self.shrink, self.ridge = shrink, ridge
        self.pop, self.pop_style, self.comp = pop, pop_style, comp  # 점수 보정 세기 (fit 없이 바꿔도 됨)

    def fit(self):
        """장소 벡터를 준비하고, train 여행객으로 설문 -> 장소 벡터 회귀(W)를 푼다."""
        f = load_table("item_features").set_index("ITEM_ID")
        self.X = f[[c for c in f.columns if c.startswith(FEATURE_SETS[self.features])]]
        self.U = load_table("user_features").set_index("TRAVELER_ID")[SURVEY_COLS]

        # 여행객 x 장소 만족도 (이력 프로필 가중치용)
        v = load_table("visits")
        self.sat = v.groupby(["TRAVELER_ID", "ITEM_ID"])["DGSTFN"].mean()

        # 회귀 목표: train 여행객마다 간 장소 벡터의 평균. 절편 열은 규제하지 않는다
        users = load_table("users")
        h = load_table("holdout_by_order")
        tr = h[h["TRAVELER_ID"].isin(users.loc[users["SPLIT"] == "train", "TRAVELER_ID"])]
        T = self.X.reindex(tr["ITEM_ID"]).set_axis(tr["TRAVELER_ID"]).groupby(level=0).mean()
        A = np.hstack([np.ones((len(T), 1)), self.U.loc[T.index].to_numpy()])
        reg = self.ridge * np.eye(A.shape[1])
        reg[0, 0] = 0
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ T.to_numpy())

        # 점수 보정용 장소 값 (둘 다 전처리에서 train 방문자로만 계산한 값)
        self.popularity = f["POPULARITY"]
        n = load_table("items").set_index("ITEM_ID")["N_TRAVELERS"].reindex(f.index).fillna(0)
        cols = [f"COMP_{g}" for g in COMPANION_GROUPS]
        # 전체 비율도 장소별 비율과 같은 기준: 여행지를 한 곳 이상 방문한 train 여행객
        prior = (users[users["TRAVELER_ID"].isin(tr["TRAVELER_ID"])]["COMPANION_GROUP"]
                 .value_counts(normalize=True).reindex(COMPANION_GROUPS).to_numpy())
        share = (f[cols].to_numpy() * n.to_numpy()[:, None] + COMP_PRIOR * prior) / (n.to_numpy()[:, None] + COMP_PRIOR)
        self.comp_lift = pd.DataFrame(np.log(share / prior), index=f.index, columns=COMPANION_GROUPS)
        return self

    def history_profile(self, known, sat):
        """가 본 곳 벡터의 가중 평균. sat는 장소마다 만족도 1~5 (모르면 nan -> 3점으로 취급)."""
        w = (np.nan_to_num(np.asarray(sat, dtype=float), nan=3.0) - 2).clip(min=0)
        if w.sum() == 0:  # 간 곳이 모두 2점 이하면 똑같이 취급
            w = np.ones(len(known))
        return w @ self.X.loc[known].to_numpy() / w.sum()

    def survey_profile(self, survey):
        """survey: user_features 열 순서의 설문 벡터 (터미널 입력은 recommend.survey_vector로 만든다)"""
        return np.r_[1.0, survey] @ self.W

    def make_profile(self, survey, known=(), sat=None):
        """설문 벡터와 가 본 곳(+만족도)으로 장소 벡터 공간 프로필을 만든다. 가 본 곳이 없으면 설문 프로필만 쓴다."""
        known = list(known)
        if self.profile_type == "survey" or not known:
            return self.survey_profile(survey)
        hist = self.history_profile(known, [np.nan] * len(known) if sat is None else sat)
        if self.profile_type == "history":
            return hist
        a = len(known) / (len(known) + self.shrink)
        return a * hist + (1 - a) * self.survey_profile(survey)

    def adjustment(self, survey, candidates):
        """후보 장소마다 점수 보정 값 (제안서 요인 ① 인기도, ② 동반자). 보정 세기가 0이면 0."""
        s = pd.Series(survey, index=self.U.columns)
        pop_w = self.pop + self.pop_style * (s["STYL_6"] - 0.5)  # STYL_6: 0 덜 알려진 곳 ~ 1 알려진 곳
        group = next(g for g in COMPANION_GROUPS if s[f"COMP_{g}"] == 1)
        return (pop_w * self.popularity.loc[candidates].to_numpy()
                + self.comp * self.comp_lift.loc[candidates, group].to_numpy())

    def score_survey(self, survey, known, sat, candidates):
        """설문 벡터와 가 본 곳(+만족도)으로 후보 장소마다 최종 점수 = 유사도 + 점수 보정."""
        p = self.make_profile(survey, known, sat)
        return similarity(p, self.X.loc[candidates].to_numpy(), self.sim) + self.adjustment(survey, candidates)

    def known_sat(self, tid, known):
        """데이터에 있는 여행객이 known 장소에 준 만족도 (없으면 nan)."""
        return self.sat.reindex([(tid, i) for i in known]).to_numpy()

    def score(self, tid, known, candidates):
        """evaluate()에 넘기는 점수 함수. 데이터에 있는 여행객의 설문과 만족도를 가져와 score_survey로 계산한다."""
        return self.score_survey(self.U.loc[tid].to_numpy(), known, self.known_sat(tid, known), candidates)

    # ------------------------------------------------ 빠른 평가 (여러 설정 비교용, 결과는 score와 같다)

    def score_parts(self, split, mode):
        """split·mode 여행객 전체의 점수 재료. 최종 점수 = S + (pop + pop_style x (q6 - 0.5)) x P + comp x L.
        유사도 S는 보정 세기와 상관없어서, 세기를 바꿔 가며 비교할 때 한 번만 계산해 다시 쓴다.
        장소 행렬도 후보마다 한 번만 꺼낸다 (여행객마다 .loc으로 꺼내는 게 가장 느렸다).
        Returns S (여행객 x 후보), q6 (여행객,), P (후보,), L (여행객 x 후보). 행은 split_matrix, 열은 catalog() 순서"""
        known, _, cand = load_split(split, mode)
        tids = split_matrix(split, mode)[0]
        Xc = self.X.loc[cand].to_numpy()
        U = self.U.loc[tids].to_numpy()
        S = np.vstack([similarity(self.make_profile(u, known[t], self.known_sat(t, known[t])), Xc, self.sim)
                       for t, u in zip(tids, U)])
        q6 = U[:, self.U.columns.get_loc("STYL_6")]
        group = U[:, [self.U.columns.get_loc(f"COMP_{g}") for g in COMPANION_GROUPS]].argmax(axis=1)
        L = self.comp_lift.loc[cand].to_numpy()[:, group].T
        return S, q6, self.popularity.loc[cand].to_numpy(), L

    def combine(self, S, q6, P, L):
        """score_parts의 재료에 지금 보정 세기를 적용한 점수 행렬 (adjustment와 같은 계산)."""
        pop_w = self.pop + self.pop_style * (q6 - 0.5)
        return S + (pop_w[:, None] * P + self.comp * L)

    def score_matrix(self, split, mode):
        """evaluate_matrix에 넘기는 점수 행렬 (여행객 x 후보)."""
        return self.combine(*self.score_parts(split, mode))

    def recommend_survey(self, survey, known=(), sat=None, n=10, candidates=None):
        """설문 벡터 + 가 본 곳으로 추천할 장소 n개 (이름, 유형, 시·도, 점수). 후보 기본값은 evaluate.catalog()."""
        items = load_table("items").set_index("ITEM_ID")
        known = list(known)
        if candidates is None:
            candidates = catalog()
        candidates = pd.Index(candidates).difference(known)
        s = pd.Series(self.score_survey(survey, known, sat, candidates), index=candidates).nlargest(n)
        return items.loc[s.index, ["ITEM_NM", "TYPE_NM", "SIDO"]].assign(SCORE=s.round(3))

    def recommend(self, tid, known=(), n=10, candidates=None):
        """데이터에 있는 여행객 한 명에게 추천할 장소 n개."""
        return self.recommend_survey(self.U.loc[tid].to_numpy(), known, self.known_sat(tid, known), n, candidates)


def build_model(name, **adjust):
    """'cbf/<특성 구성>/<프로필>/<유사도>' -> 학습한 CBF. adjust: pop, pop_style, comp (점수 보정 세기)"""
    _, feat, prof, sim = name.split("/")
    return CBF(feat, prof, sim, **adjust).fit()


def tune_adjustment(name):
    """name 설정에 점수 보정 세기를 그리드로 바꿔 가며 val cold·warm을 채점한다.
    보정은 fit 결과와 유사도를 바꾸지 않아서, 학습과 유사도 계산은 한 번만 하고 보정만 바꿔 더한다.
    고르는 기준은 cold·warm MAIN_METRIC 평균."""
    model = build_model(name)
    parts = {mode: model.score_parts("val", mode) for mode in ("cold", "warm")}
    rows = []
    for pop in POP_GRID:
        for pop_style in POP_STYLE_GRID:
            for comp in COMP_GRID:
                model.pop, model.pop_style, model.comp = pop, pop_style, comp
                row = {"pop": pop, "pop_style": pop_style, "comp": comp}
                for mode, p in parts.items():
                    mean = evaluate_matrix(model.combine(*p), "val", mode)[0]
                    row.update({f"{mode} {k}": v for k, v in mean.items()})
                row["SELECT"] = (row[f"cold {MAIN_METRIC}"] + row[f"warm {MAIN_METRIC}"]) / 2
                rows.append(row)
    return pd.DataFrame(rows).round(4)


def check_fast_path():
    """빠른 평가가 여행객 한 명씩 채점하는 evaluate와 같은 결과를 내는지 확인한다 (점수 계산을 고친 뒤 실행).
    한 사람용(score)과 여러 사람용(score_matrix) 코드가 두 벌이라 한쪽만 고쳐 어긋나는 것을 막는다."""
    for name, adjust in ((DEFAULT_MODEL, {}), (DEFAULT_MODEL, DEFAULT_ADJUST), ("cbf/content/survey/cosine", {})):
        model = build_model(name, **adjust)
        for mode in ("cold", "warm"):
            slow = evaluate(model.score, "val", mode)[1].set_index("TRAVELER_ID")
            fast = evaluate_matrix(model.score_matrix("val", mode), "val", mode)[1].set_index("TRAVELER_ID")
            diff = np.abs(slow.to_numpy() - fast.to_numpy()).max()
            ok = slow.index.equals(fast.index) and diff < 1e-9
            print(f"{'OK ' if ok else '다름'} {name} {adjust or '보정 없음'} {mode}: 최대 차이 {diff:.1e}")
            if not ok:
                raise SystemExit("빠른 평가 결과가 evaluate와 달라요. score와 score_parts/combine을 같이 고쳐 주세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="DEFAULT_MODEL의 보정 전·후와 인기도를 test로 채점")
    ap.add_argument("--check", action="store_true", help="빠른 평가(evaluate_matrix)가 evaluate와 같은 결과인지만 확인")
    args = ap.parse_args()
    if args.check:
        check_fast_path()
        return
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    show = [f"{m}@{k}" for k in TOP_K for m in ("Recall", "NDCG")]
    items = load_table("items").set_index("ITEM_ID")

    # 기본 설정 비교 (보정 없음). cold에서는 가 본 곳이 없어서 설문 프로필만 비교한다
    profiles_of = {"cold": ("survey",), "warm": PROFILES}
    for mode, profiles in profiles_of.items():
        results = {"popularity": evaluate(popularity_scorer(), "val", mode)[0]}
        for feat in FEATURE_SETS:
            for prof in profiles:
                for sim in SIMILARITIES:
                    name = f"cbf/{feat}/{prof}/{sim}"
                    results[name] = evaluate_matrix(build_model(name).score_matrix("val", mode), "val", mode)[0]
        table = pd.DataFrame(results).T.round(4)
        table.index.name = "MODEL"
        table.to_csv(METRIC_DIR / f"cbf_val_{mode}.csv")
        n_users = len(load_split("val", mode)[1])
        print(f"\n[val / {mode}] 여행객 {n_users}명 평균 ({MAIN_METRIC} 높은 순)")
        print(table.sort_values(MAIN_METRIC, ascending=False)[show].to_string())

    # 점수 보정: DEFAULT_MODEL에 보정 세기를 바꿔 가며 비교 (인기도만 / 동반자만 / 둘 다)
    grid = tune_adjustment(DEFAULT_MODEL)
    grid.to_csv(METRIC_DIR / "cbf_val_adjust.csv", index=False)
    no_pop, no_comp = grid["pop"].eq(0) & grid["pop_style"].eq(0), grid["comp"].eq(0)
    picks = {"보정 없음": grid[no_pop & no_comp], "인기도만": grid[no_comp], "동반자만": grid[no_pop], "둘 다": grid}
    summary = pd.DataFrame({k: g.loc[g["SELECT"].idxmax()] for k, g in picks.items()}).T
    cols = ["pop", "pop_style", "comp"] + [f"{m} {k}" for m in ("cold", "warm") for k in (MAIN_METRIC, "Recall@20")]
    print(f"\n[val / 점수 보정] {DEFAULT_MODEL}, 각 경우에서 cold·warm {MAIN_METRIC} 평균이 가장 높은 세기")
    print(summary[cols].to_string())
    adjust = summary.loc["둘 다", ["pop", "pop_style", "comp"]].to_dict()
    print(f"  -> 고른 보정 세기 {adjust} (터미널 추천에 쓰려면 DEFAULT_ADJUST에 반영)")

    # 예시: val 여행객 한 명. cold는 설문만, warm은 앞에서 간 곳까지 보고 추천
    known, hidden, _ = load_split("val", "warm")
    tid = next(iter(hidden))
    model = build_model(DEFAULT_MODEL, **adjust)
    print(f"\n예시 여행객 {tid} ({DEFAULT_MODEL} + 보정)")
    print("  앞에서 간 곳:", ", ".join(items.loc[known[tid], "ITEM_NM"]))
    print("  뒤에서 간 곳:", ", ".join(items.loc[hidden[tid], "ITEM_NM"]))
    for mode, given in (("cold", []), ("warm", known[tid])):
        print(f"  [{mode}]")
        print(model.recommend(tid, given).to_string())

    if args.test:
        base = build_model(DEFAULT_MODEL)
        for mode in ("cold", "warm"):
            test = pd.DataFrame({
                "popularity": evaluate(popularity_scorer(), "test", mode)[0],
                DEFAULT_MODEL: evaluate_matrix(base.score_matrix("test", mode), "test", mode)[0],
                f"{DEFAULT_MODEL} + 보정": evaluate_matrix(model.score_matrix("test", mode), "test", mode)[0],
            }).T.round(4)
            test.index.name = "MODEL"
            test.to_csv(METRIC_DIR / f"cbf_test_{mode}.csv")
            print(f"\n[test / {mode}]")
            print(test[show].to_string())


if __name__ == "__main__":
    main()
