"""
모델 기반 협업 필터링: 행렬 분해 (MF). 라이브러리 없이 numpy로 직접 구현했다.

두 가지를 만든다. 목적이 달라서 학습 데이터와 평가 방법도 다르다.

1. 평점 MF (ExplicitMF) - "이 장소에 가면 만족도가 몇 점일까"를 예측. RMSE로 평가
   예측 평점 = 전체 평균 mu + 여행객 편향 b_u + 장소 편향 b_i + p_u . q_i
   SGD로 학습 (확률적 경사 하강법). 평점 하나를 볼 때마다 오차 e = r - 예측으로
     b_u += lr (e - reg b_u),  b_i += lr (e - reg b_i)
     p_u += lr (e q_i - reg p_u),  q_i += lr (e p_u - reg q_i)
   학습: ratings_train + ratings_val_known + ratings_test_known (전처리 설계대로, MF용 장소 138곳)
   평가: ratings_val_hidden / ratings_test_hidden의 RMSE, MAE. 베이스라인(전체 평균, 장소 평균, 편향만)과 비교.

2. 방문 MF (ImplicitMF) - "이 장소에 갈까"를 예측. Top-K(Recall, NDCG)로 평가 (evaluate.py, CBF와 같은 채점)
   평점의 90%가 4~5점이라 평점이 높은 곳을 고르는 것보다 갈 만한 곳을 고르는 것이 추천에 맞다.
   간 곳 p=1, 안 간 곳 p=0으로 두고, 간 곳에 신뢰도 c = 1 + alpha를 더 준다 (Hu, Koren, Volinsky 2008).
   손실 = sum_ui c_ui (p_ui - x_u . y_i)^2 + reg (|x_u|^2 + |y_i|^2)
   ALS로 학습 (교대 최소제곱): 장소 벡터를 고정하면 여행객 벡터는 닫힌 식, 반대도 마찬가지라 번갈아 푼다.
     x_u = (Y^T C_u Y + reg I)^-1 Y^T C_u p_u
   학습은 train 여행객의 방문만 쓴다. val·test 여행객은 학습에 넣지 않고 평가할 때 벡터를 따로 구한다(fold-in).
     그래서 터미널에서 새 사용자가 들어와도 같은 방법으로 추천할 수 있다.
   fold-in: 장소 벡터 Y는 고정하고, 가 본 곳으로 x_u를 위 식으로 푼다. 규제는 0 대신 설문 벡터 쪽으로 당긴다.
     x_u = (Y^T C_u Y + reg I)^-1 (Y^T C_u p_u + reg x_survey)
     x_survey: 설문(cbf.SURVEY_COLS) -> 여행객 벡터 릿지 회귀 (train 여행객으로 학습)
     -> 가 본 곳이 없으면(cold) x_u = x_survey. 가 본 곳이 많을수록 설문보다 기록을 더 믿는다.
   장소 범위: MIN_ITEM_TRAVELERS(5명) 이상인 곳만 쓰면 후보 1,547곳 중 138곳만 점수가 생겨서,
     train 방문자 1명 이상인 곳 전체(min_travelers=1)와 비교한다. 점수가 없는 후보는 맨 뒤로 보낸다.

실행 (프로젝트 최상위 폴더에서):
    python -m src.modeling.mf            # val로 평점 MF·방문 MF 설정 비교
    python -m src.modeling.mf --test     # 위에 더해 DEFAULT_IMPLICIT를 test로 채점 (마지막에 한 번만)
    python -m src.modeling.mf --check    # 빠른 평가가 evaluate와 같은 결과인지 확인
"""
import argparse

import numpy as np
import pandas as pd

from src.config import METRIC_DIR, MIN_ITEM_TRAVELERS, SEED, TOP_K
from src.modeling.cbf import MAIN_METRIC, SURVEY_COLS
from src.modeling.evaluate import (catalog, evaluate, evaluate_matrix, load_split, load_table, popularity_scorer,
                                   split_matrix)


# ================================================================ 1. 평점 MF (SGD)

# val 그리드 후보
EXPLICIT_GRID = {"k": (2, 4, 8, 16), "reg": (0.02, 0.05, 0.1, 0.2), "epochs": (20, 50, 100)}
EXPLICIT_LR = 0.01


class ExplicitMF:
    """편향이 있는 MF. k=0이면 편향만 쓰는 베이스라인 (mu + b_u + b_i)."""

    def __init__(self, k=8, lr=EXPLICIT_LR, reg=0.05, epochs=50, seed=SEED):
        self.k, self.lr, self.reg, self.epochs, self.seed = k, lr, reg, epochs, seed

    def fit(self, ratings):
        """ratings: TRAVELER_ID, ITEM_ID, RATING 열이 있는 표."""
        self.uidx = {t: i for i, t in enumerate(ratings["TRAVELER_ID"].unique())}
        self.iidx = {t: i for i, t in enumerate(ratings["ITEM_ID"].unique())}
        u = ratings["TRAVELER_ID"].map(self.uidx).to_numpy()
        i = ratings["ITEM_ID"].map(self.iidx).to_numpy()
        r = ratings["RATING"].to_numpy(dtype=float)
        rng = np.random.default_rng(self.seed)
        self.mu = r.mean()
        self.bu, self.bi = np.zeros(len(self.uidx)), np.zeros(len(self.iidx))
        self.P = rng.normal(0, 0.1, (len(self.uidx), self.k))
        self.Q = rng.normal(0, 0.1, (len(self.iidx), self.k))
        for _ in range(self.epochs):
            for n in rng.permutation(len(r)):  # 매 epoch 평점 순서를 섞는다
                uu, ii = u[n], i[n]
                e = r[n] - (self.mu + self.bu[uu] + self.bi[ii] + self.P[uu] @ self.Q[ii])
                self.bu[uu] += self.lr * (e - self.reg * self.bu[uu])
                self.bi[ii] += self.lr * (e - self.reg * self.bi[ii])
                pu = self.P[uu].copy()  # p_u와 q_i를 같은 오차로 동시에 고친다
                self.P[uu] += self.lr * (e * self.Q[ii] - self.reg * pu)
                self.Q[ii] += self.lr * (e * pu - self.reg * self.Q[ii])
        return self

    def predict(self, tids, iids):
        """평점 예측. 학습에 없던 여행객·장소는 그 부분을 0으로 (편향만 또는 전체 평균). 1~5로 자른다."""
        u = np.array([self.uidx.get(t, -1) for t in tids])
        i = np.array([self.iidx.get(t, -1) for t in iids])
        pred = np.full(len(u), self.mu)
        pred += np.where(u >= 0, self.bu[u], 0) + np.where(i >= 0, self.bi[i], 0)
        both = (u >= 0) & (i >= 0)
        pred[both] += np.einsum("nk,nk->n", self.P[u[both]], self.Q[i[both]])
        return pred.clip(1, 5)


def explicit_train_ratings():
    """평점 MF 학습 데이터: train 전체 + val·test 여행객의 앞부분 (전처리 설계대로)."""
    return pd.concat([load_table(f"ratings_{s}") for s in ("train", "val_known", "test_known")], ignore_index=True)


def rating_errors(pred, truth):
    e = np.asarray(pred) - np.asarray(truth)
    return {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.abs(e).mean(), "N": len(e)}


def evaluate_explicit(split="val"):
    """평점 MF 그리드와 베이스라인을 split의 hidden 평점으로 채점한다."""
    train = explicit_train_ratings()
    hid = load_table(f"ratings_{split}_hidden")
    rows = {"global mean": rating_errors(np.full(len(hid), train["RATING"].mean()), hid["RATING"]),
            "item mean": rating_errors(hid["ITEM_ID"].map(train.groupby("ITEM_ID")["RATING"].mean())
                                       .fillna(train["RATING"].mean()), hid["RATING"])}
    for reg in EXPLICIT_GRID["reg"]:
        for epochs in EXPLICIT_GRID["epochs"]:
            for k in (0,) + EXPLICIT_GRID["k"]:
                m = ExplicitMF(k=k, reg=reg, epochs=epochs).fit(train)
                name = f"bias only/reg{reg}/ep{epochs}" if k == 0 else f"mf/k{k}/reg{reg}/ep{epochs}"
                rows[name] = rating_errors(m.predict(hid["TRAVELER_ID"], hid["ITEM_ID"]), hid["RATING"])
    t = pd.DataFrame(rows).T
    t["N"] = t["N"].astype(int)
    t.index.name = "MODEL"
    return t.round(4)


# ================================================================ 2. 방문 MF (ALS)

# val 그리드 후보
IMPLICIT_GRID = {"min_travelers": (1, MIN_ITEM_TRAVELERS), "k": (8, 16, 32, 64), "reg": (0.1, 1.0, 10.0),
                 "alpha": (1.0, 10.0, 40.0)}
ALS_ITERS = 15
SURVEY_RIDGE = 10.0  # 설문 -> 여행객 벡터 회귀의 L2 규제 세기 (CBF의 RIDGE와 같은 값)
# 고른 설정 (python -m src.modeling.mf의 val 결과에서 cold·warm NDCG@10 평균 1등).
# k는 32가 정점 (64는 warm이 떨어짐), alpha 10도 그리드 안쪽. reg는 0.1~1이면 차이가 거의 없다
DEFAULT_IMPLICIT = {"min_travelers": 1, "k": 32, "reg": 1.0, "alpha": 10.0}


def als_solve(F, P_rows, alpha, reg, prior=None):
    """반대쪽 벡터 F (n, k)를 고정하고 행마다 벡터를 푼다. P_rows: 행마다 간 곳(=1) 위치 목록.
    x = (F^T F + alpha F_S^T F_S + reg I)^-1 (F_S^T (1 + alpha) 1 + reg prior)
    안 간 곳은 c=1, p=0이라 F^T F에 한 번에 들어가고, 간 곳 S만 (c - 1) = alpha를 더한다."""
    k = F.shape[1]
    FtF = F.T @ F + reg * np.eye(k)
    out = np.zeros((len(P_rows), k))
    for r, idx in enumerate(P_rows):
        b = np.zeros(k) if prior is None else reg * prior[r]
        if len(idx):
            Fs = F[idx]
            A = FtF + alpha * Fs.T @ Fs
            b = b + (1 + alpha) * Fs.sum(axis=0)
        else:
            A = FtF
        out[r] = np.linalg.solve(A, b)
    return out


class ImplicitMF:
    def __init__(self, min_travelers=1, k=16, reg=10.0, alpha=10.0, iters=ALS_ITERS, seed=SEED):
        self.min_travelers, self.k, self.reg, self.alpha = min_travelers, k, reg, alpha
        self.iters, self.seed = iters, seed

    def fit(self):
        """train 여행객의 방문(holdout_by_order 전체)으로 ALS를 돌리고, 설문 -> 여행객 벡터 회귀를 푼다."""
        users, items = load_table("users"), load_table("items")
        h = load_table("holdout_by_order")
        train_ids = users.loc[users["SPLIT"] == "train", "TRAVELER_ID"]
        keep = set(items.loc[items["N_TRAVELERS"] >= self.min_travelers, "ITEM_ID"])
        tr = h[h["TRAVELER_ID"].isin(train_ids) & h["ITEM_ID"].isin(keep)]

        self.item_ids = np.sort(tr["ITEM_ID"].unique())
        self.ipos = {it: i for i, it in enumerate(self.item_ids)}
        self.user_ids = np.sort(tr["TRAVELER_ID"].unique())
        u = pd.Categorical(tr["TRAVELER_ID"], categories=self.user_ids).codes
        i = tr["ITEM_ID"].map(self.ipos).to_numpy()
        by_user = [i[u == r] for r in range(len(self.user_ids))]
        by_item = [u[i == c] for c in range(len(self.item_ids))]

        rng = np.random.default_rng(self.seed)
        self.Y = rng.normal(0, 0.1, (len(self.item_ids), self.k))
        for _ in range(self.iters):
            X = als_solve(self.Y, by_user, self.alpha, self.reg)
            self.Y = als_solve(X, by_item, self.alpha, self.reg)
        X = als_solve(self.Y, by_user, self.alpha, self.reg)  # 마지막 Y에 맞춘 여행객 벡터

        # 설문 -> 여행객 벡터 릿지 회귀 (절편은 규제하지 않음). 방문 기록이 없는 사람의 시작점
        self.U = load_table("user_features").set_index("TRAVELER_ID")[SURVEY_COLS]
        A = np.hstack([np.ones((len(X), 1)), self.U.loc[self.user_ids].to_numpy()])
        reg = SURVEY_RIDGE * np.eye(A.shape[1])
        reg[0, 0] = 0
        self.W = np.linalg.solve(A.T @ A + reg, A.T @ X)
        return self

    def user_vector(self, survey, known=()):
        """설문 벡터와 가 본 곳으로 여행객 벡터 (fold-in). 모델에 없는 장소는 무시한다."""
        prior = np.r_[1.0, survey] @ self.W
        idx = np.array([self.ipos[i] for i in known if i in self.ipos], dtype=int)
        return als_solve(self.Y, [idx], self.alpha, self.reg, prior=prior[None])[0]

    def item_matrix(self, candidates):
        """후보 장소 벡터. 모델에 없는 장소는 nan 행 (점수를 -inf로 만든다)."""
        pos = np.array([self.ipos.get(i, -1) for i in candidates])
        Yc = np.full((len(candidates), self.k), np.nan)
        Yc[pos >= 0] = self.Y[pos[pos >= 0]]
        return Yc

    @staticmethod
    def _finish(s):
        return np.where(np.isnan(s), -np.inf, s)

    def score_survey(self, survey, known, candidates):
        return self._finish(self.item_matrix(candidates) @ self.user_vector(survey, known))

    def score(self, tid, known, candidates):
        """evaluate()에 넘기는 점수 함수."""
        return self.score_survey(self.U.loc[tid].to_numpy(), known, candidates)

    def score_matrix(self, split, mode):
        """evaluate_matrix에 넘기는 점수 행렬 (행은 split_matrix, 열은 catalog() 순서). 결과는 score와 같다."""
        known, _, cand = load_split(split, mode)
        tids = split_matrix(split, mode)[0]
        U = self.U.loc[tids].to_numpy()
        priors = np.hstack([np.ones((len(U), 1)), U]) @ self.W
        rows = [np.array([self.ipos[i] for i in known[t] if i in self.ipos], dtype=int) for t in tids]
        Xv = als_solve(self.Y, rows, self.alpha, self.reg, prior=priors)
        return self._finish(Xv @ self.item_matrix(cand).T)

    def recommend_survey(self, survey, known=(), n=10, candidates=None):
        """설문 벡터 + 가 본 곳으로 추천할 장소 n개 (이름, 유형, 시·도, 점수). cbf.CBF.recommend_survey와 같은 모양."""
        items = load_table("items").set_index("ITEM_ID")
        known = list(known)
        candidates = pd.Index(catalog() if candidates is None else candidates).difference(known)
        s = pd.Series(self.score_survey(survey, known, candidates), index=candidates).nlargest(n)
        return items.loc[s.index, ["ITEM_NM", "TYPE_NM", "SIDO"]].assign(SCORE=s.round(3))

    def recommend(self, tid, known=(), n=10, candidates=None):
        return self.recommend_survey(self.U.loc[tid].to_numpy(), known, n, candidates)


def implicit_name(cfg):
    return f"mf/min{cfg['min_travelers']}/k{cfg['k']}/reg{cfg['reg']}/a{cfg['alpha']}"


def tune_implicit():
    """방문 MF 그리드를 val cold·warm으로 채점한다. 고르는 기준은 cold·warm MAIN_METRIC 평균."""
    rows = []
    for mt in IMPLICIT_GRID["min_travelers"]:
        for k in IMPLICIT_GRID["k"]:
            for reg in IMPLICIT_GRID["reg"]:
                for alpha in IMPLICIT_GRID["alpha"]:
                    cfg = {"min_travelers": mt, "k": k, "reg": reg, "alpha": alpha}
                    model = ImplicitMF(**cfg).fit()
                    row = {"MODEL": implicit_name(cfg), **cfg}
                    for mode in ("cold", "warm"):
                        mean = evaluate_matrix(model.score_matrix("val", mode), "val", mode)[0]
                        row.update({f"{mode} {m}": v for m, v in mean.items()})
                    row["SELECT"] = (row[f"cold {MAIN_METRIC}"] + row[f"warm {MAIN_METRIC}"]) / 2
                    rows.append(row)
    return pd.DataFrame(rows).round(4)


def explicit_ranking_scorer(model):
    """비교용: 평점 MF의 예측 평점 순으로 추천. 평점 MF에 없는 장소(MF용 138곳 밖)는 맨 뒤."""
    def score(tid, known, cand):
        s = model.predict([tid] * len(cand), cand)
        return np.where(np.isin(cand, list(model.iidx)), s, -np.inf)
    return score


def check_fast_path():
    """score_matrix(여러 사람 한 번에)가 score(한 사람씩)와 같은 결과인지 확인한다 (cbf.check_fast_path와 같은 목적)."""
    for cfg in (DEFAULT_IMPLICIT, {**DEFAULT_IMPLICIT, "min_travelers": MIN_ITEM_TRAVELERS}):
        model = ImplicitMF(**cfg).fit()
        for mode in ("cold", "warm"):
            slow = evaluate(model.score, "val", mode)[1].set_index("TRAVELER_ID")
            fast = evaluate_matrix(model.score_matrix("val", mode), "val", mode)[1].set_index("TRAVELER_ID")
            diff = np.abs(slow.to_numpy() - fast.to_numpy()).max()
            ok = slow.index.equals(fast.index) and diff < 1e-9
            print(f"{'OK ' if ok else '다름'} {implicit_name(cfg)} {mode}: 최대 차이 {diff:.1e}")
            if not ok:
                raise SystemExit("빠른 평가 결과가 evaluate와 달라요. score와 score_matrix를 같이 고쳐 주세요.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="DEFAULT_IMPLICIT를 test로 채점")
    ap.add_argument("--check", action="store_true", help="빠른 평가(evaluate_matrix)가 evaluate와 같은 결과인지만 확인")
    args = ap.parse_args()
    if args.check:
        check_fast_path()
        return
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    show = [f"{m}@{k}" for k in TOP_K for m in ("Recall", "NDCG")]

    # 1. 평점 MF: RMSE
    ex = evaluate_explicit("val")
    ex.to_csv(METRIC_DIR / "mf_val_rating.csv")
    best_ex = ex.drop(index=["global mean", "item mean"])["RMSE"].idxmin()
    print(f"\n[val / 평점 예측] hidden 평점 {ex['N'].iloc[0]}건 (RMSE 낮은 순 10개)")
    print(ex.sort_values("RMSE").head(10).to_string())
    print(f"  베이스라인: {ex.loc[['global mean', 'item mean'], ['RMSE', 'MAE']].to_dict('index')}")

    # 2. 방문 MF: Top-K
    grid = tune_implicit()
    grid.to_csv(METRIC_DIR / "mf_val_implicit.csv", index=False)
    best = grid.loc[grid["SELECT"].idxmax()]
    cfg = {"min_travelers": int(best["min_travelers"]), "k": int(best["k"]),
           "reg": float(best["reg"]), "alpha": float(best["alpha"])}
    cols = ["MODEL"] + [f"{m} {k}" for m in ("cold", "warm") for k in (MAIN_METRIC, "Recall@20")]
    print(f"\n[val / 방문 MF] cold·warm {MAIN_METRIC} 평균 높은 순 10개")
    print(grid.sort_values("SELECT", ascending=False)[cols].head(10).to_string(index=False))
    print(f"  -> 고른 설정 {cfg} (터미널 추천에 쓰려면 DEFAULT_IMPLICIT에 반영)")

    # 고른 설정을 인기도·평점 MF 순위와 나란히 비교
    ex_model = ExplicitMF(**_parse_explicit(best_ex)).fit(explicit_train_ratings())
    model = ImplicitMF(**cfg).fit()
    for mode in ("cold", "warm"):
        t = pd.DataFrame({
            "popularity": evaluate(popularity_scorer(), "val", mode)[0],
            f"평점 MF 순위 ({best_ex})": evaluate(explicit_ranking_scorer(ex_model), "val", mode)[0],
            implicit_name(cfg): evaluate_matrix(model.score_matrix("val", mode), "val", mode)[0],
        }).T.round(4)
        t.index.name = "MODEL"
        t.to_csv(METRIC_DIR / f"mf_val_{mode}.csv")
        print(f"\n[val / {mode}] 여행객 {len(load_split('val', mode)[1])}명 평균")
        print(t[show].to_string())

    # 예시: val 여행객 한 명
    items = load_table("items").set_index("ITEM_ID")
    known, hidden, _ = load_split("val", "warm")
    tid = next(iter(hidden))
    print(f"\n예시 여행객 {tid} ({implicit_name(cfg)})")
    print("  앞에서 간 곳:", ", ".join(items.loc[known[tid], "ITEM_NM"]))
    print("  뒤에서 간 곳:", ", ".join(items.loc[hidden[tid], "ITEM_NM"]))
    print(model.recommend(tid, known[tid]).to_string())

    if args.test:
        # 방문 MF는 최종 시스템(터미널 추천과 발표 그래프)과 같은 DEFAULT_IMPLICIT로 채점한다.
        # 위에서 고른 그리드 1등은 val 차이가 거의 없어서(reg 0.1과 1.0), test를 본 뒤 바꾸지 않도록 DEFAULT로 고정한다
        cfg = DEFAULT_IMPLICIT
        model = ImplicitMF(**cfg).fit()
        test_ex = ex_model.predict(load_table("ratings_test_hidden")["TRAVELER_ID"],
                                   load_table("ratings_test_hidden")["ITEM_ID"])
        print(f"\n[test / 평점 예측] {best_ex}: {rating_errors(test_ex, load_table('ratings_test_hidden')['RATING'])}")
        for mode in ("cold", "warm"):
            t = pd.DataFrame({
                "popularity": evaluate(popularity_scorer(), "test", mode)[0],
                implicit_name(cfg): evaluate_matrix(model.score_matrix("test", mode), "test", mode)[0],
            }).T.round(4)
            t.index.name = "MODEL"
            t.to_csv(METRIC_DIR / f"mf_test_{mode}.csv")
            print(f"\n[test / {mode}]")
            print(t[show].to_string())


def _parse_explicit(name):
    """'mf/k8/reg0.05/ep50' 또는 'bias only/reg0.05/ep50' -> ExplicitMF 인자."""
    parts = name.split("/")
    k = int(parts[1][1:]) if parts[0] == "mf" else 0
    return {"k": k, "reg": float(parts[-2][3:]), "epochs": int(parts[-1][2:])}


if __name__ == "__main__":
    main()
