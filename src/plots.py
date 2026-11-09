"""
발표용 성능 그래프. 결과는 results/figures/에 PNG로, 그래프에 쓴 숫자는 results/metrics/에 CSV로 저장한다.

  1_models       모델 비교: 무작위, 인기순, 평점 MF, CBF, 방문 MF (cold·warm, NDCG@10, 95% 신뢰구간)
  2_cbf_steps    CBF 단계별 성능: 유사도만 -> +인기도 -> +Q6 개인화 -> +동반자 -> +텍스트
  3_warm_region  warm 사용자를 "정답 중 다녀온 곳과 같은 시·군·구 장소가 있는지"로 나눈 비교
  4_tuning       설정 곡선: CBF 텍스트 세기, 방문 MF의 k와 alpha (val 그리드 CSV를 그대로 읽음)

1~3은 여행객마다 NDCG@10을 다시 계산해서 평균과 95% 신뢰구간(t 분포)을 그린다.
단계 사이의 차이는 같은 여행객끼리 비교한 대응 t-검정의 p값이다.
4는 cbf.py, mf.py가 저장한 val 그리드 결과(cbf_val_text.csv, mf_val_implicit.csv)를 쓴다.

실행 (프로젝트 최상위 폴더에서, cbf.py와 mf.py를 먼저 실행해 둔다):
    python -m src.plots                 # val (몇 분 걸림)
    python -m src.plots --split test    # 모든 모델이 끝난 뒤 test로 한 번만
"""
import argparse

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd
from scipy import stats

from src.config import FIGURE_DIR, METRIC_DIR, SEED
from src.modeling import cbf, mf
from src.modeling.evaluate import evaluate, evaluate_matrix, load_split, load_table, popularity_scorer, split_matrix

METRIC = "NDCG@10"
MODES = ("cold", "warm")
MODE_TITLE = {"cold": "cold: 설문만 입력", "warm": "warm: 설문 + 다녀온 곳 입력"}

# 색: 모델은 항상 같은 색 (CBF 파랑, 방문 MF 주황), 기준 모델은 회색.
# 설정 곡선(4)은 cold 보라, warm 청록. 두 쌍 모두 색각 이상 구분 검사를 통과한 조합
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
MODEL_COLOR = {"무작위": "#d6d5ce", "인기순": "#b4b2a9", "평점 MF 순위": "#b4b2a9",
               "CBF": "#2a78d6", "방문 MF": "#eb6834"}
MODE_COLOR = {"cold": "#6250d6", "warm": "#1baf7a"}

plt.rcParams.update({
    "font.family": "Malgun Gothic",  # 한글 (Windows). 다른 OS는 설치된 한글 글꼴 이름으로 바꾼다
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
})


# ---------------------------------------------------------------- 계산

def ci(x):
    """평균과 95% 신뢰구간 (t 분포)."""
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def per_user(model, split, mode):
    """여행객마다 METRIC (index TRAVELER_ID)."""
    return evaluate_matrix(model.score_matrix(split, mode), split, mode)[1].set_index("TRAVELER_ID")[METRIC]


def random_scores(split, mode, n_runs=20):
    """무작위 추천: 난수 점수로 n_runs번 채점한 여행객별 평균 (한 번만 하면 값이 크게 흔들려서)."""
    rng = np.random.default_rng(SEED)
    shape = split_matrix(split, mode)[2].shape
    runs = [evaluate_matrix(rng.random(shape), split, mode)[1].set_index("TRAVELER_ID")[METRIC]
            for _ in range(n_runs)]
    return pd.concat(runs, axis=1).mean(axis=1)


def rating_mf_model():
    """평점 MF 중 val RMSE가 가장 낮은 설정 (mf.py가 저장한 mf_val_rating.csv에서 고른다)."""
    t = pd.read_csv(METRIC_DIR / "mf_val_rating.csv", index_col="MODEL")
    best = t.drop(index=["global mean", "item mean"])["RMSE"].idxmin()
    return mf.ExplicitMF(**mf._parse_explicit(best)).fit(mf.explicit_train_ratings())


def cbf_steps():
    """CBF 단계별 설정: 앞 단계에 하나씩 더한다. 세기는 터미널 추천과 같은 기본값."""
    A = cbf.DEFAULT_ADJUST
    return [("① 유사도만", {}),
            ("② +인기도", {"pop": A["pop"]}),
            ("③ +Q6 개인화", {"pop": A["pop"], "pop_style": A["pop_style"]}),
            ("④ +동반자", dict(A)),
            ("⑤ +텍스트", {**A, "text": cbf.DEFAULT_TEXT})]


def same_area_users(split):
    """warm 여행객마다 정답 중 다녀온 곳과 같은 시·군·구 장소가 있는지 (index TRAVELER_ID, bool)."""
    items = load_table("items").set_index("ITEM_ID")
    known, hidden, _ = load_split(split, "warm")
    out = {}
    for t, truth in hidden.items():
        areas = set(items.loc[known[t], ["SIDO", "SIGUNGU"]].dropna().itertuples(index=False))
        out[t] = any(a in areas for a in items.loc[truth, ["SIDO", "SIGUNGU"]].itertuples(index=False))
    return pd.Series(out)


def compute(split):
    """그래프 1~3에 필요한 여행객별 점수. Returns {(이름, mode): Series}"""
    res = {}
    print("기준 모델 채점")
    for mode in MODES:
        res[("무작위", mode)] = random_scores(split, mode)
        res[("인기순", mode)] = evaluate(popularity_scorer(), split, mode)[1].set_index("TRAVELER_ID")[METRIC]
        res[("평점 MF 순위", mode)] = (evaluate(mf.explicit_ranking_scorer(rating_mf_model()), split, mode)[1]
                                     .set_index("TRAVELER_ID")[METRIC])
    print("방문 MF 학습·채점")
    m = mf.ImplicitMF(**mf.DEFAULT_IMPLICIT).fit()
    for mode in MODES:
        res[("방문 MF", mode)] = per_user(m, split, mode)
    for name, adjust in cbf_steps():
        print(f"CBF {name}")
        model = cbf.build_model(cbf.DEFAULT_MODEL, **adjust)
        for mode in MODES:
            res[(name, mode)] = per_user(model, split, mode)
    for mode in MODES:
        res[("CBF", mode)] = res[("⑤ +텍스트", mode)]
    return res


def summary_table(res, split):
    """그래프에 쓴 평균·신뢰구간을 표로 (그래프의 표 버전)."""
    rows = []
    for (name, mode), s in res.items():
        m, lo, hi = ci(s)
        rows.append({"MODEL": name, "MODE": mode, "N": len(s), METRIC: m, "CI_LOW": lo, "CI_HIGH": hi})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- 그리기

def bar_with_ci(ax, x, series_list, colors, width=0.62, labels=True):
    """막대 + 95% 신뢰구간. 막대 사이는 배경색 2px 간격."""
    stats_ = [ci(s) for s in series_list]
    means = [s[0] for s in stats_]
    err = np.array([[m - lo for m, lo, _ in stats_], [hi - m for m, _, hi in stats_]])
    ax.bar(x, means, width=width, color=colors, edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.errorbar(x, means, yerr=err, fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    if labels:
        for xi, (m, _, hi) in zip(x, stats_):
            ax.text(xi, hi + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color=INK2)
    return stats_


def p_text(a, b):
    """같은 여행객끼리 비교한 b - a의 평균 차이와 p값."""
    d = (b - a).mean()
    p = stats.ttest_rel(b, a).pvalue
    p_str = "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"
    return d, p, f"{d:+.3f}\n{p_str}"


def fig_models(res, split, n_users):
    names = ["무작위", "인기순", "평점 MF 순위", "CBF", "방문 MF"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, mode in zip(axes, MODES):
        x = np.arange(len(names))
        st = bar_with_ci(ax, x, [res[(n, mode)] for n in names], [MODEL_COLOR[n] for n in names])
        best = int(np.argmax([s[0] for s in st]))
        ax.annotate("Final recommendation model", (x[best], st[best][2] + 0.04), ha="center", fontsize=9, color=INK,
                    fontweight="bold")
        ax.set_xticks(x, names)
        ax.set_title(f"{MODE_TITLE[mode]} ({n_users[mode]}명)", loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel(f"{METRIC} (평균, 95% 신뢰구간)")
    axes[0].set_ylim(0, 0.42)
    fig.suptitle(f"모델 비교 ({split}): cold는 CBF, warm은 방문 MF가 가장 좋다", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, "후보 1,516곳 전체 순위. 평점 MF 순위는 MF용 138곳만 점수가 있음. "
             "막대 위 숫자는 평균, 선은 95% 신뢰구간.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def fig_cbf_steps(res, split, n_users):
    names = [n for n, _ in cbf_steps()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, mode in zip(axes, MODES):
        x = np.arange(len(names))
        st = bar_with_ci(ax, x, [res[(n, mode)] for n in names], [MODEL_COLOR["CBF"]] * len(names), labels=False)
        for i in range(1, len(names)):
            _, p, txt = p_text(res[(names[i - 1], mode)], res[(names[i], mode)])
            sig = p < 0.05
            ax.text(x[i], st[i][2] + 0.01, txt, ha="center", va="bottom", fontsize=8.5, linespacing=1.1,
                    color=INK if sig else MUTED, fontweight="bold" if sig else "normal")
        ax.text(x[0], st[0][2] + 0.01, f"{st[0][0]:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(x, names, fontsize=9)
        ax.set_title(f"{MODE_TITLE[mode]} ({n_users[mode]}명)", loc="left", fontsize=11, color=INK)
    axes[0].set_ylabel(f"{METRIC} (평균, 95% 신뢰구간)")
    axes[0].set_ylim(0, 0.32)
    fig.suptitle(f"CBF 단계별 성능 ({split}): 인기도 보정만 확실한 개선", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, "막대 위 숫자: 직전 단계 대비 평균 차이와 p값 (같은 여행객끼리 대응 t-검정). "
             "굵은 글씨는 p<0.05.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def fig_warm_region(res, split):
    same = same_area_users(split)
    groups = [(True, "같은 시·군·구 정답 있음"), (False, "같은 시·군·구 정답 없음")]
    names = ["인기순", "CBF", "방문 MF"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    w = 0.26
    for j, n in enumerate(names):
        xs, ss = [], []
        for g, (flag, _) in enumerate(groups):
            ids = same.index[same == flag]
            xs.append(g + (j - 1) * w)
            ss.append(res[(n, "warm")].loc[ids])
        bar_with_ci(ax, xs, ss, [MODEL_COLOR[n]] * 2, width=w)
    ax.set_xticks(range(len(groups)),
                  [f"{lab}\n({int((same == flag).sum())}명)" for flag, lab in groups])
    ax.set_ylabel(f"warm {METRIC} (평균, 95% 신뢰구간)")
    ax.set_ylim(0, 0.5)
    handles = [Patch(color=MODEL_COLOR[n], label=n) for n in names]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.suptitle(f"warm 지역별 비교 ({split}): 방문 MF의 강점은 근처 장소", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, "정답(여행 뒤 30%) 중 다녀온 곳(앞 70%)과 같은 시·군·구의 장소가 있는지로 나눔. "
             "오른쪽 그룹은 인원이 적어 신뢰구간이 넓다.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def tuning_panel(ax, x_labels, cold, warm, chosen, title, xlabel):
    x = np.arange(len(x_labels))
    cold, warm = np.asarray(cold, float), np.asarray(warm, float)
    for mode, y in (("cold", cold), ("warm", warm)):
        ax.plot(x, y, color=MODE_COLOR[mode], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    # 선 끝 직접 라벨. 두 끝점이 가까우면 겹치지 않게 위아래로 벌린다
    gap = 0.012
    yc, yw = cold[-1], warm[-1]
    if abs(yw - yc) < 2 * gap:
        mid = (yw + yc) / 2
        yw, yc = (mid + gap, mid - gap) if yw >= yc else (mid - gap, mid + gap)
    for mode, y in (("cold", yc), ("warm", yw)):
        ax.text(x[-1] + 0.12, y, mode, color=INK2, fontsize=9, va="center")
    c = x_labels.index(chosen)
    ax.axvline(c, color=AXIS, linewidth=1, zorder=1)
    ax.text(c, 0.97, " 선택", transform=ax.get_xaxis_transform(), color=INK, fontsize=9, va="top",
            fontweight="bold")
    ax.set_xticks(x, x_labels)
    ax.set_xlim(-0.3, len(x) - 0.4)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel)


def fig_tuning():
    t = pd.read_csv(METRIC_DIR / "cbf_val_text.csv")
    g = pd.read_csv(METRIC_DIR / "mf_val_implicit.csv")
    D = mf.DEFAULT_IMPLICIT
    base = g[(g["min_travelers"] == D["min_travelers"]) & (g["reg"] == D["reg"])]
    ks = base[base["alpha"] == D["alpha"]].sort_values("k")
    al = base[base["k"] == D["k"]].sort_values("alpha")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    fmt = lambda v: f"{v:g}"
    tuning_panel(axes[0], [fmt(v) for v in t["text"]], t[f"cold {METRIC}"], t[f"warm {METRIC}"],
                 fmt(cbf.DEFAULT_TEXT), "CBF 텍스트 점수 세기", "세기 (0 = 텍스트 없음)")
    tuning_panel(axes[1], [fmt(v) for v in ks["k"]], ks[f"cold {METRIC}"], ks[f"warm {METRIC}"],
                 fmt(D["k"]), "방문 MF 임베딩 크기 k", f"k (alpha {fmt(D['alpha'])}, reg {fmt(D['reg'])})")
    tuning_panel(axes[2], [fmt(v) for v in al["alpha"]], al[f"cold {METRIC}"], al[f"warm {METRIC}"],
                 fmt(D["alpha"]), "방문 MF 간 곳 가중치 alpha", f"alpha (k {fmt(D['k'])}, reg {fmt(D['reg'])})")
    axes[0].set_ylabel(f"val {METRIC}")
    axes[0].set_ylim(0, 0.34)
    fig.suptitle("설정 곡선 (val): 고른 값이 cold·warm 평균 기준 가장 좋다", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.01, "cbf_val_text.csv, mf_val_implicit.csv의 val 그리드 결과. "
             "고르는 기준은 cold·warm NDCG@10 평균.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=("val", "test"),
                    help="test는 모든 모델이 끝난 뒤 한 번만")
    args = ap.parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    split = args.split

    res = compute(split)
    n_users = {mode: len(res[("CBF", mode)]) for mode in MODES}
    summary_table(res, split).to_csv(METRIC_DIR / f"plots_{split}_summary.csv", index=False)

    figs = {"1_models": fig_models(res, split, n_users),
            "2_cbf_steps": fig_cbf_steps(res, split, n_users),
            "3_warm_region": fig_warm_region(res, split)}
    if split == "val":
        figs["4_tuning"] = fig_tuning()  # 설정은 val로만 고르므로 val에서만 그린다
    for name, fig in figs.items():
        path = FIGURE_DIR / f"{split}_{name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("저장:", path.relative_to(FIGURE_DIR.parents[1]))


if __name__ == "__main__":
    main()
