"""
데이터 탐색(EDA): 그래프 6개(results/figures/eda_*.png)와 요약(results/metrics/eda_summary.txt)을 만든다.

실행 (전처리 후, 프로젝트 최상위에서): python -m src.preprocessing.eda
"""
import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.config import COMPANION_GROUPS, FIGURE_DIR, METRIC_DIR, PROCESSED_DIR, TOURIST_TYPES
from src.preprocessing.load_data import read

# 방문지 유형 코드 -> 이름
TYPE_NAMES = {
    1: "자연관광지", 2: "역사·유적·종교", 3: "문화시설", 4: "상업지구", 5: "레저·스포츠",
    6: "테마시설", 7: "산책로·둘레길", 8: "지역 축제", 9: "역·터미널·휴게소", 10: "상점",
    11: "식당·카페", 12: "기타", 13: "체험 관광지", 21: "집", 22: "친구·친지집", 23: "사무실", 24: "숙소",
}
SIDO_ORDER = ["서울", "경기", "인천"]
OTHER = "주소 없음·수도권 밖"
COMPANION_SHORT = {"solo": "혼자", "couple": "연인·배우자", "family_kids": "아이 동반 가족",
                   "family_adults": "성인 가족", "friends": "친구·모임"}

# 그래프 색 (남긴 것 파랑, 뺀 것 회색)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
KEEP, DROP, ACCENT = "#2a78d6", "#c3c2b7", "#eb6834"
SIDO_COLOR = {"서울": "#2a78d6", "경기": "#1baf7a", "인천": "#eb6834", OTHER: "#b4b2a9"}

# 한글 글꼴: Windows 맑은 고딕, macOS 애플고딕, Linux 나눔고딕 중 설치된 것
INSTALLED = {f.name for f in font_manager.fontManager.ttflist}
KOREAN_FONT = next((f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in INSTALLED), "DejaVu Sans")

plt.rcParams.update({
    "font.family": KOREAN_FONT,
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10,
})

summary = []  # eda_summary.txt에 쓸 줄


def note(line):
    """요약 한 줄을 출력하고 summary에 모은다."""
    print(line)
    summary.append(line)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


def label_bars(ax, bars, fmt="{:,.0f}", horizontal=False):
    """막대 끝에 값을 적는다."""
    for b in bars:
        if horizontal:
            ax.annotate(fmt.format(b.get_width()), (b.get_width(), b.get_y() + b.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        else:
            ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK2)


def visit_types():
    """1. 원본 방문 기록의 방문지 유형 분포 (추천 대상으로 남긴 유형은 파랑)."""
    raw = read("visit")
    counts = raw["VISIT_AREA_TYPE_CD"].value_counts()
    keep = {int(c) for c in TOURIST_TYPES}
    counts = counts.sort_values()
    colors = [KEEP if c in keep else DROP for c in counts.index]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh([TYPE_NAMES.get(c, str(c)) for c in counts.index], counts.values, color=colors)
    label_bars(ax, bars, horizontal=True)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("방문 기록 수")
    n_keep = counts[[c for c in counts.index if c in keep]].sum()
    ax.set_title(f"원본 방문 기록 {len(raw):,}건의 방문지 유형  (파랑: 추천 대상으로 남긴 유형 {n_keep:,}건)",
                 loc="left", fontsize=11)
    save(fig, "eda_1_visit_types")

    note(f"[1] 원본 방문 기록 {len(raw):,}건, 여행지 유형 {n_keep:,}건 ({n_keep / len(raw):.0%})")
    top = counts.sort_values(ascending=False).head(5)
    note("    많은 유형: " + ", ".join(f"{TYPE_NAMES.get(c, c)} {n:,}" for c, n in top.items()))


def ratings(visits):
    """2. 여행지 방문의 만족도(1~5) 분포."""
    r = visits["DGSTFN"].dropna().astype(int)
    counts = r.value_counts().reindex(range(1, 6), fill_value=0)
    share = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=[DROP if s < 4 else KEEP for s in counts.index])
    for b, s in zip(bars, share):
        ax.annotate(f"{b.get_height():,}\n({s:.0%})", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("만족도 (1 매우 불만족 ~ 5 매우 만족)")
    ax.set_ylabel("방문 수")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_title(f"여행지 방문 {len(r):,}건의 만족도  (4~5점 {share[[4, 5]].sum():.0%}, 평균 {r.mean():.2f})",
                 loc="left", fontsize=11)
    save(fig, "eda_2_ratings")

    note(f"[2] 만족도 {len(r):,}건: " + ", ".join(f"{k}점 {v:.1%}" for k, v in share.items())
         + f" / 평균 {r.mean():.2f}, 표준편차 {r.std():.2f}")


def long_tail(visits, items):
    """3. 장소별 방문한 여행객 수 (긴 꼬리)와 상위 15곳."""
    per_item = visits.groupby("ITEM_ID")["TRAVELER_ID"].nunique().sort_values(ascending=False)
    n = len(per_item)
    top10 = per_item.head(int(np.ceil(n * 0.1))).sum() / per_item.sum()
    one = (per_item == 1).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax1.plot(np.arange(1, n + 1), per_item.values, color=KEEP, linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("장소 순위 (방문자 많은 순)")
    ax1.set_ylabel("방문한 여행객 수 (로그)")
    ax1.axhline(5, color=ACCENT, linewidth=1, linestyle="--")
    ax1.annotate("MF용 기준 (5명 이상)", (n * 0.55, 5), xytext=(0, 4), textcoords="offset points",
                 color=ACCENT, fontsize=8)
    ax1.set_title(f"장소 {n:,}곳: 상위 10%가 방문의 {top10:.0%}, 1명만 간 곳 {one:.0%}", loc="left", fontsize=11)

    top = per_item.head(15)[::-1]
    names = items.set_index("ITEM_ID").loc[top.index, "ITEM_NM"]
    bars = ax2.barh(names, top.values, color=KEEP)
    label_bars(ax2, bars, horizontal=True)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", visible=True)
    ax2.set_xlabel("방문한 여행객 수")
    ax2.set_title("방문자가 많은 장소 상위 15곳", loc="left", fontsize=11)
    save(fig, "eda_3_long_tail")

    note(f"[3] 장소 {n:,}곳, 방문자 수 중앙값 {per_item.median():.0f}명, 최대 {per_item.max()}명 ({names.iloc[-1]})")
    note(f"    상위 10% 장소가 방문의 {top10:.0%}, 1명만 간 장소 {one:.0%}, 5명 이상 {(per_item >= 5).sum()}곳")


def travelers(visits, users):
    """4. 여행객별 간 여행지 수, 동반자 그룹, 연령대."""
    per_user = visits.groupby("TRAVELER_ID")["ITEM_ID"].nunique()
    per_user = per_user.reindex(users["TRAVELER_ID"], fill_value=0)  # 여행지를 한 곳도 안 간 사람은 0
    capped = per_user.clip(upper=10)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    counts = capped.value_counts().reindex(range(0, 11), fill_value=0)
    bars = ax.bar([str(i) if i < 10 else "10+" for i in counts.index], counts.values,
                  color=[DROP if i < 2 else KEEP for i in counts.index])
    label_bars(ax, bars)
    ax.set_xlabel("간 여행지 수")
    ax.set_ylabel("여행객 수")
    ax.set_title(f"여행객 {len(per_user):,}명이 간 여행지 수 (중앙값 {per_user.median():.0f}곳)", loc="left", fontsize=11)

    ax = axes[1]
    comp = users["COMPANION_GROUP"].value_counts().reindex(COMPANION_GROUPS, fill_value=0)
    bars = ax.bar([COMPANION_SHORT[g] for g in comp.index], comp.values, color=KEEP)
    label_bars(ax, bars)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("동반자 그룹", loc="left", fontsize=11)

    ax = axes[2]
    age = users["AGE_GRP"].value_counts().sort_index()
    bars = ax.bar([f"{a}대" for a in age.index], age.values, color=KEEP)
    label_bars(ax, bars)
    ax.set_title("연령대", loc="left", fontsize=11)
    save(fig, "eda_4_travelers")

    note(f"[4] 여행객 {len(per_user):,}명: 여행지 0곳 {(per_user == 0).sum()}명, 1곳 {(per_user == 1).sum()}명, "
         f"2곳 이상 {(per_user >= 2).sum()}명, 중앙값 {per_user.median():.0f}곳, 평균 {per_user.mean():.2f}곳")
    note("    동반자: " + ", ".join(f"{g} {n}" for g, n in comp.items()))
    note("    연령대: " + ", ".join(f"{a}대 {n}" for a, n in age.items())
         + " / 성별: " + ", ".join(f"{g} {n}" for g, n in users["GENDER"].value_counts().items()))


def places(items):
    """5. 장소 위치(시·도별 색)와 유형별 장소 수."""
    sido = items["SIDO"].where(items["SIDO"].isin(SIDO_ORDER), OTHER)
    has_xy = items["X"].notna() & items["Y"].notna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.2]})
    for s in SIDO_ORDER + [OTHER]:
        m = has_xy & (sido == s)
        ax1.scatter(items.loc[m, "X"], items.loc[m, "Y"], s=8, alpha=0.6, color=SIDO_COLOR[s],
                    label=f"{s} {m.sum():,}")
    ax1.set_xlim(126.0, 128.0)  # 수도권 범위
    ax1.set_ylim(36.85, 38.35)
    ax1.set_aspect(1.25)
    ax1.grid(visible=False)
    ax1.set_xlabel("경도")
    ax1.set_ylabel("위도")
    ax1.legend(frameon=False, fontsize=8, loc="lower left")
    ax1.set_title(f"장소 {has_xy.sum():,}곳의 위치 (좌표 없는 {(~has_xy).sum()}곳 제외)", loc="left", fontsize=11)

    types = items["TYPE_CD"].astype(int)
    table = pd.crosstab(types, sido).reindex(columns=SIDO_ORDER + [OTHER], fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values().index]
    left = np.zeros(len(table))
    for s in table.columns:
        ax2.barh([TYPE_NAMES[t] for t in table.index], table[s], left=left, color=SIDO_COLOR[s], label=s)
        left += table[s].to_numpy()
    for y, total in enumerate(left):
        ax2.annotate(f"{total:,.0f}", (total, y), xytext=(3, 0), textcoords="offset points",
                     va="center", fontsize=8, color=INK2)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", visible=True)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.set_xlabel("장소 수")
    ax2.set_title("유형별 장소 수 (시·도별)", loc="left", fontsize=11)
    save(fig, "eda_5_places")

    note(f"[5] 장소 {len(items):,}곳: " + ", ".join(f"{s} {n}" for s, n in sido.value_counts().items())
         + f" (주소 없음 {items['SIDO'].isna().sum()}, 수도권 밖 {(items['SIDO'].notna() & (sido == OTHER)).sum()})"
         + f" / 좌표 없음 {(~has_xy).sum()}곳")
    note("    유형: " + ", ".join(f"{TYPE_NAMES[t]} {n}" for t, n in types.value_counts().items()))


def dates(visits, weather):
    """6. 날짜별 방문 수 (주말, 비 온 날 구분)."""
    day = pd.to_datetime(visits["VISIT_START_YMD"]).dt.normalize()
    per_day = day.value_counts().sort_index()
    rainy_day = (weather.assign(DATE=pd.to_datetime(weather["DATE"]))
                 .groupby("DATE")["RAINY"].mean() >= 0.5)  # 그날 방문의 절반 이상이 비 온 관측소면 비 온 날
    rainy_day = rainy_day.reindex(per_day.index, fill_value=False)
    weekend = per_day.index.day_of_week >= 5

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(per_day.index, per_day.values, width=0.9,
           color=np.where(rainy_day, ACCENT, np.where(weekend, KEEP, "#8fb8e8")))
    ax.set_ylabel("여행지 방문 수")
    ax.set_title("날짜별 여행지 방문  (진한 파랑 주말, 연한 파랑 평일, 주황 비 온 날)", loc="left", fontsize=11)
    save(fig, "eda_6_dates")

    wk = day.dt.day_of_week >= 5
    note(f"[6] 방문 기간 {per_day.index.min():%Y-%m-%d} ~ {per_day.index.max():%Y-%m-%d}, 방문이 있는 날 {len(per_day)}일")
    note(f"    주말 방문 {wk.mean():.0%}, 비 온 날 방문 {weather['RAINY'].mean():.0%}, "
         f"월별: " + ", ".join(f"{m}월 {n}" for m, n in day.dt.month.value_counts().sort_index().items()))


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    visits = pd.read_csv(PROCESSED_DIR / "visits.csv")
    items = pd.read_csv(PROCESSED_DIR / "items.csv")
    users = pd.read_csv(PROCESSED_DIR / "users.csv")
    weather = pd.read_csv(PROCESSED_DIR / "weather.csv")

    visit_types()
    ratings(visits)
    long_tail(visits, items)
    travelers(visits, users)
    places(items)
    dates(visits, weather)

    (METRIC_DIR / "eda_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"저장: {FIGURE_DIR}/eda_*.png, {METRIC_DIR / 'eda_summary.txt'}")


if __name__ == "__main__":
    main()
