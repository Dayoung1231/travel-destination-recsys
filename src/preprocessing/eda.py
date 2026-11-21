"""
데이터 탐색(EDA): 기본 점검(results/metrics/eda_inspect.txt), 그래프 6개(results/figures/eda_*.png),
요약(results/metrics/eda_summary.txt)을 만든다.

기본 점검은 표마다 크기, 열별 자료형과 결측치, 숫자 열 통계, 앞 5행을 본다.
  원본 방문지정보, 여행객 Master, 여행, 동반자정보 (TL+VL), TourAPI 관광지 목록·장소 설명,
  전처리 후 여행지 방문(visits.csv)
원본 표는 추천에 쓰는 열만 고른 표도 같이 낸다.

실행 (전처리 후, 프로젝트 최상위에서): python -m src.preprocessing.eda
"""
import matplotlib

matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from src.config import COMPANION_GROUPS, FIGURE_DIR, METRIC_DIR, PROCESSED_DIR, TOURAPI_DIR, TOURIST_TYPES
from src.preprocessing.external import load_tourapi
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


# 원본 방문지정보 23개 열 중 추천에 쓰는 13개.
# 나머지 10개는 값만 계산해 두거나(재방문·추천 의향, REVISIT_YN) 쓰지 않는다(POI_NM, 코드 열 등)
USED_VISIT_COLS = [
    "TRAVEL_ID", "VISIT_AREA_ID", "VISIT_ORDER",   # 여행·방문 연결, 여행 앞쪽 70% / 뒤쪽 30% 나누기
    "VISIT_AREA_NM", "POI_ID", "X_COORD", "Y_COORD",  # 같은 장소 합치기, TourAPI 매칭, 가까운 관측소
    "ROAD_NM_ADDR", "LOTNO_ADDR",                  # 시·도, 시·군·구 (도로명이 없으면 지번)
    "VISIT_AREA_TYPE_CD",                          # 여행지 유형 필터, CBF 장소 벡터
    "RESIDENCE_TIME_MIN", "DGSTFN",                # 장소별 체류 시간, 만족도
    "VISIT_START_YMD",                             # 방문 날짜의 날씨
]

# 원본 여행객 Master 36개 열 중 추천에 쓰는 9개 (설문 프로필 cbf.SURVEY_COLS에 들어가는 열).
# 여행 스타일 2·3·4·7·8과 동반자 수는 전처리 결과에는 남기지만 모델에는 넣지 않는다 (빼도 val 점수가 안 떨어짐).
# 동반자 그룹은 동반자정보 표, 여행 미션은 여행 표에서 와서 이 표에는 없다
USED_TRAVELLER_COLS = [
    "TRAVELER_ID",                                          # 여행·방문 연결, train / val / test 분할
    "GENDER", "AGE_GRP",                                    # 성별, 연령대
    "TRAVEL_STYL_1", "TRAVEL_STYL_5", "TRAVEL_STYL_6",      # 자연-도시, 휴양-체험, 덜 알려진 곳-알려진 곳
    "TRAVEL_MOTIVE_1", "TRAVEL_MOTIVE_2", "TRAVEL_MOTIVE_3",  # 여행 동기 (최대 3개)
]

# 원본 여행 10개 열 중 추천에 쓰는 3개.
# 여행 시작·종료 날짜로 여행 일수를 계산하지만 모델에는 넣지 않는다 (빼도 val 점수가 안 떨어짐)
USED_TRAVEL_COLS = [
    "TRAVEL_ID", "TRAVELER_ID",   # 여행객과 방문 연결
    "TRAVEL_MISSION_CHECK",       # 여행 미션 (복수 선택, '3;4;11')
]

# 원본 동반자정보 6개 열 중 추천에 쓰는 2개
USED_COMPANION_COLS = [
    "TRAVEL_ID",   # 여행과 연결 (이 표에 없는 여행은 혼자 간 여행)
    "REL_CD",      # 동반자와의 관계 -> 동반자 그룹 5개 (features.companion_group)
]

# TourAPI 관광지 목록(area_list.csv) 22개 열 중 추천에 쓰는 7개.
# 중분류·소분류(lclsSystm2, 3)는 장소 표에 가져오기만 하고 모델에는 넣지 않는다
USED_TOURAPI_COLS = [
    "contentid",          # 관광지 ID: 장소 설명(overview.csv)과 연결
    "contenttypeid",      # 관광 타입: 상설 장소에 행사가 매칭되지 않게 거른다
    "title", "addr1",     # 이름, 주소(시·도): 여행 로그 장소와 매칭
    "mapx", "mapy",       # 경도, 위도: 매칭 거리
    "lclsSystm1",         # 공식 대분류: CBF 장소 벡터 7칸
]
# 코드 -> 이름 (README 2-2의 수집 조건)
TOUR_TYPE_NAMES = {"12": "관광지", "14": "문화시설", "15": "축제공연행사", "28": "레포츠"}
TOUR_REGION_NAMES = {11: "서울", 28: "인천", 41: "경기"}


def column_info(df):
    """열별 자료형, 결측치 수, 결측 비율 표."""
    info = pd.DataFrame({"자료형": df.dtypes.astype(str), "결측치": df.isna().sum()})
    info["결측 비율"] = (info["결측치"] / len(df)).map("{:.1%}".format)
    return info


def stats_text(df, stat_cols):
    """숫자 열 통계 표. 통계를 낼 열이 없으면(숫자 열이 모두 ID·코드) 그 사실을 적는다."""
    if not stat_cols:
        return "통계를 낼 숫자 열 없음 (문자열이거나 ID·코드)"
    return df[stat_cols].describe().round(2).to_string()


def inspect(name, df, stat_cols, head_cols):
    """기본 점검: 크기, 열별 자료형·결측치, 숫자 열 통계, 앞 5행을 출력한다.
    stat_cols: 통계를 볼 열 (ID·코드·좌표 열은 평균이 의미가 없어서 고른 열만 본다)
    head_cols: 앞 5행에서 보여 줄 열
    앞 5행은 AI Hub 원본 행이 그대로 보여서 화면에만 출력하고, 파일에 쓸 줄(반환값)에는 넣지 않는다."""
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 200):
        lines = [f"===== {name} =====",
                 f"크기 (행, 열): {df.shape}",
                 "", "--- 열별 자료형과 결측치 ---", column_info(df).to_string(),
                 "", "--- 숫자 열 통계 ---", stats_text(df, stat_cols), ""]
        print("\n".join(lines))
        print("--- 앞 5행 (일부 열) ---")
        print(df[head_cols].head().to_string(index=False), "\n")
    return lines


def inspect_used(name, df, cols, stat_cols):
    """표에서 추천에 쓰는 열(cols)만 골라 열별 자료형·결측치와 숫자 열 통계를 출력한다.
    stat_cols: cols 중 통계를 볼 열. ID·코드·좌표처럼 평균이 의미가 없는 숫자 열은 넣지 않는다."""
    with pd.option_context("display.unicode.east_asian_width", True, "display.width", 200):
        lines = [f"===== {name} =====",
                 f"전체 {df.shape[1]}개 열 중 {len(cols)}개, {len(df):,}행",
                 "", "--- 열별 자료형과 결측치 ---", column_info(df[cols]).to_string(),
                 "", "--- 숫자 열 통계 ---", stats_text(df, stat_cols), ""]
        print("\n".join(lines))
    return lines


def counts_text(s, names=None):
    """값별 개수와 비율 표 (많은 순). names가 있으면 코드 옆에 이름을 붙인다."""
    c = s.value_counts()
    out = pd.DataFrame({"개수": c.map("{:,}".format), "비율": (c / len(s)).map("{:.1%}".format)})
    if names:
        out.index = [f"{k} {names[k]}" for k in c.index]
    out.index.name = None
    return out.to_string()


def inspect_tourapi(items):
    """TourAPI 관광지 목록과 장소 설명을 점검한다. 숫자 열이 좌표·코드뿐이라 통계 대신 값 분포를 본다."""
    t = load_tourapi().drop(columns=["NORM", "SIDO"])  # load_tourapi가 매칭용으로 붙인 열은 뺀다
    lines = inspect("TourAPI 관광지 목록 (area_list.csv)", t, [], ["title", "contenttypeid", "lclsSystm1", "addr1"])
    lines += inspect_used("TourAPI 관광지 목록 중 추천에 쓰는 열", t, USED_TOURAPI_COLS, [])
    matched = items["TOUR_ID"].nunique()
    with pd.option_context("display.unicode.east_asian_width", True):
        dist = ["===== TourAPI 관광지 목록의 값 분포 =====",
                "", "--- 관광 타입 (contenttypeid) ---", counts_text(t["contenttypeid"], TOUR_TYPE_NAMES),
                "", "--- 시·도 (lDongRegnCd) ---", counts_text(t["lDongRegnCd"], TOUR_REGION_NAMES),
                "", "--- 공식 대분류 (lclsSystm1) ---", counts_text(t["lclsSystm1"]),
                "", f"여행 로그 장소와 매칭된 관광지: {matched:,} / {len(t):,}곳 ({matched / len(t):.1%})", ""]
        print("\n".join(dist))

    # 설명이 빈 곳은 결측치로 읽는다. 두 열 모두 추천에 쓴다 (contentid로 잇고, overview로 TF-IDF 점수)
    ov = pd.read_csv(TOURAPI_DIR / "overview.csv", dtype=str, keep_default_na=False, na_values=[""])
    length = ov["overview"].dropna().str.len().rename("설명 길이(글자 수)")
    lines += dist + inspect("TourAPI 장소 설명 (overview.csv)", ov.assign(overview=ov["overview"].str.slice(0, 30)),
                            [], ["contentid", "overview"])
    more = ["--- 설명 길이 통계 (설명이 있는 곳) ---", length.describe().round(1).to_frame().T.to_string(), ""]
    print("\n".join(more))
    return lines + more


def inspect_all(visits, items):
    """원본 네 표, TourAPI 두 파일, 전처리 후 여행지 방문을 점검하고 eda_inspect.txt에 저장한다."""
    num_cols = ["RESIDENCE_TIME_MIN", "DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION"]
    raw_visit = read("visit").drop(columns="SOURCE")
    report = inspect("원본 방문지정보 (TL+VL)", raw_visit, num_cols,
                     ["VISIT_ORDER", "VISIT_AREA_NM", "VISIT_AREA_TYPE_CD", "RESIDENCE_TIME_MIN", "DGSTFN"])
    report += inspect_used("원본 방문지정보 중 추천에 쓰는 열", raw_visit, USED_VISIT_COLS,
                           ["VISIT_ORDER", "RESIDENCE_TIME_MIN", "DGSTFN"])
    raw_traveller = read("traveller").drop(columns="SOURCE")
    report += inspect("원본 여행객 Master (TL+VL)", raw_traveller,
                      ["AGE_GRP", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM"] + [f"TRAVEL_STYL_{i}" for i in range(1, 9)],
                      ["GENDER", "AGE_GRP", "TRAVEL_STYL_1", "TRAVEL_MOTIVE_1", "TRAVEL_COMPANIONS_NUM"])
    report += inspect_used("원본 여행객 Master 중 추천에 쓰는 열", raw_traveller, USED_TRAVELLER_COLS,
                           ["AGE_GRP", "TRAVEL_STYL_1", "TRAVEL_STYL_5", "TRAVEL_STYL_6"])
    raw_travel = read("travel").drop(columns="SOURCE")
    report += inspect("원본 여행 (TL+VL)", raw_travel, [],
                      ["TRAVEL_MISSION_CHECK", "TRAVEL_START_YMD", "TRAVEL_END_YMD"])
    report += inspect_used("원본 여행 중 추천에 쓰는 열", raw_travel, USED_TRAVEL_COLS, [])
    raw_companion = read("companion").drop(columns="SOURCE")
    report += inspect("원본 동반자정보 (TL+VL)", raw_companion, [],
                      ["REL_CD", "COMPANION_GENDER", "COMPANION_AGE_GRP"])
    report += inspect_used("원본 동반자정보 중 추천에 쓰는 열", raw_companion, USED_COMPANION_COLS, [])
    report += inspect_tourapi(items)
    report += inspect("전처리 후 여행지 방문 (visits.csv)", visits, num_cols,
                      ["VISIT_ORDER", "ITEM_ID", "ITEM_NM", "VISIT_AREA_TYPE_CD", "XY_SOURCE", "DGSTFN"])
    (METRIC_DIR / "eda_inspect.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


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

    inspect_all(visits, items)
    visit_types()
    ratings(visits)
    long_tail(visits, items)
    travelers(visits, users)
    places(items)
    dates(visits, weather)

    (METRIC_DIR / "eda_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"저장: {FIGURE_DIR}/eda_*.png, {METRIC_DIR / 'eda_summary.txt'}, {METRIC_DIR / 'eda_inspect.txt'}")


if __name__ == "__main__":
    main()
