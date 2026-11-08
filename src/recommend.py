"""
터미널 추천: 설문을 입력받아 수도권 여행지를 추천한다.

실행 (프로젝트 최상위 폴더에서):
    python -m src.recommend
    python -m src.recommend --n 20     # 추천 개수

흐름
  1. 기본 정보: 성별, 연령대, 동반자
  2. 여행 스타일 3문항 (1~7 척도, AI Hub 여행 스타일 문항 중 cbf.SURVEY_STYLES)
  3. 여행 미션(복수 선택), 여행 동기(최대 3개). 보기는 cbf.SURVEY_MISSIONS, SURVEY_MOTIVES
  4. 원하는 지역 (서울 / 경기 / 인천 / 상관없음)
  5. (선택) 가 본 곳과 만족도 -> 입력하면 이력까지 반영 (warm), 건너뛰면 설문만 (cold)
문항은 CBF 설문 프로필이 쓰는 항목(cbf.SURVEY_COLS)과 같다.
설문 답은 전처리와 같은 함수(features.build_user_features)로 특성 벡터로 바꾼다.
"""
import argparse
import unicodedata

import pandas as pd

from src.config import COMPANION_GROUPS, COMPANION_LABELS, REGIONS
from src.modeling.cbf import (DEFAULT_ADJUST, DEFAULT_MODEL, DEFAULT_TEXT, SURVEY_MISSIONS, SURVEY_MOTIVES,
                              SURVEY_STYLES, build_model)
from src.modeling.evaluate import catalog, load_table
from src.preprocessing.features import build_user_features
from src.preprocessing.places import norm_name

# 화면에 보여 줄 방문지 유형 이름 (config.TOURIST_TYPES의 한글)
TYPE_KO = {1: "자연 관광지", 2: "역사·유적·종교", 3: "문화시설", 5: "레저·스포츠",
           6: "테마시설", 7: "산책로·둘레길", 8: "지역 축제", 13: "체험 관광지"}
AGE_GROUPS = {"20대": 20, "30대": 30, "40대": 40, "50대": 50, "60대 이상": 60}

# TRAVEL_STYL_1~8 문항: 1에 가까울수록 왼쪽, 7에 가까울수록 오른쪽을 선호 (4는 중립).
# 설문에는 SURVEY_STYLES만 나온다
STYLE_QUESTIONS = {
    1: ("자연", "도시"),
    2: ("숙박 여행", "당일 여행"),
    3: ("새로운 지역", "익숙한 지역"),
    4: ("편하지만 비싼 숙소", "불편하지만 저렴한 숙소"),
    5: ("휴양·휴식", "체험 활동"),
    6: ("잘 알려지지 않은 곳", "잘 알려진 곳"),
    7: ("계획에 따른 여행", "상황에 따른 여행"),
    8: ("사진 촬영이 중요하지 않음", "사진 촬영이 중요함"),
}

# AI Hub 코드표 MIS (여행 미션), TMT (여행 동기). 설문에는 SURVEY_MISSIONS, SURVEY_MOTIVES만 나온다
ALL_MISSIONS = {
    1: "쇼핑", 2: "테마파크·놀이시설·동식물원", 3: "역사 유적지", 4: "시티투어",
    5: "야외 스포츠·레포츠", 6: "문화예술·공연·전시 관람", 7: "유흥·오락(나이트라이프)", 8: "캠핑",
    9: "지역 축제·이벤트", 10: "온천·스파", 11: "교육·체험 프로그램", 12: "드라마 촬영지",
    13: "종교·성지 순례", 21: "웰니스 여행", 22: "SNS 인생샷 여행", 23: "호캉스",
    24: "신규 여행지 발굴", 25: "반려동물 동반", 26: "인플루언서 따라하기", 27: "친환경 여행(플로깅)",
    28: "등반 여행",
}
ALL_MOTIVES = {
    1: "일상 탈출, 지루함 탈피", 2: "휴식, 피로 해소", 3: "동반자와의 친밀감 증진",
    4: "자아 찾기, 나를 돌아보기", 5: "SNS 사진 등록 등 과시", 6: "운동, 건강 증진",
    7: "새로운 경험", 8: "역사 탐방, 문화적 경험 등 교육", 9: "특별한 목적(칠순, 신혼여행 등)", 10: "기타",
}
MISSIONS = {c: ALL_MISSIONS[c] for c in SURVEY_MISSIONS}
MOTIVES = {c: ALL_MOTIVES[c] for c in SURVEY_MOTIVES}


def region_label(sido):
    """화면에 보여 줄 시·도. 주소가 없어 모르는 장소는 '지역 미상'."""
    return sido if isinstance(sido, str) else "지역 미상"


# ---------------------------------------------------------------- 입력 도우미

def pad(text, width):
    """터미널 표시 폭 기준으로 오른쪽을 공백으로 채운다. 한글 같은 전각 문자는 두 칸을 차지해서
    글자 수로 맞추면(f'{t:<24}') 열이 어긋난다."""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - w, 0)


def ask_choice(question, options):
    """보기 중 하나를 번호로 고른다. options: 보기 문구 목록 -> 고른 위치(0부터)"""
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        ans = input("> ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(options):
            return int(ans) - 1
        print(f"  1~{len(options)} 중 번호 하나를 입력해 주세요.")


def ask_multi(question, options, max_n=None):
    """보기 여러 개를 쉼표로 고른다 ('1, 6, 12'). 엔터만 치면 없음. options: {코드: 문구} -> 고른 코드 목록.
    화면에는 1번부터 이어지는 번호를 보여 주고, 고른 번호를 원래 코드(AI Hub 코드표)로 바꿔 돌려준다."""
    limit = f", 최대 {max_n}개" if max_n else ""
    print(f"\n{question} (번호를 쉼표로 구분{limit}, 없으면 엔터)")
    codes, labels = list(options), list(options.values())
    for k in range(0, len(labels), 2):  # 두 개씩 나란히
        print("  " + "".join(f"{k + j + 1:>2}. {pad(t, 34)}" for j, t in enumerate(labels[k:k + 2])).rstrip())
    while True:
        ans = input("> ").replace(" ", "")
        if not ans:
            return []
        parts = ans.split(",")
        picked = [int(a) for a in parts if a.isdigit() and 1 <= int(a) <= len(codes)]
        if len(picked) == len(parts) and len(set(picked)) == len(picked):
            if max_n is None or len(picked) <= max_n:
                return [codes[p - 1] for p in picked]
        print(f"  보기에 있는 번호를 쉼표로 구분해 입력해 주세요{limit}.")


def ask_scale(i, left, right):
    """1~7 척도 한 문항. 엔터만 치면 중립(4)."""
    print(f"  Q{i}. {left}  1 - 2 - 3 - (4) - 5 - 6 - 7  {right}")
    while True:
        ans = input("  > ").strip()
        if not ans:
            return 4
        if ans.isdigit() and 1 <= int(ans) <= 7:
            return int(ans)
        print("    1~7 중 하나를 입력해 주세요 (엔터는 중립 4).")


def ask_visited(items):
    """가 본 곳을 이름으로 찾아 만족도와 함께 받는다. -> [(ITEM_ID, 만족도)]"""
    print("\n8. [선택] 수도권에서 가 본 관광지가 있으면 알려 주세요. 취향을 더 잘 맞힐 수 있어요.")
    print("  장소 이름 일부를 입력하면 찾아 드려요. 다 입력했으면 엔터.")
    names = items["ITEM_NM"].map(norm_name)
    visited = []
    while True:
        q = norm_name(input("\n  장소 이름 > "))
        if not q:
            return visited
        found = items[names.str.contains(q, regex=False)].sort_values("N_TRAVELERS", ascending=False).head(8)
        if found.empty:
            print("  찾는 장소가 없어요. 다른 이름으로 입력해 보세요.")
            continue
        opts = [f"{r.ITEM_NM} ({TYPE_KO[r.TYPE_CD]}, {region_label(r.SIDO)})" for r in found.itertuples()]
        k = ask_choice("  어느 곳인가요?", opts + ["이 중에 없음"])
        if k == len(found):
            continue
        if found.index[k] in [i for i, _ in visited]:  # 두 번 넣으면 이력 프로필에 두 배로 반영된다
            print(f"  이미 추가한 곳이에요: {found.iloc[k]['ITEM_NM']}")
            continue
        sat = ask_choice("  그곳은 얼마나 만족스러웠나요?", ["매우 불만족", "불만족", "보통", "만족", "매우 만족"]) + 1
        visited.append((found.index[k], sat))
        print(f"  추가했어요: {found.iloc[k]['ITEM_NM']} (만족도 {sat})")


# ---------------------------------------------------------------- 설문 -> 특성 벡터

def survey_vector(answers, columns):
    """설문 답 -> user_features와 같은 열 순서의 벡터.
    전처리의 build_users 결과와 같은 모양의 한 줄짜리 표를 만들어 build_user_features에 넣는다
    (변환 규칙을 한 곳에서만 관리하려고). 설문에 없는 항목(여행 기간, 나머지 스타일)은 자리만 채우고
    columns로 고를 때 빠진다."""
    row = {"TRAVELER_ID": "input", "GENDER": answers["gender"], "AGE_GRP": answers["age"],
           "COMPANION_GROUP": answers["companion"], "TRAVEL_DAYS": 1}
    row.update({f"TRAVEL_STYL_{i}": answers["styles"].get(i, 4) for i in STYLE_QUESTIONS})
    row.update({f"MIS_{c}": 1 for c in answers["missions"]})
    row.update({f"MOT_{c}": 1 for c in answers["motives"]})
    f = build_user_features(pd.DataFrame([row]))
    return f.set_index("TRAVELER_ID").reindex(columns=columns, fill_value=0).iloc[0].to_numpy(dtype=float)


def run_survey(items):
    print("=" * 60)
    print(" 수도권 여행지 추천 설문 (약 1분)")
    print("=" * 60)
    a = {}
    a["gender"] = ("M", "F")[ask_choice("1. 성별을 알려 주세요.", ["남성", "여성"])]
    a["age"] = list(AGE_GROUPS.values())[ask_choice("2. 연령대를 알려 주세요.", list(AGE_GROUPS))]
    a["companion"] = COMPANION_GROUPS[ask_choice("3. 누구와 여행하나요?",
                                                 [COMPANION_LABELS[g] for g in COMPANION_GROUPS])]

    print("\n4. 여행 스타일: 양쪽 중 어느 쪽에 가까운지 1~7로 답해 주세요. (엔터는 중립 4)")
    a["styles"] = {s: ask_scale(q, *STYLE_QUESTIONS[s]) for q, s in enumerate(SURVEY_STYLES, 1)}

    a["missions"] = ask_multi("5. 이번 여행에서 하고 싶은 것을 모두 골라 주세요.", MISSIONS)
    a["motives"] = ask_multi("6. 여행을 떠나는 이유를 골라 주세요.", MOTIVES, max_n=3)
    regions = list(REGIONS) + ["상관없음"]
    a["region"] = regions[ask_choice("7. 어느 지역으로 가고 싶나요?", regions)]
    a["visited"] = ask_visited(items)
    return a


# ---------------------------------------------------------------- 실행

def recommend(answers, n=10, model=None):
    """설문 답 -> 추천 표. 가 본 곳은 추천에서 빼고, 지역을 골랐으면 그 지역 장소만 후보로 둔다."""
    model = model or build_model(DEFAULT_MODEL, **DEFAULT_ADJUST, text=DEFAULT_TEXT)
    items = load_table("items").set_index("ITEM_ID")
    survey = survey_vector(answers, model.U.columns)
    known = [i for i, _ in answers["visited"]]
    sat = [s for _, s in answers["visited"]]
    cand = pd.Index(catalog())  # 수도권 장소만 ('상관없음'이어도)
    if answers["region"] in REGIONS:
        cand = cand[items.loc[cand, "SIDO"] == answers["region"]]
    return model.recommend_survey(survey, known, sat, n=n, candidates=cand)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="추천 개수")
    args = ap.parse_args()

    items = load_table("items").set_index("ITEM_ID")
    answers = run_survey(items)
    print("\n추천을 계산하고 있어요...")
    rec = recommend(answers, args.n)

    basis = (f"설문 + 가 본 곳 {len(answers['visited'])}곳" if answers["visited"] else "설문")
    print("\n" + "=" * 60)
    print(f" 추천 여행지 {len(rec)}곳 ({basis} 기준)")
    print("=" * 60)
    rec = rec.join(items["TYPE_CD"])
    for rank, r in enumerate(rec.itertuples(), 1):
        print(f" {rank:>2}. {r.ITEM_NM}  -  {TYPE_KO[r.TYPE_CD]}, {region_label(r.SIDO)}  (추천 점수 {r.SCORE:.2f})")


if __name__ == "__main__":
    main()
