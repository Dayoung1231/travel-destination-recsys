# References

과제 명세서 요구 사항:
- 인터넷(블로그, Kaggle, GitHub 등)의 코드를 쓰면 **반드시 출처를 밝혀요.** 밝히지 않으면 표절로 간주돼요.
- 수업에서 다루지 않은 모듈·클래스·라이브러리·함수·메서드는 **파라미터까지 설명**해요.

코드를 추가할 때마다 바로 적어 주세요. 제출 직전에 몰아서 쓰면 빠뜨리기 쉬워요.

## 1. Code sources

| Where it is used | Source (URL) | What was taken / changed | Added by |
|---|---|---|---|
| `src/modeling/mf.py` (example) | https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/ | SGD update rule, rewritten for our data | |

## 2. Libraries and functions not covered in class

| Function / class | Used in | What it does | Parameters we use |
|---|---|---|---|
| `sklearn.feature_extraction.text.TfidfVectorizer` (example) | `src/preprocessing/features.py` | Turns place descriptions into TF-IDF vectors | `max_features`: vocabulary size limit, `min_df`: ignore terms in fewer documents |

## 3. Datasets

| Data | Source | License / terms |
|---|---|---|
| Domestic Travel Log (Seoul Capital Area) (2023) | AI Hub, https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71776 | AI Hub terms: AI training use only, no sharing with unapproved parties, attribution to NIA required |
| TourAPI Korean Tourism Information Service | Korea Tourism Organization, https://www.data.go.kr/data/15101578/openapi.do | No usage restrictions (images excluded) |
| ASOS Daily Weather Data | KMA, https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36 | KOGL Type 1 (attribution) |

**AI Hub attribution** (use in reports and presentations):
>
> This project uses the "Domestic Travel Log Data (Seoul Capital Area) (2023)" from AI Hub, a result of a project by the National Information Society Agency (NIA) of Korea.
