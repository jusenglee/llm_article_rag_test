import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

# ----------------------------
# 1. 파일 로드
# ----------------------------
path = Path("author_features.jsonl")  # 데이터 파일 경로
records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except Exception:
            continue

df = pd.DataFrame(records)
print(f"✅ Loaded {len(df):,} records")

# ----------------------------
# 2. 기본 통계 점검
# ----------------------------
print("\n📊 기본 통계 요약:")
print(df.describe(include="all").T)

# ----------------------------
# 3. 결측치 및 상수형 피처 점검
# ----------------------------
print("\n⚠️ 결측치 비율:")
print(df.isna().mean().sort_values(ascending=False))

constant_cols = []
for c in df.columns:
    # dict나 list 타입은 제외
    if df[c].apply(lambda x: isinstance(x, (dict, list))).any():
        continue
    try:
        if df[c].nunique() == 1:
            constant_cols.append(c)
    except Exception:
        continue

if constant_cols:
    print(f"\n⚠️ 상수값(모두 동일) 컬럼: {constant_cols}")
else:
    print("\n✅ 상수형 컬럼 없음")

# ----------------------------
# 4. 연도 다양성 / 활동 연수 분포
# ----------------------------
df["active_years"] = df["active_years"].astype(float)
print("\n📈 활동 연도 분포:")
print(df["active_years"].value_counts().head())

plt.hist(df["active_years"], bins=range(1, 10))
plt.title("Active Years Distribution")
plt.xlabel("Active Years")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 5. 공동저자 분포 / 네트워크 중심성 값
# ----------------------------
if "coauthor_count" in df.columns:
    plt.hist(df["coauthor_count"], bins=30)
    plt.title("Coauthor Count Distribution")
    plt.xlabel("coauthor_count")
    plt.ylabel("Count")
    plt.show()

if "betweenness_centrality" in df.columns:
    unique_bc = df["betweenness_centrality"].nunique()
    zero_ratio = (df["betweenness_centrality"] == 0).mean()
    print(f"🕸 betweenness_centrality 고유값 수: {unique_bc}, 0 비율: {zero_ratio:.2%}")

# ----------------------------
# 6. 인용수/영향력 피처 분석
# ----------------------------
if "impact_velocity" in df.columns:
    print(f"\nImpact Velocity: mean={df['impact_velocity'].mean():.3f}, std={df['impact_velocity'].std():.3f}")

if "recency_weighted_score" in df.columns:
    print(f"Recency Weighted Score: unique={df['recency_weighted_score'].nunique()}")

# ----------------------------
# 7. 토픽 일관성 분석
# ----------------------------
if "topic_consistency" in df.columns:
    print(f"\nTopic Consistency unique values: {df['topic_consistency'].nunique()}")
    if df["topic_consistency"].nunique() == 1:
        print("⚠️ 모든 연구자 topic_consistency가 동일함 → 임베딩 계산 안 됐을 가능성")

# ----------------------------
# 8. 인용 분포 및 로그 스케일 확인
# ----------------------------
cit_counts = []
for d in df["citations_per_year"]:
    if isinstance(d, dict):
        cit_counts.append(sum(d.values()))
df["total_citations"] = cit_counts
plt.hist(np.log1p(df["total_citations"]), bins=40)
plt.title("Log-Scaled Total Citations Distribution")
plt.xlabel("log(1+citations)")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 9. 중복 저자 이름 점검
# ----------------------------
dupes = [name for name, count in Counter(df["author"]).items() if count > 1]
print(f"\n🧍 중복된 author 이름 수: {len(dupes)}")

# ----------------------------
# 10. 요약 평가 출력
# ----------------------------
print("\n✅ 품질 점검 완료 요약:")
print(f"- 총 저자 수: {len(df):,}")
print(f"- 중복 이름 수: {len(dupes):,}")
print(f"- 평균 활동연도: {df['active_years'].mean():.2f}")
print(f"- 평균 공동저자 수: {df['coauthor_count'].mean():.2f}" if "coauthor_count" in df else "- 공동저자 피처 없음")
print(f"- topic_consistency 상수화 여부: {'Yes' if df['topic_consistency'].nunique() == 1 else 'No'}")
print(f"- impact_velocity 평균: {df['impact_velocity'].mean():.3f}")
