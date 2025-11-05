import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# 설정
# -----------------------------
input_path = "../data/v5_oag_publication_1.json"
output_path = "../data/expanded_author_features.jsonl"

# optional: multilingual 모델 (topic embedding 계산용)
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

# -----------------------------
# 1. 원본 논문 데이터 로드
# -----------------------------
authors_data = defaultdict(lambda: {
    "papers": [],
    "years": set(),
    "citations_per_year": defaultdict(int),
    "keywords": [],
})

with open(input_path, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Loading publications"):
        paper = json.loads(line)
        year = paper.get("year")
        n_citation = paper.get("n_citation", 0)
        keywords = paper.get("keywords", [])
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        for author in paper.get("authors", []):
            name = author.get("name")
            if not name:
                continue
            rec = authors_data[name]
            rec["papers"].append({"title": title, "abstract": abstract, "year": year})
            rec["years"].add(year)
            rec["citations_per_year"][year] += n_citation
            rec["keywords"].extend(keywords)

# -----------------------------
# 2. 피처 계산
# -----------------------------
expanded = []
for author, rec in tqdm(authors_data.items(), desc="Computing features"):
    years = sorted([y for y in rec["years"] if isinstance(y, int)])
    if not years:
        continue

    active_years = len(years)
    papers_per_year = {str(y): sum(1 for p in rec["papers"] if p["year"] == y) for y in years}
    citations_per_year = {str(y): rec["citations_per_year"][y] for y in years}

    # 토픽 일관성 계산: 연도별 평균 임베딩의 코사인 유사도 평균
    year_embeddings = []
    for y in years:
        texts = [p["title"] + " " + p["abstract"] for p in rec["papers"] if p["year"] == y]
        if not texts:
            continue
        emb = embed_model.encode(texts, convert_to_tensor=True, normalize_embeddings=True).mean(dim=0)
        year_embeddings.append(emb)
    if len(year_embeddings) >= 2:
        sims = [float(util.cos_sim(year_embeddings[i], year_embeddings[i + 1])) for i in range(len(year_embeddings) - 1)]
        topic_consistency = np.mean(sims)
    else:
        topic_consistency = 1.0

    # 총 피처
    obj = {
        "author": author,
        "papers_per_year": papers_per_year,
        "citations_per_year": citations_per_year,
        "active_years": active_years,
        "topic_consistency": float(topic_consistency),
        "keyword_count": len(set(rec["keywords"])),
        "total_papers": len(rec["papers"]),
        "total_citations": sum(citations_per_year.values()),
    }
    expanded.append(obj)

# -----------------------------
# 3. 저장
# -----------------------------
with open(output_path, "w", encoding="utf-8") as f:
    for row in expanded:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Saved expanded dataset → {output_path}")
print(f"총 연구자 수: {len(expanded):,}")
