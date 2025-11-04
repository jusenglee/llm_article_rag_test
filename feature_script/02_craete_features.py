import json
from collections import defaultdict
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "../data/aminer_clean_sample_10000.jsonl"

# ---------------------------
# 데이터 로드 및 연도별 분류
# ---------------------------
authors = defaultdict(lambda: defaultdict(list))  # author -> year -> papers[]
coauthor_graph = nx.Graph()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        year = rec.get("year")
        if not year:
            continue
        kws = rec.get("keywords", [])
        cites = rec.get("n_citation", 0)
        authors_in_paper = [a if isinstance(a, str) else a.get("name") for a in rec.get("authors", [])]
        for a in authors_in_paper:
            authors[a][year].append({
                "keywords": kws,
                "citations": cites,
                "coauthors": [c for c in authors_in_paper if c != a]
            })
        # coauthor network
        for i in range(len(authors_in_paper)):
            for j in range(i+1, len(authors_in_paper)):
                coauthor_graph.add_edge(authors_in_paper[i], authors_in_paper[j])

print(f"Loaded {len(authors)} authors")

# ---------------------------
# 피처 계산
# ---------------------------
model = SentenceTransformer("intfloat/multilingual-e5-large")

def safe_convert(o):
    if isinstance(o, (np.float32, np.float64)):
        return float(o)
    if isinstance(o, set):
        return list(o)
    return o

def topic_vector(keywords):
    if not keywords:
        return np.zeros((1024,))
    text = " ".join(keywords)
    return model.encode(text)

results = {}
for author, year_dict in authors.items():
    years = sorted(year_dict.keys())
    paper_count = sum(len(v) for v in year_dict.values())
    citations_per_year = {y: sum(p["citations"] for p in v) for y, v in year_dict.items()}
    papers_per_year = {y: len(v) for y, v in year_dict.items()}
    active_years = len(years)

    # topic consistency
    year_vecs = [topic_vector(sum((p["keywords"] for p in v), [])) for y, v in sorted(year_dict.items())]
    topic_sim = np.mean([cosine_similarity([year_vecs[i]], [year_vecs[i+1]])[0][0]
                         for i in range(len(year_vecs)-1)]) if len(year_vecs) > 1 else 1.0

    # coauthor metrics
    all_coauthors = set(sum((p["coauthors"] for v in year_dict.values() for p in v), []))
    coauthor_count = len(all_coauthors)

    # 신규 공동저자 비율
    seen, new_ratios = set(), []
    for y in years:
        year_coauthors = set(sum((p["coauthors"] for p in year_dict[y]), []))
        new = year_coauthors - seen
        ratio = len(new) / len(year_coauthors) if year_coauthors else 0
        new_ratios.append(ratio)
        seen |= year_coauthors
    new_coauthor_ratio = np.mean(new_ratios)

    # impact velocity
    latest = max(years)
    recent = [c for y, c in citations_per_year.items() if y >= latest - 2]
    impact_velocity = sum(recent) / sum(citations_per_year.values()) if sum(citations_per_year.values()) else 0

    # recency weighted score
    recency_weighted_score = sum(
        papers_per_year[y] / (1 + (latest - y)) for y in years
    )

    results[author] = {
        "papers_per_year": papers_per_year,
        "citations_per_year": citations_per_year,
        "active_years": active_years,
        "topic_consistency": topic_sim,
        "coauthor_count": coauthor_count,
        "new_coauthor_ratio": new_coauthor_ratio,
        "impact_velocity": impact_velocity,
        "recency_weighted_score": recency_weighted_score
    }

# ---------------------------
# 네트워크 중심성 계산
# ---------------------------
centrality = nx.betweenness_centrality(coauthor_graph)
for a, c in centrality.items():
    if a in results:
        results[a]["betweenness_centrality"] = c

print(f"Calculated features for {len(results)} authors")

# ---------------------------
# 저장
# ---------------------------
with open("author_features.jsonl", "w", encoding="utf-8") as f:
    for k, v in results.items():
        obj = {"author": k, **v}
        f.write(json.dumps(obj, default=safe_convert, ensure_ascii=False) + "\n")

print("✅ Saved author_features.jsonl")
