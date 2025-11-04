import json
import random
from tqdm import tqdm

# ---------------------------
# 기본 설정
# ---------------------------
INPUT_FILE = "../data/v5_oag_publication_1.json"
OUTPUT_FILE = "../data/aminer_clean_sample_10000.jsonl"
TARGET_COUNT = 10_000
SEED = 42

random.seed(SEED)

# ---------------------------
# 필드 추출 함수
# ---------------------------
def extract_paper_info(rec):
    """논문 데이터에서 주요 필드만 추출"""
    return {
        "paper_id": rec.get("id"),
        "title": rec.get("title"),
        "abstract": rec.get("abstract"),
        "year": rec.get("year"),
        "keywords": rec.get("keywords", []),
        "n_citation": rec.get("n_citation", 0),
        "doi": rec.get("doi"),
        "venue": rec.get("venue"),
        "authors": [
            {
                "author_id": a.get("id"),
                "name": a.get("name"),
                "org": a.get("org"),
            }
            for a in rec.get("authors", [])
            if a.get("name")
        ],
        "coauthors": [
            a.get("name") for a in rec.get("authors", [])
            if a.get("name")
        ],
        "references": rec.get("references", []),
    }

# ---------------------------
# 메인 처리 로직
# ---------------------------
def main():
    sample_data = []
    total_count = 0

    print(f"🔍 원본 파일 로드 중: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 필터 조건 (옵션)
            if not rec.get("year") or not rec.get("title"):
                continue
            if len(rec.get("authors", [])) == 0:
                continue

            # 정제
            clean = extract_paper_info(rec)
            sample_data.append(clean)
            total_count += 1

            if len(sample_data) >= TARGET_COUNT:
                break

    print(f"✅ 총 {len(sample_data)}건 추출 완료")

    # ---------------------------
    # 결과 저장
    # ---------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for obj in sample_data:
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"💾 결과 저장 완료 → {OUTPUT_FILE}")
    print(f"📊 처리된 총 라인 수: {total_count:,}")

if __name__ == "__main__":
    main()
