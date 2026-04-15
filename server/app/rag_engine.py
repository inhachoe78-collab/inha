import os
import time
import re
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import requests

from app.firebase_client import get_firestore_client

db = get_firestore_client()

# 1. API 키 설정 (공백 제거)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# 2. 모델 및 URL 설정 (주소에서 중복/오타 가능성을 완전히 제거한 하드코딩)
GENERATION_MODEL = "gemini-1.5-flash"  # 안정성을 위해 1.5-flash 권장
GENERATE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GENERATION_MODEL}:generateContent"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent"


def expense_to_sentence(expense: Dict[str, Any]) -> str:
    return (
        f"{expense.get('date', '날짜미상')}에 {expense.get('category', '기타')} 카테고리로 "
        f"{expense.get('amount', 0)}원을 지출하였다. "
        f"결제수단은 {expense.get('payment_method', '미상')}이며, "
        f"사용처는 {expense.get('place', '알수없음')}이다. "
        f"메모: {expense.get('memo', '')}."
    )


def month_of(date_str: str) -> str:
    return date_str[:7]


def build_monthly_summary(expenses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for expense in expenses:
        date = expense.get("date", "0000-00")
        grouped[month_of(date)].append(expense)

    summaries = []
    months = sorted(grouped.keys())
    previous_total = None

    for month in months:
        items = grouped[month]
        total = sum(int(x.get("amount", 0)) for x in items)

        category_totals = defaultdict(int)
        for x in items:
            category_totals[x.get("category", "기타")] += int(x.get("amount", 0))

        if not category_totals:
            continue

        top_category = max(category_totals.items(), key=lambda x: x[1])[0]
        top_amount = category_totals[top_category]

        if previous_total is None:
            diff_text = "이전 달 데이터가 없어 증감 비교는 불가능하다."
        else:
            diff = total - previous_total
            if diff > 0:
                diff_text = f"전월 대비 총지출이 {diff}원 증가하였다."
            elif diff < 0:
                diff_text = f"전월 대비 총지출이 {abs(diff)}원 감소하였다."
            else:
                diff_text = "전월 대비 총지출 변화가 없다."

        summaries.append({
            "ref": f"summary:{month}",
            "text": (
                f"{month} 소비 요약이다. 총지출은 {total}원이다. "
                f"가장 큰 지출 카테고리는 {top_category}이며 {top_amount}원이다. {diff_text}"
            )
        })
        previous_total = total

    return summaries


def call_embed_api(text: str) -> List[float]:
    if not GEMINI_API_KEY:
        raise RuntimeError("CONFIG_ERROR: GEMINI_API_KEY가 없습니다.")

    # 404 방지: 헤더 인증 대신 쿼리 파라미터 인증 사용
    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}
    payload = {"content": {"parts": [{"text": text}]}}

    response = requests.post(EMBED_URL, params=params, headers=headers, json=payload, timeout=60)
    
    if response.status_code != 200:
        raise RuntimeError(f"EMBED_API_ERROR_{response.status_code}: {response.text}")
        
    data = response.json()
    return data["embedding"]["values"]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0


def get_user_expenses(uid: str) -> List[Dict[str, Any]]:
    docs = db.collection("users").document(uid).collection("expenses").stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


def retrieve_relevant_docs(uid: str, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    expenses = get_user_expenses(uid)
    query_embedding = call_embed_api(question)
    scored_docs = []

    for expense in expenses:
        if "embedding" not in expense or "rag_text" not in expense:
            continue
        score = cosine_similarity(query_embedding, expense["embedding"])
        scored_docs.append({"ref": f"expense:{expense['id']}", "text": expense["rag_text"], "score": score})

    summaries = build_monthly_summary(expenses)
    for summary in summaries:
        summary_embedding = call_embed_api(summary["text"])
        score = cosine_similarity(query_embedding, summary_embedding)
        scored_docs.append({"ref": summary["ref"], "text": summary["text"], "score": score})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:top_k]


def build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    context = "\n\n".join([f"[{doc['ref']}]\n{doc['text']}" for doc in docs])
    return f"""너는 개인 가계부 분석 도우미다. 아래 참고 문서만 근거로 답변해라.
내용이 없으면 "문서에서 확인되지 않습니다."라고 답해라.

[질문]
{question}

[참고 문서]
{context}
""".strip()


def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return "CONFIG_ERROR: GEMINI_API_KEY 누락"

    # 404 방지: 쿼리 파라미터 인증 및 타임아웃 설정
    params = {"key": GEMINI_API_KEY}
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GENERATE_URL, params=params, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            if response.status_code in (429, 500, 503) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
                
            return f"API_HTTP_ERROR_{response.status_code}: {response.text}"
        except Exception as e:
            if attempt < max_retries - 1: continue
            return f"SYSTEM_ERROR: {str(e)}"
    return "응답 생성 실패"


def answer_question(uid: str, question: str) -> Dict[str, Any]:
    start = time.time()
    docs = retrieve_relevant_docs(uid, question, top_k=3)
    retrieval_elapsed = time.time() - start

    prompt = build_prompt(question, docs)

    gen_start = time.time()
    answer = call_gemini(prompt)
    generation_elapsed = time.time() - gen_start

    return {
        "answer": answer,
        "references": [doc["ref"] for doc in docs],
        "retrieval_seconds": round(retrieval_elapsed, 3),
        "generation_seconds": round(generation_elapsed, 3),
        "total_seconds": round(retrieval_elapsed + generation_elapsed, 3),
    }

def build_expense_rag_record(expense_data: dict) -> dict:
    """
    지출 내역 데이터를 받아 RAG 시스템에 필요한 
    텍스트 요약(rag_text)과 임베딩(embedding)을 포함한 레코드를 생성합니다.
    """
    # 1. 지출 데이터를 문장으로 변환
    text = expense_to_sentence(expense_data)
    
    # 2. 문장을 벡터(Embedding)로 변환
    embedding = call_embed_api(text)
    
    # 3. 기존 데이터에 RAG용 정보 추가해서 반환
    record = expense_data.copy()
    record["rag_text"] = text
    record["embedding"] = embedding
    record["updated_at"] = datetime.utcnow().isoformat()
    
    return record