import re
import time
import os
from collections import defaultdict
from typing import Any, Dict, List

import requests
from app.firebase_client import get_firestore_client

# Firestore 초기화
db = get_firestore_client()
expenses_ref = db.collection("expenses")

# 모델 및 URL 설정 (가장 보수적이고 표준적인 경로)
GENERATION_MODEL = "gemini-1.5-flash"
# f-string 대신 고정 문자열을 사용하여 미세한 오타나 공백을 방지합니다.
GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def load_expenses() -> List[Dict[str, Any]]:
    try:
        docs = expenses_ref.stream()
        expenses = []
        for doc in docs:
            data = doc.to_dict()
            expenses.append({"id": doc.id, **data})
        return expenses
    except Exception as e:
        print(f"ERROR (load_expenses): {e}")
        return []

def expense_to_sentence(expense: Dict[str, Any]) -> str:
    return (
        f"{expense.get('date', '날짜미상')}에 {expense.get('category', '기타')} 카테고리로 "
        f"{expense.get('amount', 0)}원을 지출하였다. "
        f"사용처는 {expense.get('place', '알수없음')}이며, 메모: {expense.get('memo', '')}."
    )

def build_monthly_summary(expenses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for expense in expenses:
        date = expense.get("date", "0000-00")
        month = date[:7]
        grouped[month].append(expense)

    summaries = []
    for month, items in grouped.items():
        total = sum(int(x.get("amount", 0)) for x in items)
        summaries.append({
            "ref": f"summary:{month}",
            "text": f"{month} 소비 요약: 총 {total}원을 지출함."
        })
    return summaries

def build_rag_documents(expenses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    docs = []
    for expense in expenses:
        docs.append({"ref": f"expense:{expense['id']}", "text": expense_to_sentence(expense)})
    docs.extend(build_monthly_summary(expenses))
    return docs

def retrieve_relevant_docs(question: str, expenses: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, str]]:
    docs = build_rag_documents(expenses)
    def get_score(q, d):
        q_set = set(re.findall(r"[가-힣a-zA-Z0-9]+", q.lower()))
        d_set = set(re.findall(r"[가-힣a-zA-Z0-9]+", d.lower()))
        return len(q_set & d_set)
    
    scored = [(get_score(question, doc["text"]), doc) for doc in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

def call_gemini(prompt: str) -> str:
    # 1. API 키를 매번 함수 실행 시점에 가져와서 신선도 유지
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "에러: GEMINI_API_KEY가 설정되지 않았습니다."

    # 2. 404 방지를 위해 고정된 표준 URL 사용
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    # 3. 쿼리 파라미터 방식으로 키 전달 (헤더 방식보다 404/403에 더 강함)
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, params=params, headers=headers, json=payload, timeout=30)
        
        # 상세 에러 메시지 캡처
        if response.status_code != 200:
            return f"에러 발생 (상태코드 {response.status_code}): {response.text}"

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "응답 후보가 없습니다."

        return candidates[0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        return f"요청 중 예외 발생: {str(e)}"

def get_recent_context(user_id: str, limit: int = 3) -> str:
    try:
        doc = db.collection("chat_sessions").document(user_id).get()
        if not doc.exists: return ""
        messages = doc.to_dict().get("messages", [])[-limit:]
        context = "\n[이전 대화 맥락]\n"
        for m in messages:
            context += f"사용자: {m.get('user', '')}\nAI: {m.get('assistant', '')}\n"
        return context
    except:
        return ""

def answer_question(question: str, user_id: str = "default_user") -> Dict[str, Any]:
    expenses = load_expenses()
    docs = retrieve_relevant_docs(question, expenses)
    history = get_recent_context(user_id)
    
    # 프롬프트 구성
    full_context = f"{history}\n질문: {question}"
    context_data = "\n\n".join([f"[{d['ref']}] {d['text']}" for d in docs])
    
    prompt = f"""너는 가계부 도우미다. 아래 정보를 근거로 답변해라.
[참고 데이터]
{context_data}

[질문 및 맥락]
{full_context}
""".strip()

    answer = call_gemini(prompt)
    return {
        "answer": answer,
        "references": [d["ref"] for d in docs]
    }