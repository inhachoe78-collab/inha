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

# 1. 모델 설정 (v1 정식 버전 주소 체계에 맞게 고정)
GENERATION_MODEL = "gemini-1.5-flash"

def load_expenses() -> List[Dict[str, Any]]:
    try:
        docs = expenses_ref.stream()
        expenses = []
        for doc in docs:
            data = doc.to_dict()
            expenses.append({"id": doc.id, **data})
        return expenses
    except Exception as e:
        print(f"FIREBASE_LOAD_ERROR: {e}")
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
    # Render 환경 변수 체크
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return "CONFIG_ERROR: Render 환경변수에 GEMINI_API_KEY가 없습니다."

    # 🔥 핵심 수정: 에러 메시지의 권고에 따라 v1beta -> v1 으로 변경
    url = f"https://generativelanguage.googleapis.com/v1/models/{GENERATION_MODEL}:generateContent"
    
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(
            url, 
            params=params, 
            headers=headers, 
            json=payload, 
            timeout=30
        )
        
        if response.status_code != 200:
            # 여기서 404가 또 뜨면 모델명이 아예 틀렸거나 리전 문제입니다.
            return f"API_HTTP_ERROR_{response.status_code}: {response.text}"

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "API_EMPTY_RESPONSE: 답변 후보가 없습니다."

        return candidates[0]["content"]["parts"][0]["text"].strip()

    except Exception as e:
        return f"SYSTEM_EXCEPTION: {str(e)}"

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
    context_data = "\n\n".join([f"[{d['ref']}] {d['text']}" for d in docs])
    
    prompt = f"""당신은 가계부 분석 도우미입니다.
참고 데이터를 바탕으로 답변하세요.

[참고 데이터]
{context_data}

[이전 맥락]
{history}

질문: {question}"""

    answer = call_gemini(prompt)
    return {
        "answer": answer,
        "references": [d["ref"] for d in docs]
    }