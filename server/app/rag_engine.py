import re
import time
from collections import defaultdict
from typing import Any, Dict, List

import requests

from app.firebase_client import get_firestore_client

db = get_firestore_client()
expenses_ref = db.collection("expenses")

GEMINI_API_KEY = __import__("os").environ.get("GEMINI_API_KEY", "")
GENERATION_MODEL = "gemini-1.5-flash"
GENERATE_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GENERATION_MODEL}:generateContent"
)


def load_expenses() -> List[Dict[str, Any]]:
    docs = expenses_ref.stream()
    expenses = []
    for doc in docs:
        data = doc.to_dict()
        expenses.append({
            "id": doc.id,
            **data
        })
    return expenses


def expense_to_sentence(expense: Dict[str, Any]) -> str:
    return (
        f"{expense['date']}에 {expense['category']} 카테고리로 "
        f"{expense['amount']}원을 지출하였다. "
        f"결제수단은 {expense['payment_method']}이며, "
        f"사용처는 {expense['place']}이다. "
        f"메모: {expense['memo']}."
    )


def month_of(date_str: str) -> str:
    return date_str[:7]


def build_monthly_summary(expenses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    grouped = defaultdict(list)
    for expense in expenses:
        grouped[month_of(expense["date"])].append(expense)

    summaries = []
    months = sorted(grouped.keys())
    previous_total = None

    for month in months:
        items = grouped[month]
        total = sum(int(x["amount"]) for x in items)

        category_totals = defaultdict(int)
        for x in items:
            category_totals[x["category"]] += int(x["amount"])

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
                f"{month} 소비 요약이다. "
                f"총지출은 {total}원이다. "
                f"가장 큰 지출 카테고리는 {top_category}이며 해당 지출은 {top_amount}원이다. "
                f"{diff_text}"
            )
        })
        previous_total = total

    return summaries


def build_rag_documents(expenses: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    docs = []
    for expense in expenses:
        docs.append({
            "ref": f"expense:{expense['id']}",
            "text": expense_to_sentence(expense)
        })

    docs.extend(build_monthly_summary(expenses))
    return docs


def normalize_tokens(text: str) -> List[str]:
    lowered = text.lower()
    return re.findall(r"[가-힣a-zA-Z0-9]+", lowered)


def extract_month_hint(question: str) -> str:
    m = re.search(r"([1-9]|1[0-2])월", question)
    if m:
        month_num = int(m.group(1))
        return f"-{month_num:02d}"
    return ""


def score_document(question: str, doc_text: str) -> int:
    q_tokens = set(normalize_tokens(question))
    d_tokens = set(normalize_tokens(doc_text))
    score = len(q_tokens & d_tokens)

    month_hint = extract_month_hint(question)
    if month_hint and month_hint in doc_text:
        score += 3

    return score


def retrieve_relevant_docs(question: str, expenses: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, str]]:
    docs = build_rag_documents(expenses)
    scored = [(score_document(question, doc["text"]), doc) for doc in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def build_prompt(question_with_history: str, docs: List[Dict[str, str]]) -> str:
    context = "\n\n".join([f"[{doc['ref']}]\n{doc['text']}" for doc in docs])

    return f"""
너는 개인 가계부 소비 분석 도우미다.
반드시 아래 제공된 [참고 문서]와 [이전 대화 맥락]을 근거로 답변해라. # 수정됨
문서나 대화 기록에 없는 내용은 추측하지 말고 "확인되지 않습니다."라고 답해라.

[질문 및 맥락]
{question_with_history}

[참고 문서]
{context}
...
""".strip()

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }

    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        response = requests.post(
            GENERATE_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "응답을 생성하지 못했습니다."

            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [part.get("text", "") for part in parts if "text" in part]
            return "\n".join(texts).strip()

        if response.status_code in (429, 500, 503) and attempt < max_retries - 1:
            time.sleep(delay)
            delay *= 2
            continue

        response.raise_for_status()

    return "응답을 생성하지 못했습니다."


# 수정 후 (이 코드로 덮어쓰세요)
def answer_question(question: str, user_id: str = "default_user") -> Dict[str, Any]:
    expenses = load_expenses()
    docs = retrieve_relevant_docs(question, expenses, top_k=4)
    
    # 1. 이전 대화 맥락 가져오기
    history_context = get_recent_context(user_id) 
    
    # 2. 질문에 이전 맥락을 합쳐서 프롬프트 생성
    prompt = build_prompt(question + history_context, docs) 
    
    answer = call_gemini(prompt)

    return {
        "answer": answer,
        "references": [doc["ref"] for doc in docs]
    }

  # app/rag_engine.py

def get_recent_context(user_id: str, limit: int = 3) -> str:
    """DB에서 대화 기록을 안전하게 읽어오는 함수"""
    try:
        # DB 연결 확인
        doc_ref = db.collection("chat_sessions").document(user_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            print(f"DEBUG: {user_id}의 세션이 존재하지 않습니다.")
            return ""
        
        # 데이터 가져오기 및 에러 방지를 위한 get() 사용
        data = doc.to_dict()
        messages = data.get("messages", [])
        
        if not messages:
            return ""

        recent = messages[-limit:] # 최근 대화만 추출
        
        context_str = "\n[이전 대화 맥락]\n"
        for msg in recent:
            # Firebase 스크린샷의 키값인 'user'와 'assistant'를 정확히 사용합니다.
            u_text = msg.get('user', '')
            a_text = msg.get('assistant', '')
            context_str += f"사용자: {u_text}\nAI: {a_text}\n"
            
        print(f"DEBUG: 맥락 불러오기 성공")
        return context_str
    except Exception as e:
        # 에러 발생 시 서버가 죽지 않도록 빈 문자열 반환 및 로그 출력
        print(f"DEBUG ERROR (get_recent_context): {e}")
        return ""