from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from google.cloud import firestore

# Firebase 및 RAG 엔진 임포트
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question

# [중요] DB 초기화 위치 (임포트 직후, app 생성 전)
db = get_firestore_client()

# 채팅 로그 저장 함수 (동기 방식으로 변경하여 에러 확인 용이하게 수정)
def save_chat_log(user_id: str, question: str, answer: str):
    try:
        session_ref = db.collection("chat_sessions").document(user_id)
        new_message = {
            "user": question,
            "assistant": answer,
            "timestamp": datetime.utcnow().isoformat()
        }

        doc = session_ref.get()
        if not doc.exists:
            session_ref.set({
                "messages": [new_message],
                "last_updated": datetime.utcnow().isoformat()
            })
        else:
            session_ref.update({
                "messages": firestore.ArrayUnion([new_message]),
                "last_updated": datetime.utcnow().isoformat()
            })
        print(f"DEBUG: {user_id}의 대화가 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"DEBUG ERROR: Firebase 저장 중 에러 발생: {e}")
        # 에러 발생 시 HTTPException을 발생시켜 Swagger에서 바로 확인 가능하게 함
        raise HTTPException(status_code=500, detail=f"Firebase Save Error: {str(e)}")

app = FastAPI(title="HouseHold RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

expenses_ref = db.collection("expenses")

# --- 모델 정의 영역 ---
class ExpenseIn(BaseModel):
    date: str
    category: str
    amount: int
    payment_method: str
    place: str
    memo: str

class Expense(ExpenseIn):
    id: str

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    references: List[str]

# --- API 엔드포인트 영역 ---
@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}

@app.get("/expenses", response_model=List[Expense])
def get_expenses():
    docs = expenses_ref.stream()
    expenses = []
    for doc in docs:
        data = doc.to_dict()
        expenses.append({"id": doc.id, **data})
    return expenses

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    # 1. AI 응답 생성 (user_id 전달 추가)
    answer_data = answer_question(request.question, user_id="default_user")
    
    # 2. 채팅 로그 저장 (BackgroundTasks를 쓰지 않고 직접 호출)
    save_chat_log(
        user_id="default_user", 
        question=request.question, 
        answer=answer_data["answer"]
    )
    
    return AskResponse(
        answer=answer_data["answer"],
        references=answer_data["references"]
    )

@app.get("/chat/history/{user_id}")
def get_chat_history(user_id: str):
    doc_ref = db.collection("chat_sessions").document(user_id)
    doc = doc_ref.get()
    if not doc.exists:
        return {"messages": []}
    return doc.to_dict()