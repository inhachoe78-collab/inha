from typing import List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from google.cloud import firestore
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question

#채팅 로그 저장 백그라운드 함수
def save_chat_log(user_id: str, question: str, answer: str):
    session_ref = db.collection("chat_sessions").document(user_id)

    new_message = {
        "user" : question,
        "assistant" : answer,
        "timestamp" : datetime.utcnow().isoformat()
    }

    doc = session_ref.get()
    if not doc.exists:
        session_ref.set({
            "messages" : [new_message],
            "last_updated" : datetime.utcnow(). isoformat()
        })
    else:
        session_ref.update({
            "messages": firestore.ArrayUnion([new_message]),
            "last_updated": datetime.utcnow().isoformat()
        })
app = FastAPI(title="HouseHold RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_firestore_client()
expenses_ref = db.collection("expenses")


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


@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/expenses", response_model=List[Expense])
def get_expenses():
    docs = expenses_ref.stream()
    expenses = []
    for doc in docs:
        data = doc.to_dict()
        expenses.append({
            "id": doc.id,
            **data
        })
    return expenses


@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn):
    doc_ref = expenses_ref.document()
    doc_ref.set(expense_in.model_dump())
    return {
        "id": doc_ref.id,
        **expense_in.model_dump()
    }


@app.put("/expenses/{expense_id}", response_model=Expense)
def update_expense(expense_id: str, expense_in: ExpenseIn):
    doc_ref = expenses_ref.document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    doc_ref.set(expense_in.model_dump())
    return {
        "id": expense_id,
        **expense_in.model_dump()
    }


@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str):
    doc_ref = expenses_ref.document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Expense not found")

    doc_ref.delete()
    return {"message": "deleted"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, background_tasks: BackgroundTasks):
    # 1. AI 응답 생성 (기존 로직 수행)
    answer_data = answer_question(request.question)
    
    # 2. 채팅 로그 저장 (BackgroundTasks 활용하여 응답 속도 최적화)
    # 현재는 테스트를 위해 'default_user'로 설정하지만, 
    # 나중에 로그인 구현 시 실제 user_id를 넘겨받으면 됩니다.
    background_tasks.add_task(
        save_chat_log, 
        user_id="default_user", 
        question=request.question, 
        answer=answer_data["answer"]
    )
    
    return AskResponse(
        answer=answer_data["answer"],
        references=answer_data["references"]
    )
# 과거 채팅 내역을 가져오는 API (안드로이드에서 호출용)
@app.get("/chat/history/{user_id}")
def get_chat_history(user_id: str):
    doc_ref = db.collection("chat_sessions").document(user_id)
    doc = doc_ref.get()
    
    if not doc.exists:
        return {"messages": []}
    
    return doc.to_dict()