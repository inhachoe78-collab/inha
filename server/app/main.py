from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import firestore  # 채팅 저장을 위해 필수

from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question, build_expense_rag_record

app = FastAPI(title="HouseHold RAG API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = get_firestore_client()

# --- Pydantic 모델 정의 ---
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
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float

# --- API 엔드포인트 ---

@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# 1. 지출 내역 가져오기
@app.get("/expenses", response_model=List[Expense])
def get_expenses():
    uid = "default_user"
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []
    for doc in docs:
        expenses.append({"id": doc.id, **doc.to_dict()})
    return expenses

# 2. 지출 내역 생성 (RAG 임베딩 포함)
@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn):
    uid = "default_user"
    try:
        doc_ref = db.collection("users").document(uid).collection("expenses").document()
        record = build_expense_rag_record(expense_in.model_dump())
        doc_ref.set(record)
        return {"id": doc_ref.id, **expense_in.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지출 생성 실패: {str(e)}")

# 3. 질문하기 및 채팅 세션 유저별 저장 (목표 항목 3)
@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    uid = "default_user" # 유저별 구분 저장
    try:
        # RAG 답변 생성
        result = answer_question(uid=uid, question=request.question)
        
        # Firestore에 채팅 로그 저장
        chat_ref = db.collection("chat_sessions").document(uid)
        new_message = {
            "user": request.question,
            "assistant": result["answer"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        doc = chat_ref.get()
        if not doc.exists:
            chat_ref.set({
                "messages": [new_message],
                "last_updated": datetime.utcnow().isoformat()
            })
        else:
            chat_ref.update({
                "messages": firestore.ArrayUnion([new_message]),
                "last_updated": datetime.utcnow().isoformat()
            })
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. 지출 내역 삭제
@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str):
    uid = "default_user"
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="해당 내역을 찾을 수 없습니다.")
    doc_ref.delete()
    return {"message": "Successfully deleted"}