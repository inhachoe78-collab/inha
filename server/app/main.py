from datetime import datetime
from typing import List, Dict, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.auth import verify_firebase_token
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

@app.get("/expenses", response_model=List[Expense])
def get_expenses(uid: str = Depends(verify_firebase_token)):
    """로그인한 사용자의 지출 내역만 가져옵니다."""
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []
    for doc in docs:
        expenses.append({"id": doc.id, **doc.to_dict()})
    return expenses

@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn, uid: str = Depends(verify_firebase_token)):
    """지출 내역을 생성하고 RAG용 벡터(Embedding)를 함께 저장합니다."""
    try:
        doc_ref = db.collection("users").document(uid).collection("expenses").document()
        # RAG 엔진을 이용해 텍스트와 임베딩 생성
        record = build_expense_rag_record(expense_in.model_dump())
        doc_ref.set(record)
        return {"id": doc_ref.id, **expense_in.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지출 생성 실패: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, uid: str = Depends(verify_firebase_token)):
    """질문에 대해 RAG 기반 답변을 제공합니다 (Firebase 토큰 필수)."""
    try:
        # rag_engine의 answer_question 호출 (uid, question 순서 엄수)
        result = answer_question(uid=uid, question=request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str, uid: str = Depends(verify_firebase_token)):
    """특정 지출 내역을 삭제합니다."""
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="해당 내역을 찾을 수 없습니다.")
    doc_ref.delete()
    return {"message": "Successfully deleted"}