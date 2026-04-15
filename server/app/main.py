from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# [수정] 인증 모듈이 없으므로 import 제거
# from app.auth import verify_firebase_token
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
def get_expenses():
    """[수정] 인증 없이 default_user의 내역을 가져옵니다."""
    uid = "default_user"
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []
    for doc in docs:
        expenses.append({"id": doc.id, **doc.to_dict()})
    return expenses

@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn):
    """[수정] 지출 내역 생성 시 RAG 임베딩을 포함합니다."""
    uid = "default_user"
    try:
        doc_ref = db.collection("users").document(uid).collection("expenses").document()
        # 한글 파일 내용 반영: build_expense_rag_record를 통해 임베딩 포함 record 생성
        record = build_expense_rag_record(expense_in.model_dump())
        doc_ref.set(record)
        return {"id": doc_ref.id, **expense_in.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"지출 생성 실패: {str(e)}")

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """[수정] 질문에 대해 RAG 기반 답변을 제공합니다."""
    uid = "default_user"
    try:
        # rag_engine의 answer_question 호출
        result = answer_question(uid=uid, question=request.question)
        return result
    except Exception as e:
        # 에러 발생 시 상세 원인을 반환하여 디버깅 용이하게 함
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str):
    """[수정] 특정 지출 내역을 삭제합니다."""
    uid = "default_user"
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="해당 내역을 찾을 수 없습니다.")
    doc_ref.delete()
    return {"message": "Successfully deleted"}