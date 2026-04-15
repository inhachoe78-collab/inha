from datetime import datetime
from typing import List, Dict, Any

from fastapi import Depends, FastAPI, HTTPException # Depends 추가
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.cloud import firestore

from app.auth import verify_firebase_token # 인증 함수 임포트
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

# --- Pydantic 모델 정의 (생략 없음) ---
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

# --- API 엔드포인트 (인증 적용) ---

@app.get("/")
def root():
    return {"message": "HouseHold RAG server is running"}

@app.get("/expenses", response_model=List[Expense])
def get_expenses(uid: str = Depends(verify_firebase_token)):
    """로그인한 유저(uid)의 지출 내역만 가져옵니다."""
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []
    for doc in docs:
        expenses.append({"id": doc.id, **doc.to_dict()})
    return expenses

@app.post("/expenses", response_model=Expense)
def create_expense(expense_in: ExpenseIn, uid: str = Depends(verify_firebase_token)):
    """로그인한 유저(uid)의 경로에 지출 내역을 생성합니다."""
    try:
        doc_ref = db.collection("users").document(uid).collection("expenses").document()
        record = build_expense_rag_record(expense_in.model_dump())
        doc_ref.set(record)
        return {"id": doc_ref.id, **expense_in.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest, uid: str = Depends(verify_firebase_token)):
    """로그인한 유저(uid)의 데이터를 기반으로 질문하고 기록을 저장합니다."""
    try:
        result = answer_question(uid=uid, question=request.question)
        
        # 채팅 로그 저장 (유저별 세션 구분)
        chat_ref = db.collection("chat_sessions").document(uid)
        new_message = {
            "user": request.question,
            "assistant": result["answer"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        doc = chat_ref.get()
        if not doc.exists:
            chat_ref.set({"messages": [new_message], "last_updated": datetime.utcnow().isoformat()})
        else:
            chat_ref.update({
                "messages": firestore.ArrayUnion([new_message]),
                "last_updated": datetime.utcnow().isoformat()
            })
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/expenses/{expense_id}")
def delete_expense(expense_id: str, uid: str = Depends(verify_firebase_token)):
    """본인의 지출 내역만 삭제 가능합니다."""
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="해당 내역을 찾을 수 없습니다.")
    doc_ref.delete()
    return {"message": "Successfully deleted"}