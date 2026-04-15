from datetime import datetime
from typing import List, Dict, Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # 추가
from pydantic import BaseModel
from google.cloud import firestore

from app.auth import verify_firebase_token
from app.firebase_client import get_firestore_client
from app.rag_engine import answer_question, build_expense_rag_record

# 1. 보안 스킴 정의 (Swagger 자물쇠용)
auth_scheme = HTTPBearer()

app = FastAPI(
    title="HouseHold RAG API",
    # Swagger UI에서 새로고침해도 인증이 유지되도록 설정
    swagger_ui_parameters={"persistAuthorization": True}
)

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

@app.get("/expenses", response_model=List[Expense])
def get_expenses(
    uid: str = Depends(verify_firebase_token),
    token: HTTPAuthorizationCredentials = Depends(auth_scheme) # 자물쇠 활성화
):
    docs = db.collection("users").document(uid).collection("expenses").stream()
    expenses = []
    for doc in docs:
        expenses.append({"id": doc.id, **doc.to_dict()})
    return expenses

@app.post("/expenses", response_model=Expense)
def create_expense(
    expense_in: ExpenseIn, 
    uid: str = Depends(verify_firebase_token),
    token: HTTPAuthorizationCredentials = Depends(auth_scheme) # 자물쇠 활성화
):
    try:
        doc_ref = db.collection("users").document(uid).collection("expenses").document()
        record = build_expense_rag_record(expense_in.model_dump())
        doc_ref.set(record)
        return {"id": doc_ref.id, **expense_in.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest, 
    uid: str = Depends(verify_firebase_token),
    token: HTTPAuthorizationCredentials = Depends(auth_scheme) # 자물쇠 활성화
):
    try:
        result = answer_question(uid=uid, question=request.question)
        
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
def delete_expense(
    expense_id: str, 
    uid: str = Depends(verify_firebase_token),
    token: HTTPAuthorizationCredentials = Depends(auth_scheme) # 자물쇠 활성화
):
    doc_ref = db.collection("users").document(uid).collection("expenses").document(expense_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="해당 내역을 찾을 수 없습니다.")
    doc_ref.delete()
    return {"message": "Successfully deleted"}