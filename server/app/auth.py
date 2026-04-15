from fastapi import Header, HTTPException
from firebase_admin import auth

def verify_firebase_token(authorization: str = Header(None)):
    """
    Header에서 'Bearer <Token>'을 읽어 Firebase UID를 반환합니다.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="토큰이 없거나 잘못되었습니다.")
    
    token = authorization.split("Bearer ")[1]
    try:
        # 가이드된 idToken을 검증하여 유저의 고유 UID를 추출
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"유효하지 않은 토큰입니다: {str(e)}")