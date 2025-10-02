# api/auth.py

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

# --- Configuration ---
# In a real application, load this from an environment variable
SECRET_KEY = "a-very-secret-key-that-you-should-change"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Password Hashing ---
# This sets up the password hashing scheme
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 Scheme ---
# This tells FastAPI where the client can get the token (the /token endpoint)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- In-memory User Database (for demonstration) ---
# In a production app, you would fetch users from a real database.
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "hashed_password": pwd_context.hash("admin123"),
        "role": "admin",
    },
    "KA01AB1234": {
        "username": "KA01AB1234", # Public user is identified by their license plate
        "full_name": "Public User",
        "hashed_password": pwd_context.hash("public123"),
        "role": "public",
    }
}

def get_user(db, username: str):
    """Fetches a user from the mock database."""
    if username in db:
        return db[username]
    return None

def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a hashed one."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a new JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_active_user(token: str = Depends(oauth2_scheme)):
    """
    A dependency for protected routes. It decodes the JWT from the request's
    Authorization header, validates it, and returns the user's data.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # You could also add token scopes or other claims here
        
    except JWTError:
        raise credentials_exception
    
    user = get_user(FAKE_USERS_DB, username)
    if user is None:
        raise credentials_exception
        
    return user