from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, Union
from pydantic import BaseModel, EmailStr
from .config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from .database import get_db
from .models import User
from app.logger import logger

# Configure password hashing and OAuth2
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

class Token(BaseModel):
    """Token response model for authentication."""
    access_token: str
    token_type: str = "bearer"  # Default to bearer token

class TokenData(BaseModel):
    """Token data model for decoded JWT tokens."""
    username: Optional[str] = None

class UserCreate(BaseModel):
    """User creation request model with validation."""
    username: str
    email: EmailStr  # Use EmailStr for email validation
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "johndoe@example.com",
                "password": "strongpassword123"
            }
        }

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: The password to verify
        hashed_password: The hashed password to compare against
    
    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error("Password verification failed", exc_info=True)
        return False

def get_password_hash(password: str) -> str:
    """
    Generate password hash using bcrypt.
    
    Args:
        password: Plain text password
    
    Returns:
        str: Hashed password
    
    Raises:
        HTTPException: If hashing fails
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        logger.error("Password hashing failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password processing failed"
        )

def get_user(db: Session, username: str) -> Optional[User]:
    """
    Retrieve user from database by username.
    
    Args:
        db: Database session
        username: Username to look up
    
    Returns:
        Optional[User]: User if found, None otherwise
    
    Raises:
        HTTPException: If database query fails
    """
    try:
        return db.query(User).filter(User.username == username).first()
    except Exception as e:
        logger.error("Database query for user failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error occurred"
        )

def create_user(db: Session, user: UserCreate) -> User:
    """
    Create a new user in the database.
    
    Args:
        db: Database session
        user: User creation data
    
    Returns:
        User: Created user object
    
    Raises:
        HTTPException: If user creation fails
    """
    try:
        logger.info(f"Creating new user: {user.username}")
        
        # Check if username already exists
        if db.query(User).filter(User.username == user.username).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
            
        # Check if email already exists
        if db.query(User).filter(User.email == user.email).first():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        db_user = User(
            username=user.username,
            email=user.email,
            hashed_password=get_password_hash(user.password)
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"User created successfully: {user.username}")
        return db_user
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        logger.error("User creation failed", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

def authenticate_user(db: Session, username: str, password: str) -> Union[User, bool]:
    """
    Authenticate user credentials.
    
    Args:
        db: Database session
        username: Username to authenticate
        password: Password to verify
    
    Returns:
        Union[User, bool]: User object if authenticated, False otherwise
    """
    try:
        user = get_user(db, username)
        if not user:
            logger.warning(f"Authentication failed: User not found - {username}")
            return False
            
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password - {username}")
            return False
            
        logger.info(f"User authenticated successfully: {username}")
        return user
        
    except Exception as e:
        logger.error("Authentication error", exc_info=True)
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.
    
    Args:
        data: Payload data for token
        expires_delta: Optional custom expiration time
    
    Returns:
        str: Encoded JWT token
    
    Raises:
        HTTPException: If token creation fails
    """
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        logger.debug(f"Access token created for user: {data.get('sub')}")
        return token
        
    except Exception as e:
        logger.error("Token creation failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create access token"
        )

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    Validate JWT token and return current user.
    
    Args:
        token: JWT token to validate
        db: Database session
    
    Returns:
        User: Current authenticated user
    
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            logger.warning("Token payload missing username")
            raise credentials_exception
            
        token_data = TokenData(username=username)
        
    except JWTError as e:
        logger.error("Token validation failed", exc_info=True)
        raise credentials_exception
    
    user = get_user(db, username=token_data.username)
    if not user:
        logger.warning(f"User not found: {token_data.username}")
        raise credentials_exception
    
    if not user.is_active:
        logger.warning(f"Inactive user attempted access: {user.username}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
        
    logger.debug(f"Current user validated: {user.username}")
    return user 