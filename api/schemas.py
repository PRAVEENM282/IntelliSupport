from typing import Optional

from pydantic import BaseModel


# --- Auth ---

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user_id: str
    email: str
    username: str


class UserProfile(BaseModel):
    user_id: str
    email: str
    username: str
    preferred_provider: str
    has_openai_key: bool
    has_groq_key: bool


class UpdateKeysRequest(BaseModel):
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    preferred_provider: Optional[str] = None


# --- Conversations ---

class ConversationOut(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    message_id: str
    role: str
    content: str
    retrieved_chunk_ids: Optional[list[str]] = None
    intent: Optional[str] = None
    provider: Optional[str] = None
    created_at: str


# --- Chat ---

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    provider: Optional[str] = None
    top_k: int = 5


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    response_text: str
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    retrieved_chunk_ids: list[str]
    provider: str


# --- Original pipeline schemas (kept for backward compatibility) ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    query_id: str
    response_id: str
    response_text: str
    intent: str
    intent_confidence: float
    retrieved_chunk_ids: list[str]
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None


class EvaluateResponse(BaseModel):
    response_id: str
    faithfulness_score: float
    relevance_score: float
    combined_score: float


class FeedbackRequest(BaseModel):
    response_id: str
    rating: int
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    chunks_indexed: int
    providers: list[str] = ["openai"]


class FileUploadResponse(BaseModel):
    doc_id: str
    title: str
    filename: str
    chunks_created: int
