from fastapi import FastAPI, File, Header, UploadFile, HTTPException, status, Request
from supabase import Client, create_client
from openai import OpenAI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime
import uuid
import hashlib
import binascii
import re
import resend
import bcrypt
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv
import requests

load_dotenv()


class Data(BaseModel):
    question: str
    response: str
    key: str
    like: bool


class DataFeedback(BaseModel):
    question: str
    response: str
    like: bool
    key: str
    correct_answer: str


class Register(BaseModel):
    email: str
    password: str
    username: str
    colliction_name: str


class Login(BaseModel):
    email: str
    password: str


url = os.getenv("url")
api_key = os.getenv("api_key")
openai_api_key = os.getenv("openai_api_key")

# for jwt
secret_key = os.getenv("secret_key")
algorithm = os.getenv("algorithm")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Initialize client
client_QD = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def create_collection(name: str):

    # Define your collection name

    # Create the collection
    client_QD.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=1536,  # Vector size (e.g., OpenAI embeddings size)
            distance=Distance.COSINE,  # Distance metric for similarity search
        ),
    )

    print(f"Collection '{name}' created successfully!")


# hash pass
pwd_context = CryptContext(
    schemes=["bcrypt"], deprecated="auto", bcrypt__truncate_error=False
)

# oauth 2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


import json
import hashlib
import binascii
import bcrypt  # موجود بالفعل في كودك
from fastapi import HTTPException, status


# ------------------------------------------------------------------
# PBKDF2 (no-salt) — نفس اللي تستخدمه في n8n tool
def hash_password_pbkdf2_nosalt(password: str) -> str:
    password_bytes = password.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha512", password_bytes, b"", 100000, dklen=64)
    return binascii.hexlify(dk).decode()


def verify_password_pbkdf2_nosalt(
    plain_password: str, hashed_password_hex: str
) -> bool:
    try:
        return hash_password_pbkdf2_nosalt(plain_password) == hashed_password_hex
    except Exception:
        return False


# ------------------------------------------------------------------
# عام — يفحص نوع الهَاش في DB ويستدعي الطريقة المناسبة
def extract_hashed_password(stored_value: str) -> str:
    """
    Normalize stored value:
    - If stored_value is JSON string that contains 'hashedPassword', return that.
    - Otherwise return stored_value as-is.
    """
    if not stored_value:
        return ""

    # If looks like JSON, try parse and extract hashedPassword
    if (stored_value.startswith("{") and stored_value.endswith("}")) or (
        "hashedPassword" in stored_value
    ):
        try:
            parsed = json.loads(stored_value)
            # If it has the field hashedPassword, use it
            if isinstance(parsed, dict) and "hashedPassword" in parsed:
                return parsed["hashedPassword"]
        except Exception:
            # not valid json, fallback to original
            pass

    return stored_value


def verify_password(plain_password: str, stored_value: str) -> bool:
    """
    Verify password against stored_value which can be:
    - PBKDF2 hex (from n8n)
    - bcrypt hash string (from old code)
    - JSON string that includes hashedPassword
    """
    if not stored_value:
        return False

    # normalize
    hashed = extract_hashed_password(stored_value)

    # 1) If looks like bcrypt (starts with $2), use bcrypt
    if isinstance(hashed, str) and hashed.startswith("$2"):
        try:
            return bcrypt.checkpw(
                plain_password.encode("utf-8"), hashed.encode("utf-8")
            )
        except Exception:
            return False

    # 2) If hex string length matches PBKDF2-512 (64 bytes -> 128 hex chars)
    #    pbkdf2 output 64 bytes -> 128 hex chars
    if (
        isinstance(hashed, str)
        and len(hashed) == 128
        and all(c in "0123456789abcdef" for c in hashed.lower())
    ):
        return verify_password_pbkdf2_nosalt(plain_password, hashed)

    # 3) fallback: try direct equality (in case stored something else)
    return hash_password_pbkdf2_nosalt(plain_password) == hashed


# ------------------------------------------------------------------


def create_token(data: dict):
    return jwt.encode(data, secret_key, algorithm=algorithm)


def decode_token(token: str):
    try:
        return jwt.decode(token, secret_key, algorithms=[algorithm])
    except JWTError:
        raise HTTPException(status_code=401, detail="invalid token")


supabase: Client = create_client(url, api_key)


client = OpenAI(api_key=openai_api_key)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ask(chat_history: str, number: str):
    system_prompt = """
        You are an expert real estate assistant. Your task is to analyze a conversation between a user and a chatbot.
        From the chat history, generate a JSON report containing the following fields:
        1. user_intent: Describe the property or type of real estate the user is interested in.
        2. user_mood: Indicate if the user seems happy, frustrated, or neutral.
        3. probability_to_buy: Estimate a percentage probability (0-100%) that the user will buy a property, based on the conversation.
        Always base your answers only on the provided chat history. Return the output strictly in JSON format.
        """

    user_prompt = f"""
            Here is the chat history:

                {chat_history}
            """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print(res.choices[0].message.content)


@app.get("/")
def hello():
    return "hello"


@app.post("/login")
def login(data: Login):
    res = supabase.table("user").select("*").eq("email", data.email).execute()

    # Check if user exists
    if not res.data or len(res.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Email not found"
        )

    user = res.data[0]

    stored_password_field = user.get("password")  # هذا اللي مخزن في DB

    # Verify password: uses the new verify_password that handles multiple formats
    if not verify_password(data.password, stored_password_field):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password"
        )

    # Create JWT
    token = create_token({"sub": user["email"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"email": user["email"], "username": user["username"]},
    }


@app.post("/register")
def register(data: Register):
    res = supabase.table("user").select("*").eq("email", data.email).execute()

    if res.data and len(res.data) > 0:
        raise HTTPException(status_code=400, detail="Email already exists")

    check = (
        supabase.table("collection")
        .select("*")
        .eq("name", data.colliction_name)
        .execute()
    )

    if check.data:
        raise HTTPException(status_code=400, detail="Colliction already exists")

    save_colliction = (
        supabase.table("collection")
        .insert(
            {
                "name": data.colliction_name,
            }
        )
        .execute()
    )
    create_collection(data.colliction_name)

    print(save_colliction.data[0]["id"])

    hashed_password = hash_password(data.password)

    user = (
        supabase.table("user")
        .insert(
            {
                "username": data.username,
                "email": data.email,
                "password": hashed_password,
                "key": save_colliction.data[0]["id"],
            }
        )
        .execute()
    )

    token = create_token({"sub": data.email})

    return {"access_token": token, "user_data": user.data[0]}


@app.get("/chats")
def chats(num: str):
    response = supabase.table("whatsapp_chat").select("*").execute()

    grouped = {}
    for chat in response.data:
        number = chat["number"]
        if number not in grouped:
            grouped[number] = []
        grouped[number].append(chat)

    for i in grouped.keys():
        ask(grouped[i], i)
    return grouped[num]


@app.post("/like")
def handel_like(data: Data):
    res = (
        supabase.table("testing")
        .insert(
            {
                "question": data.question,
                "response": data.response,
                "like": True,
                "key": data.key,
            }
        )
        .execute()
    )
    return {"status": "ok", "data": res.data}


@app.post("/dislike")
def handel_like(data: DataFeedback):
    res = (
        supabase.table("testing")
        .insert(
            {
                "question": data.question,
                "response": data.response,
                "like": False,
                "corrected_response": data.correct_answer,
                "key": data.key,
            }
        )
        .execute()
    )
    return {"status": "ok", "data": res.data}


class Key(BaseModel):
    key: str


@app.post("/history")
def history(req: Request):
    token = req.headers.get("token")
    print(token)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    user_data = get_current_user(token)

    print(user_data)
    email = user_data.get("sub")

    if not email:
        raise HTTPException(status_code=401, detail="Invalid token data")

    res = supabase.table("user").select("key").eq("email", email).execute()
    key = res.data[0]["key"]

    res = supabase.table("testing").select("*").eq("key", key).execute()

    print(res)
    return res.data


def chunking(text: str, size: int = 500, overlab: int = 100):
    start = 0
    chunks = []
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + size - overlab

        supabase.table("chunks").insert({"chunk": chunk, "size": len(chunk)}).execute()

    return chunks


def get_current_user(token: str):
    """Decode token and return the user info"""
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload  # contains e.g. {"id": 1, "email": "..."}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/getinfo")
def get_key(request: Request):
    token = request.headers.get("token")
    print(token)
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    user_data = get_current_user(token)

    print(user_data)
    email = user_data.get("sub")

    if not email:
        raise HTTPException(status_code=401, detail="Invalid token data")

    res = supabase.table("user").select("key").eq("email", email).execute()
    key = res.data[0]["key"]

    collection_data = (
        supabase.table("collection")
        .select("name, instruction")  # add instruction here
        .eq("id", key)
        .single()
        .execute()
    )

    collection_name = collection_data.data["name"]
    if not res.data or len(res.data) == 0:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "key": res.data[0]["key"],
        "email": email,
        "collection_name": collection_name,
        "instruction": collection_data.data.get("instruction"),
    }


def chunking(text: str, size: int = 500, overlap: int = 100):
    start = 0
    chunks = []
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = start + size - overlap
    return chunks


@app.post("/upload")
async def upload(file: UploadFile = File(...), authorization: str = Header(...)):
    # Get token
    token = authorization.replace("Bearer ", "")

    # Decode JWT to get user email
    user_info = decode_token(token)
    email = user_info.get("sub")

    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get user's collection name from database
    user_data = (
        supabase.table("user").select("key").eq("email", email).single().execute()
    )

    user_key = user_data.data["key"]  # ✅ extract the key value

    collection_data = (
        supabase.table("collection")
        .select("name")
        .eq("id", user_key)
        .single()
        .execute()
    )

    collection_name = collection_data.data["name"]
    if not user_data.data:
        raise HTTPException(status_code=404, detail="User not found")

    # Read and clean text
    content = await file.read()
    text = content.decode("utf-8")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?;:\s\u0600-\u06ff]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Create chunks
    chunks = chunking(text, 500, 100)

    # Upload to Qdrant
    points = []
    for i, chunk in enumerate(chunks):
        # Create embedding
        emb = client.embeddings.create(model="text-embedding-3-small", input=chunk)

        # Create point
        point = {
            "id": str(uuid.uuid4()),
            "vector": emb.data[0].embedding,
            "payload": {
                "content": chunk,
                "title": f"Chunk {i+1}",
                "language": (
                    "AR" if any("\u0600" <= c <= "\u06ff" for c in chunk) else "EN"
                ),
                "content_type": "text_chunk",
                "metadata": {
                    "source": file.filename,
                    "line": i + 1,
                },
            },
        }
        points.append(point)

    # Upload to Qdrant
    requests.put(
        f"{QDRANT_URL}/collections/{collection_name}/points?wait=true",
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
        json={"points": points},
    )

    return {"status": "success", "chunks": len(chunks), "collection": collection_name}


class Chat(BaseModel):
    message: str


@app.post("/api/chat")
async def chatbot(chat: Chat):
    print(chat.message)
    system_prompt = """
AI Assistant Testing Page (Developer Console)

This page is designed to help you test and refine your AI assistant before it goes live.
It allows you to type questions, view the assistant’s responses, and improve its behavior through feedback.

Here’s what you can do on this page:

✍️ Add custom instructions in the “Agent Instructions” section — these tell the assistant how to behave and what kind of responses it should give.

💬 Ask test questions in the input box and click “Send Question” to see how the assistant replies.

👍👎 Rate the responses — click “Like” if the answer is good, or “Dislike” if it’s wrong. If it’s wrong, you can type the correct answer to help train and improve the assistant.

🧩 Test full AI and n8n integration — your question is sent to the connected system, and the response comes back based on your workflow setup.

The main goal of this page is to improve the assistant’s accuracy and behavior by collecting real feedback and teaching it correct answers (a process known as Reinforcement Learning).
        
        


🧠 Knowledge Base Feeder — Explanation

This page lets developers upload Markdown (.md) files to feed and train the AI assistant’s knowledge base.
It processes each file, splits it into chunks, and saves it into a vector database for smart search and retrieval.

⚙️ Main Features

Authentication Check

When the page loads, it checks if there’s a valid token in localStorage.

If no token is found, the user is redirected to /login.

File Upload System

You can drag and drop or browse for a file.

Only files ending in .md (Markdown format) are accepted.

If another file type is uploaded, it shows an error message.

Upload Process

Once a valid file is selected, click the “Upload” button.

The system sends a POST request to the FastAPI backend (http://127.0.0.1:8000/upload) with:

The file (as FormData)

The Authorization token

The backend then:

Processes the Markdown file

Splits it into text chunks

Saves them into the vector database

Returns how many chunks were created and which collection they were stored in

Upload Status Messages

✅ Success: Shows how many chunks were created and the collection name.

❌ Error: Shows a message if the upload failed or the file type was invalid.

File Management

You can remove the selected file before uploading.

After a successful upload, the file preview and message reset after 3 seconds.

Styling

The interface uses a dark GitHub-like theme.

Bootstrap 5 handles layout and responsiveness.

The .upload-zone highlights when a file is dragged over it.

💡 In Short

This component is your developer console for uploading and managing Markdown documents that enrich your AI system’s knowledge base — ensuring your assistant can answer more accurately from structured content.
        

important : you are here to clerefy do not talk about ticnical things make the answer simple and clear
        
        """

    user_prompt = f"""
            you have this question and i need your help:
            i want a clear answer not long answer
                {chat.message}
            """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    print(res.choices[0].message.content)

    return res.choices[0].message.content


class KnoledgData(BaseModel):
    collectionName: str
    instructions: str


@app.post("/instructions")
def save(data: KnoledgData):
    res = (
        supabase.table("collection")
        .update({"instruction": data.instructions})
        .eq("name", data.collectionName)
        .execute()
    )
    return {"message": "true"}


@app.post("/publish")
def publish(authorization: str = Header(...)):
    token = authorization.replace("Bearer", "").replace("bearer", "").strip()

    user_info = decode_token(token)
    email = user_info.get("sub")

    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    user= supabase.table("user").select({"id"}).eq("email",email).execute()
    resend.api_key=os.getenv("RESEND")
    body={
        "from":"info@updates.zuccess.ai",
        "to":"abulailabood7@gmail.com",
        "subject":"New puplished",
        "text": f"the user {email} with id = {user.data[0]["id"]} has finished editing"

    }
    

    email=resend.Emails.send(body)
    print("email is sent successfully")
    return user
