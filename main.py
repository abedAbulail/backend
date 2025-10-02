# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import httpx
# from typing import Optional

# app = FastAPI()

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[],  # Your React app URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Ziina API Configuration - ONLY NEED THIS!
# ZIINA_ACCESS_TOKEN = "0rdrAVDckx4YX/vk0y52p71t+e/488iFgDtNYHeNdIGsAh8AjKG6ejzo/VKAq6rq"  # Get from Ziina dashboard
# ZIINA_BASE_URL = "https://api-v2.ziina.com/api"


# class PaymentRequest(BaseModel):
#     amount: float  # In base units (e.g., 1050 for 10.50 AED)
#     currency_code: str = "AED"
#     message: str
#     customer_email: Optional[str] = None
#     customer_name: Optional[str] = None
#     allow_tips: bool = False


# class PaymentResponse(BaseModel):
#     payment_id: str
#     payment_url: str
#     status: str


# @app.post("/api/create-payment", response_model=PaymentResponse)
# async def create_payment(payment: PaymentRequest):
#     """Create a Ziina payment intent"""

#     headers = {
#         "Authorization": f"Bearer {ZIINA_ACCESS_TOKEN}",
#         "Content-Type": "application/json",
#     }

#     payload = {
#         "amount": int(payment.amount),  # Must be integer in base units
#         "currency_code": payment.currency_code,
#         "message": payment.message,
#         "success_url": "http://localhost:3000/payment-success",
#         "cancel_url": "http://localhost:3000/payment-cancel",
#         "failure_url": "http://localhost:3000/payment-failure",
#         "allow_tips": payment.allow_tips,
#         "test": True  # Test mode - no real payment needed!
#     }
#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.post(
#                 f"{ZIINA_BASE_URL}/payment_intent",
#                 headers=headers,
#                 json=payload,
#                 timeout=30.0,
#             )
#             response.raise_for_status()
#             data = response.json()

#             return PaymentResponse(
#                 payment_id=data.get("id"),
#                 payment_url=data.get("redirect_url"),
#                 status=data.get("status"),
#             )
#         except httpx.HTTPError as e:
#             raise HTTPException(
#                 status_code=500, detail=f"Payment creation failed: {str(e)}"
#             )


# @app.get("/api/payment-status/{payment_id}")
# async def get_payment_status(payment_id: str):
#     """Check payment status"""

#     headers = {
#         "Authorization": f"Bearer {ZIINA_ACCESS_TOKEN}",
#         "Content-Type": "application/json",
#     }

#     async with httpx.AsyncClient() as client:
#         try:
#             response = await client.get(
#                 f"{ZIINA_BASE_URL}/payment_intent/{payment_id}",
#                 headers=headers,
#                 timeout=30.0,
#             )
#             response.raise_for_status()
#             payment_data = response.json()

#             # Payment statuses from Ziina:
#             # - requires_payment_instrument
#             # - requires_user_action
#             # - pending
#             # - completed
#             # - failed
#             # - canceled

#             status = payment_data.get("status")

#             if status == "completed":
#                 print(f"✅ Payment {payment_id} successful!")
#                 # TODO: Update your order/database here

#             return payment_data

#         except httpx.HTTPError as e:
#             raise HTTPException(
#                 status_code=500, detail=f"Failed to fetch payment status: {str(e)}"
#             )


# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from typing import Optional
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ "https://whatsapp-automation-landing-pag.onrender.com"],  # Your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


load_dotenv()
# Ziina API Configuration - ONLY NEED THIS!
ZIINA_ACCESS_TOKEN = os.getenv("ZIINA_ACCESS_TOKEN")
# "0rdrAVDckx4YX/vk0y52p71t+e/488iFgDtNYHeNdIGsAh8AjKG6ejzo/VKAq6rq"  # Get from Ziina dashboard
ZIINA_BASE_URL = os.getenv("ZIINA_BASE_URL")


class PaymentRequest(BaseModel):
    amount: float  # In base units (e.g., 1050 for 10.50 AED)
    currency_code: str = "USD"
    message: str
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    allow_tips: bool = False


class PaymentResponse(BaseModel):
    payment_id: str
    payment_url: str
    status: str


@app.post("/api/create-payment", response_model=PaymentResponse)
async def create_payment(payment: PaymentRequest):
    """Create a Ziina payment intent"""

    headers = {
        "Authorization": f"Bearer {ZIINA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "amount": int(payment.amount),  # Must be integer in base units
        "currency_code": payment.currency_code,
        "message": payment.message,
        "success_url": "http://localhost:3000/payment-success",
        "cancel_url": "http://localhost:3000/payment-cancel",
        "failure_url": "http://localhost:3000/payment-failure",
        "allow_tips": payment.allow_tips,
        "test": False,  # Test mode - no real payment needed!
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{ZIINA_BASE_URL}/payment_intent",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            return PaymentResponse(
                payment_id=data.get("id"),
                payment_url=data.get("redirect_url"),
                status=data.get("status"),
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=500, detail=f"Payment creation failed: {str(e)}"
            )


@app.get("/api/payment-status/{payment_id}")
async def get_payment_status(payment_id: str):
    """Check payment status"""

    headers = {
        "Authorization": f"Bearer {ZIINA_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{ZIINA_BASE_URL}/payment_intent/{payment_id}",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            payment_data = response.json()

            # Payment statuses from Ziina:
            # - requires_payment_instrument
            # - requires_user_action
            # - pending
            # - completed
            # - failed
            # - canceled

            status = payment_data.get("status")

            if status == "completed":
                print(f"✅ Payment {payment_id} successful!")
                # TODO: Update your order/database here

            return payment_data

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch payment status: {str(e)}"
            )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
