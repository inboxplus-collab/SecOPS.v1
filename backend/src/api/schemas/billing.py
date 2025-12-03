from __future__ import annotations

from pydantic import BaseModel, EmailStr


class CheckoutRequest(BaseModel):
    price_id: str | None = None
    tier: str = "pro"
    user_email: EmailStr | None = None


class CheckoutResponse(BaseModel):
    url: str


class SubscriptionState(BaseModel):
    customer: str
    tier: str
