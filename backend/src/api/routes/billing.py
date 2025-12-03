from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request, status
import stripe

from api.deps import get_optional_current_user
from api.schemas.billing import CheckoutRequest, CheckoutResponse, SubscriptionState
from services.billing import BillingDB, BillingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/billing",
    tags=["Billing"],
)


@router.post("/checkout", response_model=CheckoutResponse, status_code=status.HTTP_201_CREATED)
async def create_checkout_session(
    payload: CheckoutRequest,
    current_user: Dict[str, Any] | None = Depends(get_optional_current_user),
) -> CheckoutResponse:
    """Create a Stripe checkout session for the requested tier."""

    customer_email = payload.user_email or (current_user or {}).get("email")
    if not customer_email:
        raise HTTPException(status_code=400, detail="Customer email is required")

    price_id = payload.price_id or BillingService.resolve_price_id(payload.tier)
    if not price_id:
        raise HTTPException(status_code=400, detail="No price configured for requested tier")

    try:
        url = BillingService.create_checkout_session(customer_email, price_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to create checkout session: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to create checkout session")

    return CheckoutResponse(url=url)


@router.post("/webhook")
async def stripe_webhook(request: Request) -> Dict[str, str]:
    """Handle Stripe webhook events to toggle subscription state."""

    payload = await request.body()

    try:
        event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Invalid Stripe webhook payload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid payload")

    if event.type == "customer.subscription.created":
        BillingDB.activate_subscription(event.data.object.customer)
    if event.type == "customer.subscription.deleted":
        BillingDB.deactivate_subscription(event.data.object.customer)

    return {"status": "ok"}


@router.get("/status", response_model=SubscriptionState)
async def get_subscription_status(current_user: Dict[str, Any] | None = Depends(get_optional_current_user)) -> SubscriptionState:
    """Return a simple subscription snapshot for the active user."""

    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")

    status_snapshot = BillingDB.get_status(current_user.get("email", "unknown"))
    return SubscriptionState(**status_snapshot)
