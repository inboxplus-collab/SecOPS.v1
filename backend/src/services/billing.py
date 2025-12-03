"""Lightweight billing service wrappers for Stripe + role propagation."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import stripe

from utils.config import settings

logger = logging.getLogger(__name__)

# Configure Stripe if a key is available. This avoids failing imports when the
# backend is booted without billing credentials.
if settings.STRIPE_SECRET_KEY:
    stripe.api_key = settings.STRIPE_SECRET_KEY


class BillingDB:
    """
    Extremely small in-memory subscription tracker.

    In production, replace with a persistent table keyed by customer_id or
    email. We keep the surface minimal so the admin dashboard and role guard
    can be wired quickly.
    """

    _subscriptions: Dict[str, str] = {}

    @classmethod
    def activate_subscription(cls, customer_identifier: str, tier: str = "pro") -> None:
        cls._subscriptions[customer_identifier] = tier
        logger.info("Subscription activated for %s with tier=%s", customer_identifier, tier)

    @classmethod
    def deactivate_subscription(cls, customer_identifier: str) -> None:
        cls._subscriptions.pop(customer_identifier, None)
        logger.info("Subscription deactivated for %s", customer_identifier)

    @classmethod
    def get_status(cls, customer_identifier: str) -> Dict[str, Optional[str]]:
        tier = cls._subscriptions.get(customer_identifier, "free")
        return {"customer": customer_identifier, "tier": tier}


class BillingService:
    """Stripe checkout orchestration and webhook helpers."""

    @staticmethod
    def _assert_configured() -> None:
        if not settings.STRIPE_SECRET_KEY:
            raise ValueError("Stripe secret key is not configured")

    @staticmethod
    def create_checkout_session(user_id: str, price_id: str) -> str:
        BillingService._assert_configured()

        success_url = f"{settings.FRONTEND_URL}/billing/success"
        cancel_url = f"{settings.FRONTEND_URL}/billing/cancel"

        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="subscription",
            customer_email=user_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
        )
        logger.info("Created Stripe checkout session for %s", user_id)
        return session.url

    @staticmethod
    def resolve_price_id(tier: str) -> Optional[str]:
        tier_normalized = tier.lower()
        if tier_normalized == "pro":
            return settings.STRIPE_PRICE_PRO
        if tier_normalized == "enterprise":
            return settings.STRIPE_PRICE_ENTERPRISE or settings.STRIPE_PRICE_PRO
        return None
