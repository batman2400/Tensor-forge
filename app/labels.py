"""Label enums and the fixed category -> team table.

Single source of truth for serving and training. tests/test_labels.py checks
these values against the OpenAPI spec.
"""

from __future__ import annotations

CATEGORIES: tuple[str, ...] = (
    "payment_refund",
    "ride_trip_issue",
    "lost_item",
    "order_missing_wrong",
    "delivery_delay",
    "food_quality",
    "account_promo",
    "safety_conduct",
    "app_technical",
    "general_inquiry",
    "spam_irrelevant",
)

TEAM_BY_CATEGORY: dict[str, str] = {
    "payment_refund": "Payments & Refunds",
    "ride_trip_issue": "Ride Operations",
    "lost_item": "Lost & Found",
    "order_missing_wrong": "Food Operations",
    "delivery_delay": "Delivery Operations",
    "food_quality": "Restaurant Quality",
    "account_promo": "Account Services",
    "safety_conduct": "Trust & Safety",
    "app_technical": "Tech Support",
    "general_inquiry": "Front-line Support",
    "spam_irrelevant": "Auto-close / Spam Filter",
}

CHANNELS: tuple[str, ...] = ("email", "chat", "call_transcript")

# The only secondary classes observed in the dataset (architecture section 5.1).
SECONDARY_CATEGORIES: tuple[str, ...] = (
    "ride_trip_issue",
    "payment_refund",
    "app_technical",
    "delivery_delay",
    "order_missing_wrong",
)

# These categories are never urgent in the labelled data. Spam is also a schema rule.
NEVER_URGENT: frozenset[str] = frozenset(
    {"account_promo", "app_technical", "general_inquiry", "spam_irrelevant"}
)
