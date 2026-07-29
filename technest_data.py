"""
technest_data.py  –  TechNest RAG Knowledge Base
==================================================
Standalone data module.  Import anywhere:

    from technest_data import DOCUMENTS, GROUND_TRUTH, OUT_OF_SCOPE_QUERIES

Knowledge base: 40 documents (35 current + 5 outdated)
Ground truth  : 36 queries, including 12 numeric-detail queries deliberately
                designed to expose embedding weaknesses.
"""

# ---------------------------------------------------------------------------
# DOCUMENTS
# ---------------------------------------------------------------------------

DOCUMENTS = [
    # ── Shipping ─────────────────────────────────────────────────────────────
    {
        "document_id": 0,
        "title": "Standard Shipping Policy",
        "category": "Shipping",
        "doc_type": "policy",
        "effective_date": "2025-06-01",
        "is_current": True,
        "text": (
            "Standard shipping takes 3 to 5 business days and costs 4.99 USD, "
            "free for orders over 50 USD. Express shipping takes 1 to 2 business "
            "days and costs 14.99 USD. Orders placed before 2 PM local time ship "
            "the same business day."
        ),
    },
    {
        "document_id": 1,
        "title": "International Shipping Guide",
        "category": "Shipping",
        "doc_type": "help page",
        "effective_date": "2025-05-15",
        "is_current": True,
        "text": (
            "International orders take 7 to 14 business days depending on the "
            "destination country. Customers are responsible for customs duties "
            "and import taxes, which are calculated at checkout for supported "
            "countries. Some products cannot be shipped internationally because "
            "of battery restrictions."
        ),
    },
    {
        "document_id": 2,
        "title": "Order Tracking and Delivery Issues",
        "category": "Orders",
        "doc_type": "help page",
        "effective_date": "2025-07-01",
        "is_current": True,
        "text": (
            "Tracking numbers are emailed within 24 hours of shipment. If a "
            "package shows as delivered but was not received, contact support "
            "within 5 days so an investigation can begin. Packages that stay "
            "in transit more than 10 business days past the estimated delivery "
            "date are eligible for reshipment or a refund."
        ),
    },
    {
        "document_id": 3,
        "title": "Return and Refund Policy",
        "category": "Returns & Refunds",
        "doc_type": "policy",
        "effective_date": "2025-06-10",
        "is_current": True,
        "text": (
            "Most items can be returned within 30 days of delivery for a full "
            "refund, provided the item is unused and in its original packaging. "
            "Opened software, personal care items, and gift cards are final "
            "sale. Refunds are issued to the original payment method within 5 "
            "to 7 business days after the returned item is received and "
            "inspected."
        ),
    },
    {
        "document_id": 4,
        "title": "How Refunds Are Processed",
        "category": "Returns & Refunds",
        "doc_type": "procedure",
        "effective_date": "2025-06-10",
        "is_current": True,
        "text": (
            "Once a return arrives at the warehouse it is inspected within 2 "
            "business days. Approved refunds are issued to the original payment "
            "method, while store-credit refunds are issued instantly. Refunds "
            "for orders originally paid with a gift card are issued as store "
            "credit only, never as cash back."
        ),
    },
    {
        "document_id": 5,
        "title": "Order Cancellation Policy",
        "category": "Orders",
        "doc_type": "policy",
        "effective_date": "2025-06-20",
        "is_current": True,
        "text": (
            "Orders can be cancelled for free within 1 hour of purchase from "
            "the order history page. After 1 hour, if the order has not yet "
            "shipped, customers may request cancellation through support, but "
            "it is not guaranteed. Orders that have already shipped cannot be "
            "cancelled and must instead be returned after delivery under the "
            "standard return policy."
        ),
    },
    # ── Warranty & Repairs ────────────────────────────────────────────────────
    {
        "document_id": 6,
        "title": "Standard Warranty Coverage",
        "category": "Warranty & Repairs",
        "doc_type": "policy",
        "effective_date": "2025-01-01",
        "is_current": True,
        "text": (
            "All TechNest products purchased in 2022 or later include a 1 year "
            "manufacturer warranty covering defects in materials and "
            "workmanship. The warranty does not cover accidental damage, water "
            "damage, or unauthorized repairs. Proof of purchase is required for "
            "every warranty claim."
        ),
    },
    {
        "document_id": 7,
        "title": "Extended Protection Plan",
        "category": "Warranty & Repairs",
        "doc_type": "policy",
        "effective_date": "2025-02-01",
        "is_current": True,
        "text": (
            "Customers may purchase an Extended Protection Plan at checkout for "
            "2 additional years of coverage beyond the standard warranty, for a "
            "fee equal to roughly 10 to 15 percent of the item price. The plan "
            "also covers one accidental damage incident per year."
        ),
    },
    {
        "document_id": 8,
        "title": "Filing a Warranty Claim",
        "category": "Warranty & Repairs",
        "doc_type": "procedure",
        "effective_date": "2025-02-15",
        "is_current": True,
        "text": (
            "To file a warranty claim, submit the product serial number, proof "
            "of purchase, and a description of the defect through the support "
            "portal. Approved claims receive a prepaid return shipping label. "
            "Repairs typically take 10 to 15 business days; if a repair cannot "
            "be completed, a replacement or refund is issued instead."
        ),
    },
    # ── Payments & Billing ────────────────────────────────────────────────────
    {
        "document_id": 9,
        "title": "Payment Methods Accepted",
        "category": "Payments & Billing",
        "doc_type": "FAQ",
        "effective_date": "2025-03-01",
        "is_current": True,
        "text": (
            "TechNest accepts major credit and debit cards, PayPal, Apple Pay, "
            "Google Pay, and TechNest gift cards. Buy-now-pay-later financing "
            "is available on orders over 100 USD through a third-party provider. "
            "Payment information is encrypted and never stored on TechNest servers."
        ),
    },
    {
        "document_id": 10,
        "title": "Billing Errors and Duplicate Charges",
        "category": "Payments & Billing",
        "doc_type": "help page",
        "effective_date": "2025-03-10",
        "is_current": True,
        "text": (
            "A duplicate charge is usually a temporary authorization hold that "
            "disappears within 3 to 5 business days. If a real duplicate "
            "charge remains after 5 business days, contact billing support "
            "with the order number so it can be reversed."
        ),
    },
    # ── Account & Security ────────────────────────────────────────────────────
    {
        "document_id": 11,
        "title": "Password Reset for Customer Accounts",
        "category": "Account & Security",
        "doc_type": "procedure",
        "effective_date": "2025-04-01",
        "is_current": True,
        "text": (
            "Customers can reset a forgotten password using the Forgot "
            "Password link on the sign-in page, which emails a reset link to "
            "the registered address. The reset link expires after 30 minutes. "
            "If the registered email is no longer accessible, customers must "
            "verify identity with order details through support."
        ),
    },
    {
        "document_id": 12,
        "title": "Two-Factor Authentication Setup",
        "category": "Account & Security",
        "doc_type": "help page",
        "effective_date": "2025-04-05",
        "is_current": True,
        "text": (
            "Two-factor authentication adds a verification code sent by text "
            "message or an authenticator app during sign-in. It can be enabled "
            "from the account security settings page and is strongly "
            "recommended for accounts with saved payment methods."
        ),
    },
    # ── Rewards & Promotions ──────────────────────────────────────────────────
    {
        "document_id": 13,
        "title": "TechNest Rewards Program",
        "category": "Rewards Program",
        "doc_type": "FAQ",
        "effective_date": "2025-01-15",
        "is_current": True,
        "text": (
            "Rewards members earn 2 points per dollar spent, redeemable for "
            "discounts starting at 500 points for 5 USD off. Points expire "
            "after 12 months of account inactivity. Membership is free and "
            "enrollment happens automatically with a customer's first purchase."
        ),
    },
    {
        "document_id": 14,
        "title": "Gift Card Terms",
        "category": "Gift Cards",
        "doc_type": "policy",
        "effective_date": "2025-01-20",
        "is_current": True,
        "text": (
            "TechNest gift cards never expire and carry no maintenance fees. "
            "Gift cards cannot be redeemed for cash except where required by "
            "law. A lost gift card code can be recovered by contacting support "
            "with proof of purchase."
        ),
    },
    {
        "document_id": 15,
        "title": "Discount Codes and Promotions",
        "category": "Promotions",
        "doc_type": "FAQ",
        "effective_date": "2025-05-01",
        "is_current": True,
        "text": (
            "Discount codes can be applied at checkout and cannot be combined "
            "with other promotional codes. Only one discount code is allowed "
            "per order. Promotional pricing is not applied retroactively to "
            "orders placed before a promotion started."
        ),
    },
    # ── Product Support ───────────────────────────────────────────────────────
    {
        "document_id": 16,
        "title": "Headphones Pairing Troubleshooting",
        "category": "Product Support",
        "doc_type": "guide",
        "effective_date": "2025-05-20",
        "is_current": True,
        "text": (
            "If wireless headphones will not pair, reset them by holding the "
            "power button for 10 seconds until the light flashes, then forget "
            "the device in the phone's Bluetooth settings before pairing "
            "again. Keep the headphones within 3 feet of the device during pairing."
        ),
    },
    {
        "document_id": 17,
        "title": "Laptop Won't Turn On Troubleshooting",
        "category": "Product Support",
        "doc_type": "guide",
        "effective_date": "2025-05-25",
        "is_current": True,
        "text": (
            "If a laptop will not power on, hold the power button for 15 "
            "seconds to perform a hard reset, then charge it for at least 30 "
            "minutes before trying again. If the charging light never turns "
            "on, the charger or battery likely needs warranty service."
        ),
    },
    # ── International Orders ──────────────────────────────────────────────────
    {
        "document_id": 18,
        "title": "International Customs and Import Duties",
        "category": "International Orders",
        "doc_type": "help page",
        "effective_date": "2025-05-15",
        "is_current": True,
        "text": (
            "Import duties and taxes for international orders are calculated "
            "at checkout for supported countries and are non-refundable once "
            "an order ships, even if the order is later returned. Refused "
            "international shipments are subject to a restocking fee equal to "
            "the return shipping cost."
        ),
    },
    # ── Customer Support ──────────────────────────────────────────────────────
    {
        "document_id": 19,
        "title": "Contacting Customer Support",
        "category": "Customer Support",
        "doc_type": "FAQ",
        "effective_date": "2025-06-01",
        "is_current": True,
        "text": (
            "Support is available by live chat and email 7 days a week from 8 AM "
            "to 10 PM local time. Phone support is available Monday through "
            "Friday from 9 AM to 6 PM. Average email response time is under 4 "
            "hours during business hours."
        ),
    },
    # ── NEW: Additional current documents (20-34) ─────────────────────────────
    {
        "document_id": 23,
        "title": "Smartphone Battery Replacement Service",
        "category": "Warranty & Repairs",
        "doc_type": "procedure",
        "effective_date": "2025-03-15",
        "is_current": True,
        "text": (
            "TechNest offers in-store battery replacement for supported "
            "smartphone models. The service costs 49.99 USD and takes 2 to 3 "
            "business days. A 90-day limited warranty covers the replacement "
            "battery itself. Customers must back up their data before dropping "
            "off the device."
        ),
    },
    {
        "document_id": 24,
        "title": "Pre-Order Policy",
        "category": "Orders",
        "doc_type": "policy",
        "effective_date": "2025-04-10",
        "is_current": True,
        "text": (
            "Pre-orders are charged in full at the time of purchase. If the "
            "release date is delayed by more than 30 days, customers may cancel "
            "for a full refund. Pre-order items ship within 2 business days of "
            "the official release date and qualify for standard free-shipping "
            "thresholds."
        ),
    },
    {
        "document_id": 25,
        "title": "Bulk and Business Orders",
        "category": "Orders",
        "doc_type": "policy",
        "effective_date": "2025-04-20",
        "is_current": True,
        "text": (
            "Business accounts purchasing 10 or more units of the same product "
            "receive a 5 percent volume discount. Orders of 50 or more units "
            "qualify for a dedicated account manager and a 12 percent discount. "
            "Volume discounts do not combine with promotional codes."
        ),
    },
    {
        "document_id": 26,
        "title": "Tablet Screen Repair Service",
        "category": "Warranty & Repairs",
        "doc_type": "procedure",
        "effective_date": "2025-03-20",
        "is_current": True,
        "text": (
            "Screen repair for supported tablet models is available for "
            "89.99 USD. The repair turnaround is 3 to 5 business days. A "
            "60-day warranty covers workmanship. Accidental damage is not "
            "covered by the standard warranty but is eligible for the Extended "
            "Protection Plan screen repair benefit."
        ),
    },
    {
        "document_id": 27,
        "title": "Free Shipping Threshold Update 2025",
        "category": "Shipping",
        "doc_type": "policy",
        "effective_date": "2025-06-01",
        "is_current": True,
        "text": (
            "Effective June 2025, orders of 50 USD or more qualify for free "
            "standard shipping within the contiguous United States. Alaska, "
            "Hawaii, and U.S. territories require a minimum of 75 USD for free "
            "shipping. Free shipping does not apply to oversized items above "
            "30 lbs."
        ),
    },
    {
        "document_id": 28,
        "title": "Rewards Points Multiplier Events",
        "category": "Rewards Program",
        "doc_type": "FAQ",
        "effective_date": "2025-05-01",
        "is_current": True,
        "text": (
            "During multiplier events, rewards members earn 3x or 5x points "
            "per dollar spent instead of the standard 2x rate. Events are "
            "announced via email at least 48 hours in advance. Multiplier "
            "points are credited within 72 hours of order delivery."
        ),
    },
    {
        "document_id": 29,
        "title": "Referral Program Terms",
        "category": "Rewards Program",
        "doc_type": "policy",
        "effective_date": "2025-02-01",
        "is_current": True,
        "text": (
            "Customers who refer a friend receive 200 bonus points when the "
            "referred friend makes their first purchase of 25 USD or more. "
            "The referred friend receives a 10 USD welcome discount. Referral "
            "links expire 60 days after they are generated."
        ),
    },
    {
        "document_id": 30,
        "title": "Smart Home Device Setup Guide",
        "category": "Product Support",
        "doc_type": "guide",
        "effective_date": "2025-05-10",
        "is_current": True,
        "text": (
            "To set up a TechNest smart home device, download the TechNest "
            "Home app and ensure your Wi-Fi network is 2.4 GHz (5 GHz networks "
            "are not supported). Keep the device within 15 feet of the router "
            "during initial setup. Setup typically completes in under 5 minutes."
        ),
    },
    {
        "document_id": 31,
        "title": "Wireless Charger Compatibility",
        "category": "Product Support",
        "doc_type": "FAQ",
        "effective_date": "2025-04-15",
        "is_current": True,
        "text": (
            "TechNest wireless chargers support Qi and MagSafe standards. "
            "Maximum charging speed is 15 watts for compatible iPhones and "
            "10 watts for Qi-certified Android devices. Cases thicker than "
            "3 mm may reduce charging efficiency by up to 30 percent."
        ),
    },
    {
        "document_id": 32,
        "title": "Product Recycling and Trade-In Program",
        "category": "Sustainability",
        "doc_type": "FAQ",
        "effective_date": "2025-01-10",
        "is_current": True,
        "text": (
            "TechNest accepts trade-ins of any working electronics for store "
            "credit. Trade-in value is quoted online and locked for 14 days. "
            "Devices must be reset to factory settings before shipping. "
            "Non-working devices are accepted for free recycling but receive "
            "no trade-in credit."
        ),
    },
    {
        "document_id": 33,
        "title": "Privacy Policy Summary",
        "category": "Account & Security",
        "doc_type": "policy",
        "effective_date": "2025-01-01",
        "is_current": True,
        "text": (
            "TechNest does not sell customer data to third parties. Purchase "
            "history is retained for 7 years for tax and warranty purposes. "
            "Customers may request deletion of their personal data within 30 "
            "days; account history needed for open warranty claims is exempt "
            "from deletion requests."
        ),
    },
    {
        "document_id": 34,
        "title": "Accessibility Features and Assistive Services",
        "category": "Customer Support",
        "doc_type": "FAQ",
        "effective_date": "2025-03-01",
        "is_current": True,
        "text": (
            "TechNest offers a toll-free TTY line for customers who are deaf "
            "or hard of hearing, available Monday through Friday 9 AM to 6 PM. "
            "Large-print instruction manuals are available upon request at no "
            "additional charge. Screen-reader-compatible order receipts can be "
            "requested by emailing accessibility@technest.com."
        ),
    },
    {
        "document_id": 35,
        "title": "Packaging and Unboxing Policy",
        "category": "Returns & Refunds",
        "doc_type": "FAQ",
        "effective_date": "2025-02-15",
        "is_current": True,
        "text": (
            "Items returned without original packaging receive a 15 percent "
            "restocking fee deducted from the refund. Packaging must include "
            "all accessories, manuals, and inserts. Missing accessories are "
            "deducted at replacement cost, capped at 25 USD per missing item. "
            "Photos of the item and packaging are required for high-value returns over 300 USD."
        ),
    },
    {
        "document_id": 36,
        "title": "Seller Marketplace Policy",
        "category": "Orders",
        "doc_type": "policy",
        "effective_date": "2025-05-01",
        "is_current": True,
        "text": (
            "TechNest Marketplace third-party sellers must maintain a 4.5-star "
            "average rating to remain listed. Seller returns are handled "
            "independently; TechNest guarantees a resolution within 10 business "
            "days if the seller does not respond. Marketplace items are clearly "
            "labeled 'Sold by [Seller Name]' at checkout."
        ),
    },
    {
        "document_id": 37,
        "title": "Data Breach Notification Policy",
        "category": "Account & Security",
        "doc_type": "policy",
        "effective_date": "2025-01-01",
        "is_current": True,
        "text": (
            "In the event of a data breach affecting customer data, TechNest "
            "will notify affected customers within 72 hours of confirmation. "
            "Notifications are sent by email and, for breaches affecting more "
            "than 500 customers, posted to the TechNest security status page."
        ),
    },
    # ── Outdated documents ────────────────────────────────────────────────────
    {
        "document_id": 20,
        "title": "Return Policy Notice (Effective 2023)",
        "category": "Returns & Refunds",
        "doc_type": "old notice",
        "effective_date": "2023-01-10",
        "is_current": False,
        "text": (
            "Effective for purchases made before January 2024, the return "
            "window was 14 days from delivery instead of the current 30-day "
            "window. This notice is retained for historical reference only "
            "and no longer applies."
        ),
    },
    {
        "document_id": 21,
        "title": "Old Standard Shipping Rates (2022)",
        "category": "Shipping",
        "doc_type": "old notice",
        "effective_date": "2022-03-01",
        "is_current": False,
        "text": (
            "Before the 2023 shipping update, standard shipping cost 7.99 USD "
            "with no free shipping threshold, and delivery took 5 to 7 "
            "business days. These rates no longer apply to current orders."
        ),
    },
    {
        "document_id": 22,
        "title": "Legacy Warranty Terms (2021)",
        "category": "Warranty & Repairs",
        "doc_type": "old notice",
        "effective_date": "2021-01-01",
        "is_current": False,
        "text": (
            "Products purchased before 2022 carried a 90-day manufacturer "
            "warranty instead of the current 1-year warranty. This notice is "
            "retained for record purposes only and does not apply to current "
            "purchases."
        ),
    },
    {
        "document_id": 38,
        "title": "Old Rewards Redemption Rate (2023)",
        "category": "Rewards Program",
        "doc_type": "old notice",
        "effective_date": "2023-06-01",
        "is_current": False,
        "text": (
            "Prior to 2024, rewards points required 750 points for a 5 USD "
            "discount, and points expired after 6 months of inactivity. These "
            "terms were updated in January 2024 and no longer apply."
        ),
    },
    {
        "document_id": 39,
        "title": "Old Buy-Now-Pay-Later Threshold (2023)",
        "category": "Payments & Billing",
        "doc_type": "old notice",
        "effective_date": "2023-09-01",
        "is_current": False,
        "text": (
            "Before April 2024, buy-now-pay-later financing was available for "
            "orders over 150 USD. The threshold was lowered to 100 USD in April "
            "2024. This document is archived and no longer reflects current policy."
        ),
    },
]

# ---------------------------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------------------------
# Queries are grouped into three tiers:
#   A) Semantic / paraphrase queries  — embeddings shine, lexical struggles
#   B) Numeric-detail queries         — lexical / BM25 shines, embeddings blur
#   C) Multi-document queries         — both layers needed
# ---------------------------------------------------------------------------

GROUND_TRUTH = {

    # ── Tier A: Paraphrase / semantic ────────────────────────────────────────
    "How can I get my money back for something I returned?": [3, 4],
    "How many days do I have to send something back?": [3],
    "My order says delivered but it never showed up": [2],
    "Can I cancel my order after I already paid for it?": [5],
    "I forgot my password and can't log in": [11],
    "How do I turn on extra security for my account?": [12],
    "My wireless headphones won't connect to my phone": [16],
    "My laptop won't start, what should I do?": [17],
    "Do gift cards expire?": [14],
    "Can I combine two discount codes on one order?": [15],
    "What are your customer support hours?": [19],
    "If I paid with a gift card and return the item, do I get cash back?": [4],
    "What happens if a package is stuck in transit past the delivery date?": [2],
    "Do you store my credit card information?": [9],
    "Can I get more warranty coverage for my device?": [7],
    "How do I set up my new smart home device?": [30],
    "My trade-in device quote expired, what do I do?": [32],
    "Is my personal data ever sold to advertisers?": [33],

    # ── Tier B: Numeric-detail queries (embedding weakness) ──────────────────
    # These queries contain or demand EXACT numbers.  A pure semantic search
    # on a sentence-embedding model tends to conflate similar-sounding
    # numbers (e.g., 14-day vs 30-day, 7.99 vs 4.99) because the vectors
    # encode meaning rather than precise digit sequences.

    "How much does standard shipping cost right now?": [0],           # $4.99 — old doc says $7.99
    "What is the cutoff time to get same-day shipping?": [0],          # 2 PM — numeric precision
    "What is the current warranty length on a new TechNest purchase?": [6],  # 1 year — old: 90 days
    "How many extra years does the Extended Protection Plan add?": [7],  # 2 years
    "How long does a warranty repair typically take?": [8],             # 10-15 business days
    "How many points do I need to get 5 dollars off?": [13],           # 500 — old: 750
    "After how many months of inactivity do my rewards points expire?": [13],  # 12 — old: 6
    "How much does a smartphone battery replacement cost?": [23],       # $49.99
    "What is the minimum order to get volume business pricing?": [25],  # 10 units → 5%
    "What is the maximum wireless charging speed for iPhones?": [31],   # 15 W
    "How long do I have to cancel a pre-order if it is delayed?": [24], # >30 days delay
    "What is the free-shipping minimum order amount?": [0, 27],         # $50 (two current docs agree)
    "How much is the screen repair service for a tablet?": [26],        # $89.99
    "How long does TechNest store my purchase history?": [33],          # 7 years
    "How many hours after a data breach must TechNest notify me?": [37], # 72 hours
    "What percentage restocking fee applies if I return without packaging?": [35],  # 15%
    "What is the minimum purchase for a friend referral bonus to apply?": [29],  # $25
    "Do I get my customs fees back if I return an international order?": [18],
    "I see two identical charges for the same order, when will they resolve?": [10],  # 3-5 days
}

OUT_OF_SCOPE_QUERIES = [
    "Do you sell reusable water bottles?",
    "Can I get a refund in cryptocurrency?",
    "Is TechNest publicly traded on the stock market?",
    "Do you have physical retail stores?",
]

# ---------------------------------------------------------------------------
# HELPERS for query classification
# ---------------------------------------------------------------------------
NUMERIC_KEYWORDS = {
    "cost", "price", "how much", "how many", "how long", "days", "hours",
    "minutes", "dollars", "percent", "points", "years", "months", "usd",
    "fee", "rate", "speed", "threshold", "minimum", "maximum",
}

def is_numeric_query(query: str) -> bool:
    """Return True if the query demands an exact numeric detail."""
    q = query.lower()
    return any(kw in q for kw in NUMERIC_KEYWORDS)
