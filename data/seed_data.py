"""
Seed MongoDB with sample e-commerce products and users.
Run this once before starting the API:
  python data/seed_data.py
"""

import asyncio
import random
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://localhost:27017/smartcart"

PRODUCTS = [
    {"product_id": "prod_001", "name": "Sony WH-1000XM5 Headphones", "category": "Electronics",
     "price": 349.99, "tags": ["wireless", "noise-cancelling", "audio", "bluetooth", "headphones"]},
    {"product_id": "prod_002", "name": "Apple AirPods Pro", "category": "Electronics",
     "price": 249.99, "tags": ["wireless", "earbuds", "apple", "audio", "bluetooth"]},
    {"product_id": "prod_003", "name": "Kindle Paperwhite", "category": "Electronics",
     "price": 139.99, "tags": ["ebook", "reading", "amazon", "display", "portable"]},
    {"product_id": "prod_004", "name": "Anker PowerCore 26800mAh", "category": "Electronics",
     "price": 65.99, "tags": ["power-bank", "charging", "portable", "usb-c", "travel"]},
    {"product_id": "prod_005", "name": "Logitech MX Master 3 Mouse", "category": "Electronics",
     "price": 99.99, "tags": ["mouse", "wireless", "ergonomic", "productivity", "office"]},
    {"product_id": "prod_006", "name": "Samsung 4K Monitor 27-inch", "category": "Electronics",
     "price": 399.99, "tags": ["monitor", "4k", "display", "gaming", "office"]},
    {"product_id": "prod_007", "name": "Mechanical Keyboard TKL RGB", "category": "Electronics",
     "price": 89.99, "tags": ["keyboard", "mechanical", "rgb", "gaming", "typing"]},
    {"product_id": "prod_008", "name": "Nike Air Zoom Pegasus 40", "category": "Footwear",
     "price": 130.00, "tags": ["running", "shoes", "nike", "sport", "fitness"]},
    {"product_id": "prod_009", "name": "Lululemon Align Leggings", "category": "Apparel",
     "price": 98.00, "tags": ["yoga", "leggings", "fitness", "comfort", "athleisure"]},
    {"product_id": "prod_010", "name": "Hydro Flask 32oz Water Bottle", "category": "Sports",
     "price": 44.95, "tags": ["water-bottle", "insulated", "fitness", "outdoor", "hydration"]},
    {"product_id": "prod_011", "name": "Instant Pot Duo 7-in-1", "category": "Kitchen",
     "price": 89.95, "tags": ["pressure-cooker", "cooking", "kitchen", "appliance", "meal-prep"]},
    {"product_id": "prod_012", "name": "Vitamix 5200 Blender", "category": "Kitchen",
     "price": 449.95, "tags": ["blender", "smoothie", "kitchen", "cooking", "healthy"]},
    {"product_id": "prod_013", "name": "Atomic Habits - James Clear", "category": "Books",
     "price": 18.99, "tags": ["self-help", "habits", "productivity", "book", "nonfiction"]},
    {"product_id": "prod_014", "name": "Deep Work - Cal Newport", "category": "Books",
     "price": 16.99, "tags": ["productivity", "focus", "career", "book", "nonfiction"]},
    {"product_id": "prod_015", "name": "Yoga Mat Premium 6mm", "category": "Sports",
     "price": 68.00, "tags": ["yoga", "fitness", "mat", "exercise", "wellness"]},
    {"product_id": "prod_016", "name": "Resistance Bands Set", "category": "Sports",
     "price": 29.99, "tags": ["fitness", "workout", "resistance", "home-gym", "exercise"]},
    {"product_id": "prod_017", "name": "Standing Desk Converter", "category": "Furniture",
     "price": 179.99, "tags": ["standing-desk", "office", "ergonomic", "work-from-home", "productivity"]},
    {"product_id": "prod_018", "name": "Ergonomic Office Chair", "category": "Furniture",
     "price": 299.99, "tags": ["chair", "ergonomic", "office", "lumbar", "work"]},
    {"product_id": "prod_019", "name": "Bose SoundLink Mini II Speaker", "category": "Electronics",
     "price": 149.99, "tags": ["speaker", "bluetooth", "portable", "audio", "bose"]},
    {"product_id": "prod_020", "name": "Nespresso Vertuo Coffee Maker", "category": "Kitchen",
     "price": 179.99, "tags": ["coffee", "espresso", "kitchen", "appliance", "morning"]},
]

USERS = [
    {"user_id": "user_001", "name": "Alice Chen", "preferences": {
        "favorite_categories": ["Electronics", "Books"], "price_range": "$50-$300"}},
    {"user_id": "user_002", "name": "Bob Martinez", "preferences": {
        "favorite_categories": ["Sports", "Footwear"], "price_range": "$30-$150"}},
    {"user_id": "user_003", "name": "Carol Singh", "preferences": {
        "favorite_categories": ["Kitchen", "Books"], "price_range": "$50-$500"}},
    {"user_id": "user_004", "name": "David Kim", "preferences": {
        "favorite_categories": ["Electronics", "Furniture"], "price_range": "$100-$500"}},
]

INTERACTIONS = [
    # Alice: tech + books user
    ("user_001", "prod_001", "purchase"), ("user_001", "prod_003", "purchase"),
    ("user_001", "prod_013", "purchase"), ("user_001", "prod_005", "view"),
    ("user_001", "prod_006", "add_to_cart"), ("user_001", "prod_014", "like"),

    # Bob: sports + fitness user
    ("user_002", "prod_008", "purchase"), ("user_002", "prod_010", "purchase"),
    ("user_002", "prod_016", "purchase"), ("user_002", "prod_015", "view"),
    ("user_002", "prod_009", "like"), ("user_002", "prod_019", "view"),

    # Carol: kitchen + reading user
    ("user_003", "prod_011", "purchase"), ("user_003", "prod_012", "purchase"),
    ("user_003", "prod_020", "purchase"), ("user_003", "prod_013", "purchase"),
    ("user_003", "prod_015", "view"), ("user_003", "prod_014", "add_to_cart"),

    # David: productivity / office setup user
    ("user_004", "prod_017", "purchase"), ("user_004", "prod_018", "purchase"),
    ("user_004", "prod_005", "purchase"), ("user_004", "prod_006", "purchase"),
    ("user_004", "prod_007", "add_to_cart"), ("user_004", "prod_013", "like"),
]

EVENT_WEIGHTS = {"view": 1.0, "like": 1.5, "add_to_cart": 2.0, "purchase": 3.0}


async def seed():
    from datetime import datetime, timezone, timedelta

    client = AsyncIOMotorClient(MONGO_URI)
    db = client["smartcart"]

    # Clear existing data
    await db.products.drop()
    await db.users.drop()
    await db.interactions.drop()

    # Seed products
    await db.products.insert_many(PRODUCTS)
    await db.products.create_index("product_id", unique=True)
    print(f"✅ Inserted {len(PRODUCTS)} products")

    # Seed users
    await db.users.insert_many(USERS)
    await db.users.create_index("user_id", unique=True)
    print(f"✅ Inserted {len(USERS)} users")

    # Seed interactions
    interaction_docs = []
    for i, (uid, pid, event) in enumerate(INTERACTIONS):
        interaction_docs.append({
            "user_id": uid,
            "product_id": pid,
            "event_type": event,
            "weight": EVENT_WEIGHTS[event],
            "timestamp": datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30)),
        })
    await db.interactions.insert_many(interaction_docs)
    await db.interactions.create_index([("user_id", 1), ("timestamp", -1)])
    print(f"✅ Inserted {len(interaction_docs)} interactions")

    client.close()
    print("\n🎉 Database seeded successfully! Run generate_embeddings.py next.")


if __name__ == "__main__":
    asyncio.run(seed())
