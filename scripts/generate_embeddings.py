"""
Precompute product embeddings using FastEmbed (no PyTorch, no API key needed).
Run once after seeding:
  python scripts/generate_embeddings.py
"""

import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from fastembed import TextEmbedding

MONGO_URI = "mongodb://localhost:27017/smartcart"


def build_product_text(product: dict) -> str:
    tags = " ".join(product.get("tags", []))
    return f"{product['name']} {product.get('category', '')} {tags}"


async def generate_embeddings():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client["smartcart"]

    products = await db.products.find({}, {"_id": 0}).to_list(length=10_000)
    print(f"Generating embeddings for {len(products)} products...")

    model = TextEmbedding("BAAI/bge-small-en-v1.5")
    texts = [build_product_text(p) for p in products]
    embeddings = list(model.embed(texts))

    for product, embedding in zip(products, embeddings):
        await db.products.update_one(
            {"product_id": product["product_id"]},
            {"$set": {"embedding": embedding.tolist()}}
        )

    print(f"Done! Stored {len(products)} embeddings (dim: {len(embeddings[0])})")
    client.close()


if __name__ == "__main__":
    asyncio.run(generate_embeddings())