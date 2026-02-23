from fastapi import APIRouter, Query, HTTPException
from services.mongo_service import mongo_service

router = APIRouter()


@router.get("")
async def list_products(
    category: str = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
):
    """List products from catalog, optionally filtered by category."""
    query = {}
    if category:
        query["category"] = category

    cursor = mongo_service.db.products.find(query, {"_id": 0, "embedding": 0}).skip(skip).limit(limit)
    products = await cursor.to_list(length=limit)
    total = await mongo_service.db.products.count_documents(query)
    return {"products": products, "total": total, "skip": skip, "limit": limit}


@router.get("/{product_id}")
async def get_product(product_id: str):
    product = await mongo_service.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    product.pop("embedding", None)  # Don't expose raw vectors
    return product
