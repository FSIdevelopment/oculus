"""Products router for managing token packages and licenses."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import stripe
from app.database import get_db
from app.models.product import Product
from app.auth.dependencies import get_current_active_user, admin_required
from app.models.user import User
from app.config import settings
from app.schemas.products import ProductCreate, ProductUpdate, ProductResponse

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.post("/api/admin/products", response_model=ProductResponse)
async def create_product(
    product_data: ProductCreate,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new product (token package or license).
    Syncs with Stripe on creation.
    Admin only.
    """
    # Create Stripe product and price
    stripe_product = None
    stripe_price = None
    
    if settings.STRIPE_SECRET_KEY:
        try:
            # Build Stripe kwargs dynamically - only include description if non-empty
            # Stripe rejects empty strings for description
            stripe_kwargs = {
                "name": product_data.name,
                "type": "service"
            }
            if product_data.description:
                stripe_kwargs["description"] = product_data.description

            stripe_product = stripe.Product.create(**stripe_kwargs)
            
            stripe_price = stripe.Price.create(
                product=stripe_product.id,
                unit_amount=product_data.price_cents,
                currency="usd"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create Stripe product: {str(e)}"
            )
    
    # Create product in database
    product = Product(
        name=product_data.name,
        price=product_data.price_cents / 100.0,
        product_type=product_data.product_type,
        token_amount=product_data.token_amount,
        stripe_product_id=stripe_product.id if stripe_product else None,
        stripe_price_id=stripe_price.id if stripe_price else product_data.stripe_price_id
    )
    
    db.add(product)
    await db.commit()
    await db.refresh(product)
    
    return ProductResponse(
        uuid=product.uuid,
        name=product.name,
        description=product_data.description,
        price=product.price,
        product_type=product.product_type,
        token_amount=product.token_amount,
        stripe_product_id=product.stripe_product_id,
        stripe_price_id=product.stripe_price_id,
        created_at=product.created_at
    )


@router.get("/api/products", response_model=list[ProductResponse])
async def list_products(
    db: AsyncSession = Depends(get_db)
):
    """List all available products (public, no auth required)."""
    result = await db.execute(select(Product))
    products = result.scalars().all()
    
    return [
        ProductResponse(
            uuid=p.uuid,
            name=p.name,
            description=None,
            price=p.price,
            product_type=p.product_type,
            token_amount=p.token_amount,
            stripe_product_id=p.stripe_product_id,
            stripe_price_id=p.stripe_price_id,
            created_at=p.created_at
        )
        for p in products
    ]


@router.put("/api/admin/products/{product_id}", response_model=ProductResponse)
async def update_product(
    product_id: str,
    product_data: ProductUpdate,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Update a product. Admin only."""
    result = await db.execute(
        select(Product).where(Product.uuid == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    # Update fields
    if product_data.name:
        product.name = product_data.name
    if product_data.description is not None:
        product.description = product_data.description
    if product_data.price_cents:
        product.price = product_data.price_cents / 100.0
    if product_data.token_amount:
        product.token_amount = product_data.token_amount
    
    await db.commit()
    await db.refresh(product)
    
    return ProductResponse(
        uuid=product.uuid,
        name=product.name,
        description=None,
        price=product.price,
        product_type=product.product_type,
        token_amount=product.token_amount,
        stripe_product_id=product.stripe_product_id,
        stripe_price_id=product.stripe_price_id,
        created_at=product.created_at
    )


@router.delete("/api/admin/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(
    product_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Soft-delete a product. Admin only."""
    result = await db.execute(
        select(Product).where(Product.uuid == product_id)
    )
    product = result.scalar_one_or_none()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found"
        )
    
    await db.delete(product)
    await db.commit()

