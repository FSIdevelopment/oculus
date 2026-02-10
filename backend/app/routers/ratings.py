"""Ratings router for strategy ratings and reviews."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from app.database import get_db
from app.models.rating import Rating
from app.models.strategy import Strategy
from app.auth.dependencies import get_current_active_user, admin_required
from app.models.user import User
from app.schemas.ratings import RatingCreate, RatingUpdate, RatingResponse, RatingListResponse

router = APIRouter()


@router.post("/api/strategies/{strategy_id}/ratings", response_model=RatingResponse)
async def create_rating(
    strategy_id: str,
    rating_data: RatingCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Rate a strategy (score: 1-5, review: optional text).
    One rating per user per strategy.
    """
    # Verify strategy exists
    result = await db.execute(
        select(Strategy).where(Strategy.uuid == strategy_id)
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Check if user already rated this strategy
    result = await db.execute(
        select(Rating).where(
            (Rating.user_id == current_user.uuid) &
            (Rating.strategy_id == strategy_id)
        )
    )
    existing_rating = result.scalar_one_or_none()
    
    if existing_rating:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You have already rated this strategy"
        )
    
    # Create rating
    rating = Rating(
        rating=rating_data.score,
        review_text=rating_data.review,
        user_id=current_user.uuid,
        strategy_id=strategy_id
    )
    
    db.add(rating)
    await db.commit()
    await db.refresh(rating)
    
    return RatingResponse(
        uuid=rating.uuid,
        user_id=rating.user_id,
        strategy_id=rating.strategy_id,
        rating=rating.rating,
        review_text=rating.review_text,
        created_at=rating.created_at
    )


@router.get("/api/strategies/{strategy_id}/ratings", response_model=RatingListResponse)
async def get_strategy_ratings(
    strategy_id: str,
    page: int = 1,
    page_size: int = 50,
    db: AsyncSession = Depends(get_db)
):
    """Get ratings for a strategy (paginated)."""
    # Verify strategy exists
    result = await db.execute(
        select(Strategy).where(Strategy.uuid == strategy_id)
    )
    strategy = result.scalar_one_or_none()
    
    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )
    
    # Get total count
    count_result = await db.execute(
        select(func.count(Rating.uuid)).where(Rating.strategy_id == strategy_id)
    )
    total = count_result.scalar()
    
    # Get ratings
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Rating)
        .where(Rating.strategy_id == strategy_id)
        .order_by(desc(Rating.created_at))
        .offset(offset)
        .limit(page_size)
    )
    ratings = result.scalars().all()
    
    total_pages = (total + page_size - 1) // page_size
    
    return RatingListResponse(
        ratings=[
            RatingResponse(
                uuid=r.uuid,
                user_id=r.user_id,
                strategy_id=r.strategy_id,
                rating=r.rating,
                review_text=r.review_text,
                created_at=r.created_at
            )
            for r in ratings
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.put("/api/ratings/{rating_id}", response_model=RatingResponse)
async def update_rating(
    rating_id: str,
    rating_data: RatingUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Update own rating."""
    result = await db.execute(
        select(Rating).where(Rating.uuid == rating_id)
    )
    rating = result.scalar_one_or_none()
    
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rating not found"
        )
    
    if rating.user_id != current_user.uuid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only update your own ratings"
        )
    
    # Update fields
    if rating_data.score:
        rating.rating = rating_data.score
    if rating_data.review is not None:
        rating.review_text = rating_data.review
    
    await db.commit()
    await db.refresh(rating)
    
    return RatingResponse(
        uuid=rating.uuid,
        user_id=rating.user_id,
        strategy_id=rating.strategy_id,
        rating=rating.rating,
        review_text=rating.review_text,
        created_at=rating.created_at
    )


@router.get("/api/admin/ratings", response_model=RatingListResponse)
async def list_all_ratings(
    page: int = 1,
    page_size: int = 50,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """List all ratings. Admin only."""
    # Get total count
    count_result = await db.execute(select(func.count(Rating.uuid)))
    total = count_result.scalar()
    
    # Get ratings
    offset = (page - 1) * page_size
    result = await db.execute(
        select(Rating)
        .order_by(desc(Rating.created_at))
        .offset(offset)
        .limit(page_size)
    )
    ratings = result.scalars().all()
    
    total_pages = (total + page_size - 1) // page_size
    
    return RatingListResponse(
        ratings=[
            RatingResponse(
                uuid=r.uuid,
                user_id=r.user_id,
                strategy_id=r.strategy_id,
                rating=r.rating,
                review_text=r.review_text,
                created_at=r.created_at
            )
            for r in ratings
        ],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.delete("/api/admin/ratings/{rating_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rating(
    rating_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Delete a rating. Admin only."""
    result = await db.execute(
        select(Rating).where(Rating.uuid == rating_id)
    )
    rating = result.scalar_one_or_none()
    
    if not rating:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rating not found"
        )
    
    await db.delete(rating)
    await db.commit()

