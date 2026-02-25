"""Admin endpoints for user management and support."""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct
from datetime import datetime, timedelta
import secrets
import string
import stripe
import redis.asyncio as redis

from app.database import get_db
from app.config import settings
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.build_iteration import BuildIteration
from app.models.license import License
from app.schemas.users import (
    UserAdminUpdate, UserDetailResponse, UserListResponse,
    UserListItem, ImpersonationTokenResponse, AdminUserCreate
)
from app.schemas.strategies import StrategyResponse, StrategyUpdate, StrategyListResponse
from app.schemas.licenses import LicenseResponse
from app.auth.dependencies import admin_required
from app.auth.security import hash_password, create_access_token
from app.services.email_service import EmailService
from app.services.build_queue import BuildQueueManager
from app.services.monitoring import MonitoringService

router = APIRouter()

# Configure Stripe
stripe.api_key = settings.STRIPE_SECRET_KEY


@router.get("/users", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: str = Query("", min_length=0),
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    List all users with pagination and search.

    - Paginated with skip/limit
    - Search by name or email (case-insensitive)
    """
    query = select(User)

    # Apply search filter
    if search:
        query = query.where(
            (User.name.ilike(f"%{search}%")) |
            (User.email.ilike(f"%{search}%"))
        )

    # Get total count
    count_result = await db.execute(select(func.count(User.uuid)).select_from(User))
    total = count_result.scalar()

    # Apply pagination
    query = query.offset(skip).limit(limit)
    result = await db.execute(query)
    users = result.scalars().all()

    items = [UserListItem.model_validate(u) for u in users]

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.post("/users", response_model=UserDetailResponse)
async def create_user(
    user_data: AdminUserCreate,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user (admin only).

    - Validates email uniqueness
    - Hashes password with bcrypt
    - Creates user in DB
    - Optionally creates Stripe customer (non-blocking)
    - Returns UserDetailResponse
    """
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already in use"
        )

    # Create new user
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        phone_number=user_data.phone_number,
        balance=0.0,
        status=user_data.status,
        user_role=user_data.user_role
    )

    # Try to create Stripe customer (non-blocking)
    if settings.STRIPE_SECRET_KEY:
        try:
            customer = stripe.Customer.create(
                email=user_data.email,
                name=user_data.name
            )
            new_user.stripe_customer_id = customer.id
        except Exception:
            # Don't fail user creation if Stripe is unavailable
            pass

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # Send welcome email with credentials (admin-created users)
    EmailService.send_welcome_email(user=new_user, password=user_data.password)
    EmailService.send_onboarding_email(user=new_user)

    return new_user


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed information about a specific user."""
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user


@router.put("/users/{user_id}", response_model=UserDetailResponse)
async def update_user(
    user_id: str,
    user_update: UserAdminUpdate,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Update user information (admin only).

    - Can update all fields including role and status
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Check email uniqueness
    if user_update.email and user_update.email != user.email:
        result = await db.execute(select(User).where(User.email == user_update.email))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )

    # Update fields
    if user_update.name:
        user.name = user_update.name
    if user_update.email:
        user.email = user_update.email
    if user_update.phone_number is not None:
        user.phone_number = user_update.phone_number
    if user_update.user_role:
        user.user_role = user_update.user_role
    if user_update.status:
        user.status = user_update.status
    if user_update.balance is not None:
        user.balance = user_update.balance

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def soft_delete_user(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Soft-delete a user (set is_active=False).

    - User data is preserved but account is deactivated
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    user.status = "inactive"
    db.add(user)
    await db.commit()


@router.post("/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a temporary password for a user.

    - Creates a random temporary password
    - In production, would send via Resend email service
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Generate temporary password
    temp_password = ''.join(
        secrets.choice(string.ascii_letters + string.digits)
        for _ in range(12)
    )

    # Hash and update password
    user.password_hash = hash_password(temp_password)
    db.add(user)
    await db.commit()

    # In production, send via Resend:
    # resend.Emails.send({
    #     "from": "noreply@oculus.com",
    #     "to": user.email,
    #     "subject": "Password Reset",
    #     "html": f"Your temporary password is: {temp_password}"
    # })

    return {
        "message": "Password reset. Email sent to user.",
        "temp_password": temp_password  # In production, don't return this
    }


@router.post("/users/{user_id}/impersonate", response_model=ImpersonationTokenResponse)
async def impersonate_user(
    user_id: str,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate an impersonation JWT token for a user.

    - Creates a token as if the user logged in
    - Includes impersonator_id claim for audit trail
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Create token with impersonator info
    token_data = {
        "sub": user.uuid,
        "email": user.email,
        "role": user.user_role,
        "impersonator_id": current_user.uuid
    }
    access_token = create_access_token(data=token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "impersonated_user_id": user.uuid,
        "impersonator_id": current_user.uuid
    }


@router.post("/users/{user_id}/support-email")
async def send_support_email(
    user_id: str,
    email_data: dict,
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a support email to a user.

    - In production, uses Resend email service
    """
    result = await db.execute(select(User).where(User.uuid == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # In production, send via Resend:
    # resend.Emails.send({
    #     "from": "support@oculus.com",
    #     "to": user.email,
    #     "subject": email_data.get("subject", "Support"),
    #     "html": email_data.get("body", "")
    # })

    return {
        "message": f"Support email sent to {user.email}",
        "user_id": user.uuid
    }


# ============================================================================
# STRATEGY ADMIN ENDPOINTS
# ============================================================================

@router.get("/strategies", response_model=StrategyListResponse)
async def list_all_strategies(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    search: str = Query("", min_length=0),
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """List ALL strategies (admin only, searchable) with owner name, build count, and annual return."""
    # Build base query with join to User table
    from sqlalchemy.orm import selectinload

    query = (
        select(
            Strategy,
            User.name.label("owner_name"),
            func.count(distinct(StrategyBuild.uuid)).label("build_count")
        )
        .join(User, Strategy.user_id == User.uuid)
        .outerjoin(StrategyBuild, Strategy.uuid == StrategyBuild.strategy_id)
        .where(Strategy.status != "deleted")
        .group_by(Strategy.uuid, User.name)
    )

    if search:
        query = query.where(Strategy.name.ilike(f"%{search}%"))

    # Count total
    count_result = await db.execute(
        select(func.count(distinct(Strategy.uuid)))
        .select_from(Strategy)
        .where(Strategy.status != "deleted")
    )
    total = count_result.scalar()

    # Fetch paginated with owner name and build count
    result = await db.execute(
        query.offset(skip).limit(limit)
    )
    rows = result.all()

    # Build response items with additional fields
    items = []
    for strategy, owner_name, build_count in rows:
        # Extract annual return from backtest_results if available
        annual_return = None
        if strategy.backtest_results and isinstance(strategy.backtest_results, dict):
            # Try to get total_return_pct or total_return from backtest results
            annual_return = strategy.backtest_results.get('total_return_pct') or strategy.backtest_results.get('total_return')

        # Calculate best return % from all builds
        best_return_pct = None
        builds_result = await db.execute(
            select(StrategyBuild)
            .where(
                StrategyBuild.strategy_id == strategy.uuid,
                StrategyBuild.status == "complete"
            )
        )
        builds = builds_result.scalars().all()

        max_return = None
        for build in builds:
            # Get all iterations for this build
            iterations_result = await db.execute(
                select(BuildIteration)
                .where(
                    BuildIteration.build_id == build.uuid,
                    BuildIteration.status == "complete"
                )
            )
            iterations = iterations_result.scalars().all()

            # Find max return from all iterations
            for iteration in iterations:
                if iteration.backtest_results and isinstance(iteration.backtest_results, dict):
                    # Try to get total_return_pct or total_return
                    return_val = iteration.backtest_results.get('total_return_pct') or iteration.backtest_results.get('total_return')
                    if return_val is not None:
                        if max_return is None or return_val > max_return:
                            max_return = return_val

        best_return_pct = max_return

        # Create response dict from strategy
        strategy_dict = {
            "uuid": strategy.uuid,
            "name": strategy.name,
            "description": strategy.description,
            "status": strategy.status,
            "strategy_type": strategy.strategy_type,
            "symbols": strategy.symbols,
            "target_return": strategy.target_return,
            "backtest_results": strategy.backtest_results,
            "marketplace_listed": strategy.marketplace_listed,
            "marketplace_price": strategy.marketplace_price,
            "version": strategy.version,
            "subscriber_count": strategy.subscriber_count,
            "rating": strategy.rating,
            "user_id": strategy.user_id,
            "created_at": strategy.created_at,
            "updated_at": strategy.updated_at,
            "owner_name": owner_name,
            "build_count": build_count,
            "annual_return": annual_return,
            "best_return_pct": best_return_pct
        }
        items.append(StrategyResponse(**strategy_dict))

    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.put("/strategies/{strategy_id}", response_model=StrategyResponse)
async def update_any_strategy(
    strategy_id: str,
    strategy_data: StrategyUpdate,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Edit any strategy (admin only)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Update fields
    if strategy_data.name is not None:
        strategy.name = strategy_data.name
    if strategy_data.description is not None:
        strategy.description = strategy_data.description
    if strategy_data.config is not None:
        strategy.symbols = strategy_data.config

    strategy.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(strategy)

    return strategy


@router.delete("/strategies/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_any_strategy(
    strategy_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """Delete any strategy (admin only)."""
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    strategy.status = "deleted"
    strategy.updated_at = datetime.utcnow()
    await db.commit()


# ============================================================================
# ADMIN LICENSE ENDPOINTS
# ============================================================================

@router.get("/strategies/{strategy_id}/versions")
async def get_strategy_versions(
    strategy_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all available versions for a strategy based on completed builds.

    Returns a list of version numbers that have been successfully built and pushed
    to the docker registry. These versions can be used when creating licenses.
    """
    # Verify strategy exists
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Get the current version from the strategy (latest)
    current_version = strategy.version or 1

    # Get all completed builds to determine which versions were successfully built
    # Each completed build represents a version that was pushed to the registry
    builds_result = await db.execute(
        select(StrategyBuild)
        .where(
            StrategyBuild.strategy_id == strategy_id,
            StrategyBuild.status == "complete"
        )
        .order_by(StrategyBuild.started_at.desc())
    )
    completed_builds = builds_result.scalars().all()

    # Get best return per build from completed iterations
    best_returns: dict[str, float] = {}
    build_ids = [b.uuid for b in completed_builds]
    if build_ids:
        iters_result = await db.execute(
            select(BuildIteration.build_id, BuildIteration.backtest_results)
            .where(
                BuildIteration.build_id.in_(build_ids),
                BuildIteration.status == "complete",
            )
        )
        for iter_build_id, br in iters_result.all():
            if br:
                ret = float(br.get("total_return_pct") or br.get("total_return") or 0)
                if iter_build_id not in best_returns or ret > best_returns[iter_build_id]:
                    best_returns[iter_build_id] = ret

    # Build a list of available versions with return data
    # Each completed build incremented the version, so we can reconstruct the version history
    versions = []
    if completed_builds:
        # The current version is the latest
        # Work backwards from current version based on number of completed builds
        num_builds = len(completed_builds)
        for i in range(num_builds):
            version = current_version - i
            if version > 0:
                build = completed_builds[i]
                versions.append({
                    "version": version,
                    "best_return_pct": best_returns.get(build.uuid)
                })

    # If no builds but strategy has a version, include it
    if not versions and current_version > 0:
        versions.append({"version": current_version, "best_return_pct": None})

    return {
        "versions": sorted(versions, key=lambda x: x["version"], reverse=True),  # Latest first
        "current_version": current_version,
        "total_builds": len(completed_builds)
    }


@router.post("/strategies/{strategy_id}/license", response_model=LicenseResponse, status_code=status.HTTP_201_CREATED)
async def generate_admin_license(
    strategy_id: str,
    strategy_version: int = Query(..., description="Strategy version number to license"),
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate a free admin license for a strategy (no payment required).

    - Revokes any existing admin licenses for this strategy first so only one
      active admin license exists per strategy at a time.
    - Sets license_type to "admin" and grants a 10-year validity period.
    - The user_id is set to the requesting admin's UUID.
    - Requires a strategy_version to specify which build to pull from docker registry.
    """
    # Verify strategy exists
    result = await db.execute(
        select(Strategy).where(
            Strategy.uuid == strategy_id,
            Strategy.status != "deleted"
        )
    )
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Strategy not found"
        )

    # Revoke any existing admin licenses for this strategy
    existing_result = await db.execute(
        select(License).where(
            License.strategy_id == strategy_id,
            License.license_type == "admin",
            License.status == "active"
        )
    )
    for existing in existing_result.scalars().all():
        existing.status = "revoked"

    # Create new admin license — 10-year validity, no Stripe dependency
    license_obj = License(
        status="active",
        license_type="admin",
        strategy_id=strategy_id,
        user_id=admin_user.uuid,
        strategy_version=strategy_version,
        expires_at=datetime.utcnow() + timedelta(days=3650),
    )
    db.add(license_obj)
    await db.commit()
    await db.refresh(license_obj)

    return license_obj


@router.get("/strategies/{strategy_id}/license", response_model=LicenseResponse)
async def get_admin_license(
    strategy_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Return the current active admin license for a strategy.

    Returns 404 when no active admin license exists so the frontend can
    display the "Generate License" button.
    """
    result = await db.execute(
        select(License).where(
            License.strategy_id == strategy_id,
            License.license_type == "admin",
            License.status == "active"
        )
    )
    license_obj = result.scalar_one_or_none()

    if not license_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active admin license found for this strategy"
        )

    return license_obj


@router.post("/licenses/{license_id}/revoke", response_model=LicenseResponse)
async def revoke_license(
    license_id: str,
    admin_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Revoke a license (admin only).

    Sets the license status to "revoked".  Works for any license type,
    allowing admins to revoke both paid and admin-issued licenses.
    """
    result = await db.execute(select(License).where(License.uuid == license_id))
    license_obj = result.scalar_one_or_none()

    if not license_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="License not found"
        )

    license_obj.status = "revoked"
    await db.commit()
    await db.refresh(license_obj)

    return license_obj


@router.get("/queue/status")
async def get_queue_status(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive build queue status for admin dashboard.

    Returns:
    - Queue length and active builds count
    - Worker statistics (total, active, capacity)
    - List of queued builds with details
    - List of active builds with progress
    - Worker health information
    """
    try:
        # Create Redis client
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

        # Create queue manager
        queue_manager = BuildQueueManager(redis_client)

        # Get comprehensive queue status
        status_data = await queue_manager.get_queue_status(db)

        # Close Redis connection
        await redis_client.close()

        return status_data

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get queue status: {str(e)}"
        )



@router.get("/health")
async def get_system_health(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive system health metrics for monitoring.

    Returns:
    - Overall health score (0-100)
    - System status (healthy/degraded/warning/critical)
    - Queue metrics (length, active builds, avg wait time)
    - Worker metrics (total, active, dead, utilization)
    - Build metrics (success rate, avg duration, counts by status)
    - Recovery metrics (retries, failures)
    """
    try:
        # Create Redis client
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

        # Create monitoring service
        monitoring = MonitoringService(redis_client)

        # Get system health
        health = await monitoring.get_system_health(db)

        # Close Redis connection
        await redis_client.close()

        return health

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system health: {str(e)}"
        )



@router.get("/scaling/recommendations")
async def get_scaling_recommendations(
    current_user: User = Depends(admin_required),
    db: AsyncSession = Depends(get_db)
):
    """
    Get worker auto-scaling recommendations.

    Returns:
    - Scaling recommendation (scale_up/scale_down/maintain)
    - Reason for recommendation
    - Current and suggested worker count
    - Key metrics used for decision
    """
    try:
        # Create Redis client
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

        # Create monitoring service
        monitoring = MonitoringService(redis_client)

        # Get scaling recommendations
        recommendations = await monitoring.get_scaling_recommendations(db)

        # Close Redis connection
        await redis_client.close()

        return recommendations

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scaling recommendations: {str(e)}"
        )

