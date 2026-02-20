"""Email notification service using Resend API."""
import logging
import resend
from app.config import settings
from app.models.user import User
from app.models.strategy import Strategy
from app.models.strategy_build import StrategyBuild
from app.models.license import License

logger = logging.getLogger(__name__)

# Configure Resend API
resend.api_key = settings.RESEND_API_KEY

# Oculus brand colors
OCULUS_PRIMARY = "#6366f1"  # Indigo
OCULUS_SUCCESS = "#10b981"  # Green
OCULUS_WARNING = "#f59e0b"  # Amber
OCULUS_DANGER = "#ef4444"   # Red
OCULUS_DARK = "#1f2937"     # Dark gray
OCULUS_LIGHT = "#f9fafb"    # Light gray


def get_email_template(title: str, content: str, cta_text: str = None, cta_url: str = None, cta_color: str = OCULUS_PRIMARY) -> str:
    """
    Generate a branded email template.

    Args:
        title: Email title/heading
        content: HTML content for the email body
        cta_text: Optional call-to-action button text
        cta_url: Optional call-to-action button URL
        cta_color: Button background color (default: Oculus primary)

    Returns:
        Complete HTML email template
    """
    cta_button = ""
    if cta_text and cta_url:
        cta_button = f"""
        <div style="text-align: center; margin: 30px 0;">
            <a href="{cta_url}" style="display: inline-block; background-color: {cta_color}; color: white; padding: 14px 28px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 16px;">
                {cta_text}
            </a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
            <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f3f4f6; padding: 40px 20px;">
                <tr>
                    <td align="center">
                        <table width="600" cellpadding="0" cellspacing="0" style="background-color: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                            <!-- Header with logo and brand -->
                            <tr>
                                <td style="background: linear-gradient(135deg, {OCULUS_PRIMARY} 0%, #4f46e5 100%); padding: 40px 40px 30px 40px; text-align: center;">
                                    <div style="background-color: white; display: inline-block; padding: 12px 24px; border-radius: 8px; margin-bottom: 20px;">
                                        <h1 style="margin: 0; color: {OCULUS_PRIMARY}; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                                            OCULUS
                                        </h1>
                                    </div>
                                    <h2 style="margin: 0; color: white; font-size: 24px; font-weight: 600; line-height: 1.3;">
                                        {title}
                                    </h2>
                                </td>
                            </tr>

                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px;">
                                    {content}
                                    {cta_button}
                                </td>
                            </tr>

                            <!-- Footer -->
                            <tr>
                                <td style="background-color: {OCULUS_LIGHT}; padding: 30px 40px; border-top: 1px solid #e5e7eb;">
                                    <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px; line-height: 1.6;">
                                        Best regards,<br>
                                        <strong style="color: {OCULUS_DARK};">The Oculus Team</strong>
                                    </p>
                                    <p style="margin: 15px 0 0 0; color: #9ca3af; font-size: 12px; line-height: 1.5;">
                                        © {settings.FRONTEND_URL.replace('http://', '').replace('https://', '').split('/')[0]} • Algorithmic Trading Platform
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
    </html>
    """


class EmailService:
    """Service for sending email notifications via Resend."""
    
    @staticmethod
    def send_build_complete_email(user: User, strategy: Strategy, build: StrategyBuild) -> bool:
        """
        Send email notification for completed build.

        Args:
            user: User who owns the build
            strategy: Strategy that was built
            build: Completed build

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            if not settings.RESEND_API_KEY:
                logger.warning("RESEND_API_KEY not configured, skipping email")
                return False

            build_url = f"{settings.FRONTEND_URL}/strategies/{strategy.uuid}/builds/{build.uuid}"

            content = f"""
            <p style="margin: 0 0 20px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Hi <strong>{user.name}</strong>,
            </p>
            <p style="margin: 0 0 25px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Great news! Your strategy build has completed successfully. 🎉
            </p>

            <div style="background-color: {OCULUS_LIGHT}; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {OCULUS_SUCCESS};">
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">Build Summary</p>
                <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Strategy:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600; text-align: right;">{strategy.name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Build ID:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-family: monospace; text-align: right;">{build.uuid[:8]}...</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Iterations:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600; text-align: right;">{build.iteration_count}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Tokens Used:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600; text-align: right;">{build.tokens_consumed:.2f}</td>
                    </tr>
                </table>
            </div>

            <p style="margin: 25px 0 0 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Your strategy is now ready to deploy and start generating signals.
            </p>
            """

            html_content = get_email_template(
                title="Strategy Build Complete! 🎉",
                content=content,
                cta_text="View Build Details",
                cta_url=build_url,
                cta_color=OCULUS_SUCCESS
            )

            resend.Emails.send({
                "from": "Oculus Algorithms <noreply@oculusalgorithms.com>",
                "to": user.email,
                "subject": f"Strategy Build Complete: {strategy.name}",
                "html": html_content
            })

            logger.info(f"Build complete email sent to {user.email} for build {build.uuid}")
            return True

        except Exception as e:
            logger.error(f"Failed to send build complete email: {e}")
            return False
    
    @staticmethod
    def send_build_failed_email(user: User, strategy: Strategy, build: StrategyBuild) -> bool:
        """
        Send email notification for failed build.

        Args:
            user: User who owns the build
            strategy: Strategy that failed to build
            build: Failed build

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            if not settings.RESEND_API_KEY:
                logger.warning("RESEND_API_KEY not configured, skipping email")
                return False

            build_url = f"{settings.FRONTEND_URL}/strategies/{strategy.uuid}/builds/{build.uuid}"

            content = f"""
            <p style="margin: 0 0 20px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Hi <strong>{user.name}</strong>,
            </p>
            <p style="margin: 0 0 25px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Unfortunately, your strategy build encountered an error and could not complete.
            </p>

            <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {OCULUS_DANGER};">
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">Build Details</p>
                <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Strategy:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600; text-align: right;">{strategy.name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Build ID:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-family: monospace; text-align: right;">{build.uuid[:8]}...</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Status:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DANGER}; font-size: 14px; font-weight: 600; text-align: right;">{build.status.upper()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Phase:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; text-align: right;">{build.phase or 'N/A'}</td>
                    </tr>
                </table>
            </div>

            <p style="margin: 25px 0 0 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Please review the build logs for more details. You can try building again or contact support if you need assistance.
            </p>
            """

            html_content = get_email_template(
                title="Strategy Build Failed",
                content=content,
                cta_text="View Build Logs",
                cta_url=build_url,
                cta_color=OCULUS_DANGER
            )

            resend.Emails.send({
                "from": "Oculus Algorithms <noreply@oculusalgorithms.com>",
                "to": user.email,
                "subject": f"Strategy Build Failed: {strategy.name}",
                "html": html_content
            })

            logger.info(f"Build failed email sent to {user.email} for build {build.uuid}")
            return True

        except Exception as e:
            logger.error(f"Failed to send build failed email: {e}")
            return False

    @staticmethod
    def send_license_expiry_warning_email(user: User, license: License, strategy: Strategy, days_remaining: int) -> bool:
        """
        Send email notification for expiring license.

        Args:
            user: User who owns the license
            license: License that is expiring
            strategy: Strategy associated with the license
            days_remaining: Number of days until expiration

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            if not settings.RESEND_API_KEY:
                logger.warning("RESEND_API_KEY not configured, skipping email")
                return False

            renew_url = f"{settings.FRONTEND_URL}/strategies/{strategy.uuid}/licenses/{license.uuid}/renew"

            # Customize message based on urgency
            if days_remaining == 1:
                urgency_color = OCULUS_DANGER
                urgency_emoji = "🚨"
                urgency_text = "URGENT: Your license expires tomorrow!"
            elif days_remaining == 3:
                urgency_color = OCULUS_WARNING
                urgency_emoji = "⚠️"
                urgency_text = "Your license expires in 3 days"
            elif days_remaining == 5:
                urgency_color = OCULUS_WARNING
                urgency_emoji = "⏰"
                urgency_text = "Your license expires in 5 days"
            else:  # 15 days
                urgency_color = OCULUS_PRIMARY
                urgency_emoji = "📅"
                urgency_text = "Your license expires in 15 days"

            content = f"""
            <p style="margin: 0 0 20px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Hi <strong>{user.name}</strong>,
            </p>
            <p style="margin: 0 0 25px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                This is a reminder that your license for <strong>{strategy.name}</strong> will expire soon.
            </p>

            <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {urgency_color};">
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">License Details</p>
                <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Strategy:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600; text-align: right;">{strategy.name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">License Type:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; text-align: right;">{license.license_type.title()}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Expires:</td>
                        <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; text-align: right;">{license.expires_at.strftime('%B %d, %Y at %I:%M %p UTC')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Days Remaining:</td>
                        <td style="padding: 6px 0; color: {urgency_color}; font-size: 14px; font-weight: 700; text-align: right;">{days_remaining}</td>
                    </tr>
                </table>
            </div>

            <div style="background-color: #fef2f2; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {OCULUS_DANGER};">
                <p style="margin: 0 0 10px 0; color: {OCULUS_DANGER}; font-size: 15px; font-weight: 700;">⚠️ Important Notice</p>
                <p style="margin: 0; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.6;">
                    When your license expires, your strategy will <strong>stop sending signals immediately</strong>. To continue receiving signals, please renew your license before it expires.
                </p>
            </div>

            <p style="margin: 25px 0 0 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                If you have any questions or need assistance, please don't hesitate to contact our support team.
            </p>
            """

            html_content = get_email_template(
                title=f"{urgency_emoji} {urgency_text}",
                content=content,
                cta_text="Renew License Now",
                cta_url=renew_url,
                cta_color=urgency_color
            )

            resend.Emails.send({
                "from": "Oculus Algorithms <noreply@oculusalgorithms.com>",
                "to": user.email,
                "subject": f"License Expiring Soon: {strategy.name} ({days_remaining} day{'s' if days_remaining != 1 else ''} remaining)",
                "html": html_content
            })

            logger.info(f"License expiry warning email sent to {user.email} for license {license.uuid} ({days_remaining} days)")
            return True

        except Exception as e:
            logger.error(f"Failed to send license expiry warning email: {e}")
            return False

    @staticmethod
    def send_welcome_email(user: User, password: str = None) -> bool:
        """
        Send welcome email to newly registered user.

        Args:
            user: Newly registered user
            password: Optional plain text password (only for admin-created users)

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            if not settings.RESEND_API_KEY:
                logger.warning("RESEND_API_KEY not configured, skipping email")
                return False

            dashboard_url = f"{settings.FRONTEND_URL}/dashboard"

            # Include credentials section only if password is provided (admin-created users)
            credentials_section = ""
            if password:
                credentials_section = f"""
                <div style="background-color: {OCULUS_LIGHT}; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {OCULUS_PRIMARY};">
                    <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">Your Login Credentials</p>
                    <table width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                            <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Email:</td>
                            <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600; text-align: right;">{user.email}</td>
                        </tr>
                        <tr>
                            <td style="padding: 6px 0; color: #6b7280; font-size: 14px;">Password:</td>
                            <td style="padding: 6px 0; color: {OCULUS_DARK}; font-size: 14px; font-family: monospace; text-align: right;">{password}</td>
                        </tr>
                    </table>
                    <p style="margin: 15px 0 0 0; color: #6b7280; font-size: 13px;">
                        Please change your password after your first login for security.
                    </p>
                </div>
                """

            content = f"""
            <p style="margin: 0 0 20px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Hi <strong>{user.name}</strong>,
            </p>
            <p style="margin: 0 0 25px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Welcome to <strong>Oculus Algorithms</strong>! We're thrilled to have you join our platform for AI-powered algorithmic trading strategies.
            </p>

            {credentials_section}

            <div style="background-color: #eff6ff; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {OCULUS_PRIMARY};">
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">🚀 What's Next?</p>
                <p style="margin: 0; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.6;">
                    You'll receive a follow-up email shortly with a complete guide on how to create your first strategy, purchase licenses, and connect to partner platforms.
                </p>
            </div>

            <p style="margin: 25px 0 0 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                In the meantime, feel free to explore your dashboard and familiarize yourself with the platform.
            </p>

            <p style="margin: 20px 0 0 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                If you have any questions, our support team is here to help!
            </p>
            """

            html_content = get_email_template(
                title="Welcome to Oculus! 🎉",
                content=content,
                cta_text="Go to Dashboard",
                cta_url=dashboard_url,
                cta_color=OCULUS_PRIMARY
            )

            resend.Emails.send({
                "from": "Oculus Algorithms <noreply@oculusalgorithms.com>",
                "to": user.email,
                "subject": "Welcome to Oculus Algorithms!",
                "html": html_content
            })

            logger.info(f"Welcome email sent to {user.email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send welcome email: {e}")
            return False

    @staticmethod
    def send_onboarding_email(user: User) -> bool:
        """
        Send comprehensive onboarding guide to new user.

        Args:
            user: Newly registered user

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            if not settings.RESEND_API_KEY:
                logger.warning("RESEND_API_KEY not configured, skipping email")
                return False

            strategies_url = f"{settings.FRONTEND_URL}/strategies"
            marketplace_url = f"{settings.FRONTEND_URL}/marketplace"
            support_email = "support@oculusalgorithms.com"

            content = f"""
            <p style="margin: 0 0 20px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Hi <strong>{user.name}</strong>,
            </p>
            <p style="margin: 0 0 25px 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                Now that you're all set up, let's walk through how to make the most of Oculus Algorithms. This guide will help you create your first AI-powered trading strategy and start generating signals.
            </p>

            <!-- Step 1: Create a Strategy -->
            <div style="background-color: white; padding: 25px; border-radius: 8px; margin: 25px 0; border: 2px solid {OCULUS_LIGHT};">
                <h3 style="margin: 0 0 15px 0; color: {OCULUS_PRIMARY}; font-size: 18px; font-weight: 700;">
                    📊 Step 1: Create Your Strategy
                </h3>
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 15px; line-height: 1.6;">
                    Creating a strategy is simple:
                </p>
                <ol style="margin: 0; padding-left: 20px; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.8;">
                    <li>Navigate to <strong>Strategies</strong> in your dashboard</li>
                    <li>Click <strong>"Create New Strategy"</strong></li>
                    <li>Describe your trading idea in plain English (e.g., "Buy tech stocks when RSI is oversold and momentum is positive")</li>
                    <li>Select your target symbols and timeframe</li>
                    <li>Our AI will design, train, and backtest your strategy automatically</li>
                </ol>
                <p style="margin: 15px 0 0 0; color: #6b7280; font-size: 13px; line-height: 1.6;">
                    💡 <strong>Tip:</strong> The more specific your description, the better the AI can optimize your strategy.
                </p>
            </div>

            <!-- Step 2: Review Your Strategy -->
            <div style="background-color: white; padding: 25px; border-radius: 8px; margin: 25px 0; border: 2px solid {OCULUS_LIGHT};">
                <h3 style="margin: 0 0 15px 0; color: {OCULUS_PRIMARY}; font-size: 18px; font-weight: 700;">
                    📈 Step 2: Review Backtest Results
                </h3>
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 15px; line-height: 1.6;">
                    Once your strategy is built, you can:
                </p>
                <ul style="margin: 0; padding-left: 20px; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.8;">
                    <li><strong>View detailed backtest results</strong> including total return, Sharpe ratio, win rate, and drawdown</li>
                    <li><strong>Analyze performance charts</strong> showing equity curves and trade history</li>
                    <li><strong>Review the AI-generated code</strong> to understand how your strategy works</li>
                    <li><strong>Iterate and improve</strong> by creating new builds with refined parameters</li>
                </ul>
                <p style="margin: 15px 0 0 0; color: #6b7280; font-size: 13px; line-height: 1.6;">
                    💡 <strong>Tip:</strong> Look for strategies with high Sharpe ratios (>1.5) and manageable drawdowns (<20%).
                </p>
            </div>

            <!-- Step 3: Purchase a License -->
            <div style="background-color: white; padding: 25px; border-radius: 8px; margin: 25px 0; border: 2px solid {OCULUS_LIGHT};">
                <h3 style="margin: 0 0 15px 0; color: {OCULUS_PRIMARY}; font-size: 18px; font-weight: 700;">
                    🔑 Step 3: Purchase a License
                </h3>
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 15px; line-height: 1.6;">
                    To deploy your strategy and receive live signals:
                </p>
                <ol style="margin: 0; padding-left: 20px; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.8;">
                    <li>Go to your strategy's detail page</li>
                    <li>Click <strong>"Purchase License"</strong></li>
                    <li>Choose between <strong>Monthly</strong> or <strong>Annual</strong> licensing</li>
                    <li>Complete the secure checkout process</li>
                </ol>
                <p style="margin: 15px 0 0 0; color: #6b7280; font-size: 13px; line-height: 1.6;">
                    💡 <strong>Pricing:</strong> License prices are dynamically calculated based on your strategy's backtest performance. Better performance = higher value!
                </p>
            </div>

            <!-- Step 4: Connect to Partner Platform -->
            <div style="background-color: white; padding: 25px; border-radius: 8px; margin: 25px 0; border: 2px solid {OCULUS_LIGHT};">
                <h3 style="margin: 0 0 15px 0; color: {OCULUS_PRIMARY}; font-size: 18px; font-weight: 700;">
                    🔗 Step 4: Connect to a Partner Platform
                </h3>
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 15px; line-height: 1.6;">
                    Once you have an active license, connect your strategy to receive signals:
                </p>
                <ol style="margin: 0; padding-left: 20px; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.8;">
                    <li>Navigate to <strong>"Licenses"</strong> in your dashboard</li>
                    <li>Click <strong>"Connect Platform"</strong> on your active license</li>
                    <li>Choose your preferred partner (e.g., SignalSynk, Atlas Trade AI, custom webhook)</li>
                    <li>Follow the integration instructions to set up your webhook URL</li>
                    <li>Your strategy will start sending real-time signals automatically</li>
                </ol>
                <p style="margin: 15px 0 0 0; color: #6b7280; font-size: 13px; line-height: 1.6;">
                    💡 <strong>Tip:</strong> Test your webhook connection with a small position size before scaling up.
                </p>
            </div>

            <!-- Additional Options -->
            <div style="background-color: #eff6ff; padding: 20px; border-radius: 8px; margin: 25px 0; border-left: 4px solid {OCULUS_PRIMARY};">
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">🎯 More Options</p>
                <ul style="margin: 0; padding-left: 20px; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.8;">
                    <li><strong>Browse the Marketplace:</strong> Discover and license strategies created by other users</li>
                    <li><strong>Publish Your Strategy:</strong> List your high-performing strategies on the marketplace and earn revenue</li>
                    <li><strong>Manage Multiple Strategies:</strong> Run a portfolio of strategies across different markets and timeframes and on different partner platforms</li>
                </ul>
            </div>

            <!-- Support Section -->
            <div style="background-color: {OCULUS_LIGHT}; padding: 20px; border-radius: 8px; margin: 25px 0;">
                <p style="margin: 0 0 12px 0; color: {OCULUS_DARK}; font-size: 14px; font-weight: 600;">💬 Need Help?</p>
                <p style="margin: 0 0 10px 0; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.6;">
                    Our support team is here to assist you:
                </p>
                <ul style="margin: 0; padding-left: 20px; color: {OCULUS_DARK}; font-size: 14px; line-height: 1.8;">
                    <li>Email us at <a href="mailto:{support_email}" style="color: {OCULUS_PRIMARY}; text-decoration: none; font-weight: 600;">{support_email}</a></li>
                </ul>
            </div>

            <p style="margin: 25px 0 0 0; color: {OCULUS_DARK}; font-size: 16px; line-height: 1.6;">
                We're excited to see what strategies you'll create. Happy trading!
            </p>
            """

            html_content = get_email_template(
                title="Getting Started with Oculus 🚀",
                content=content,
                cta_text="Create Your First Strategy",
                cta_url=strategies_url,
                cta_color=OCULUS_PRIMARY
            )

            resend.Emails.send({
                "from": "Oculus Algorithms <noreply@oculusalgorithms.com>",
                "to": user.email,
                "subject": "Getting Started with Oculus Algorithms - Your Complete Guide",
                "html": html_content
            })

            logger.info(f"Onboarding email sent to {user.email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send onboarding email: {e}")
            return False

