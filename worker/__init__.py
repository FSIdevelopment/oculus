"""
Local Training Worker - Redis consumer for ML training jobs on M3 Max.

This worker receives training jobs from Redis, runs the existing ML pipeline,
and reports results back via Redis.
"""

__version__ = "1.0.0"

