# Redis Persistence Configuration for Oculus Strategy

This document provides configuration guidelines for Redis persistence to ensure build queue durability.

## Overview

The Oculus Strategy platform uses Redis for:
- Build queue management (sorted set: `oculus:build_queue`)
- Worker health tracking
- Progress updates (pub/sub)
- Build stop signals

To prevent data loss during Redis restarts or crashes, we recommend configuring Redis with **AOF (Append-Only File)** persistence.

## Recommended Configuration

### 1. AOF Persistence (Recommended)

Add the following to your `redis.conf`:

```conf
# Enable AOF persistence
appendonly yes

# AOF filename
appendfilename "appendonly.aof"

# Fsync policy: everysec provides good balance between performance and durability
# Options: always (slow, safest), everysec (recommended), no (fastest, least safe)
appendfsync everysec

# Rewrite AOF file when it grows by 100%
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Don't use fsync during rewrite (better performance)
no-appendfsync-on-rewrite no
```

### 2. RDB Snapshots (Optional, for additional safety)

You can also enable RDB snapshots as a backup:

```conf
# Save snapshot every 900 seconds if at least 1 key changed
save 900 1

# Save snapshot every 300 seconds if at least 10 keys changed
save 300 10

# Save snapshot every 60 seconds if at least 10000 keys changed
save 60 10000

# RDB filename
dbfilename dump.rdb

# Directory for RDB and AOF files
dir /var/lib/redis
```

### 3. Docker Configuration

If running Redis in Docker, mount a volume for persistence:

```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --appendfsync everysec
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"

volumes:
  redis-data:
```

### 4. Environment-Specific Recommendations

#### Development
- Use `appendfsync everysec` for good balance
- Enable both AOF and RDB for safety

#### Production
- Use `appendfsync everysec` (recommended)
- Enable both AOF and RDB
- Set up Redis replication for high availability
- Consider Redis Sentinel or Redis Cluster for automatic failover

## Verification

After configuring persistence, verify it's working:

```bash
# Connect to Redis CLI
redis-cli

# Check AOF status
CONFIG GET appendonly
# Should return: 1) "appendonly" 2) "yes"

# Check fsync policy
CONFIG GET appendfsync
# Should return: 1) "appendfsync" 2) "everysec"

# Check if AOF file exists
ls -lh /var/lib/redis/appendonly.aof
```

## Recovery from Redis Failure

If Redis crashes or restarts:

1. **Automatic Recovery**: Redis will automatically load data from AOF file on startup
2. **Manual Recovery**: If AOF is corrupted, use `redis-check-aof --fix appendonly.aof`
3. **PostgreSQL Fallback**: The build queue system automatically falls back to PostgreSQL if Redis is unavailable

## Monitoring

Monitor Redis persistence health:

```bash
# Check last save time
redis-cli INFO persistence | grep rdb_last_save_time

# Check AOF status
redis-cli INFO persistence | grep aof_enabled

# Check AOF size
redis-cli INFO persistence | grep aof_current_size
```

## Performance Considerations

- **AOF with `everysec`**: ~1 second of data loss in worst case, minimal performance impact
- **AOF with `always`**: No data loss, but ~50% slower writes
- **RDB only**: Faster, but can lose data between snapshots
- **Both AOF + RDB**: Best safety, slightly more disk I/O

## Backup Strategy

1. **Daily backups**: Copy AOF and RDB files to backup location
2. **Retention**: Keep at least 7 days of backups
3. **Test restores**: Periodically test restoring from backups

```bash
# Example backup script
#!/bin/bash
BACKUP_DIR="/backups/redis/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR
cp /var/lib/redis/appendonly.aof $BACKUP_DIR/
cp /var/lib/redis/dump.rdb $BACKUP_DIR/
```

## Additional Resources

- [Redis Persistence Documentation](https://redis.io/docs/management/persistence/)
- [Redis AOF Best Practices](https://redis.io/docs/management/persistence/#append-only-file)
- [Redis Backup Strategies](https://redis.io/docs/management/persistence/#backing-up-redis-data)

