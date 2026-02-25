#!/bin/bash
# Deploy the Strategies Container Build Agent on the signalSynk-Strategies Droplet.
# Run as root from the directory containing main.py and builder.py:
#   bash deploy.sh
set -e

DEPLOY_DIR="/opt/strategies-build-agent"
SERVICE_NAME="strategies-build-agent"

echo "=== Deploying Strategies Container Build Agent ==="

# Ensure pip3 is available
echo "Ensuring pip3 is available..."
if ! command -v pip3 &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3-pip
fi

# Install Python deps
echo "Installing Python dependencies..."
pip3 install fastapi==0.115.0 "uvicorn[standard]==0.30.6" python-dotenv==1.0.1

# Create deploy directory and copy service files
echo "Copying files to $DEPLOY_DIR..."
mkdir -p "$DEPLOY_DIR"
cp main.py builder.py requirements.txt "$DEPLOY_DIR/"

# Create .env file placeholder if it doesn't exist
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    cat > "$DEPLOY_DIR/.env" << 'ENVFILE'
BUILD_API_KEY=REPLACE_WITH_SHARED_SECRET
DO_REGISTRY_URL=registry.digitalocean.com/oculus-strategies
DO_REGISTRY_TOKEN=REPLACE_WITH_DO_TOKEN
ENVFILE
    echo ""
    echo "WARNING: Created $DEPLOY_DIR/.env with placeholder values."
    echo "         Edit it with real credentials before starting the service."
fi

# Create systemd unit
echo "Creating systemd unit..."
cat > "/etc/systemd/system/$SERVICE_NAME.service" << UNIT
[Unit]
Description=Strategies Container Build Agent
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=$DEPLOY_DIR
EnvironmentFile=$DEPLOY_DIR/.env
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8088
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

echo ""
echo "=== Deployment complete ==="
systemctl status "$SERVICE_NAME" --no-pager
echo ""
echo "Next steps:"
echo "  1. Edit $DEPLOY_DIR/.env with real credentials"
echo "  2. systemctl restart $SERVICE_NAME"
echo "  3. curl http://localhost:8088/health"
