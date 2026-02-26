#!/bin/bash

# Signal Synk Broker Development Server Startup Script
# This script starts both the backend and frontend servers
# Runs as non-root user, handles permission issues gracefully

echo "🚀 Starting Oculus Algorithms Development Servers..."
echo ""

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if running as root and warn
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Warning: Running as root. Consider running as non-root user for security."
    echo "   Continuing anyway..."
    echo ""
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source ~/Development/virtualenv/signalsynk/bin/activate

# Start backend in background
echo "📡 Starting backend server on http://localhost:8000..."
(cd backend && python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Error: Backend server failed to start"
    exit 1
fi

# Start frontend in background
echo "🎨 Starting frontend server on http://localhost:3000..."
cd frontend
# Ensure Homebrew binaries are in PATH for npm/node
export PATH="/opt/homebrew/bin:$PATH"
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 2

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Error: Frontend server failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "✅ Servers starting..."
echo ""
echo "📍 Backend API: http://localhost:8000"
echo "📍 Frontend UI: http://localhost:3000"
echo ""
echo "📝 Login credentials (auto-created on first login):"
echo "   Email: admin@oculusalgorithms.com"
echo "   Password: Admin123!"
echo ""
echo "Press Ctrl+C to stop both servers"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Wait for user interrupt
trap cleanup INT TERM
wait

