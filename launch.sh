#!/bin/bash
# Resonance Sanctuary — Universal Launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "🌀 Starting Resonance Sanctuary..."

# Start Python API server
if [ -d "venv" ]; then
    source venv/bin/activate
    python api_server.py &
    API_PID=$!
    echo "   ✅ API server starting..."
    sleep 2
else
    echo "   ⚠️ No virtual environment found. Install dependencies first."
    echo "      python3 -m venv venv"
    echo "      source venv/bin/activate"
    echo "      pip install -r requirements.txt"
    exit 1
fi

# Start Electron frontend
cd electron-frontend
npx electron . &
ELEC_PID=$!
echo "   ✅ Electron GUI launching..."
echo ""
echo "🌀 Resonance Sanctuary is running."
echo "   Close the window or press Ctrl+C to stop."

trap "kill $API_PID $ELEC_PID 2>/dev/null; exit" INT TERM
wait $ELEC_PID
kill $API_PID 2>/dev/null
