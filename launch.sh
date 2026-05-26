#!/bin/bash
echo "🌀 Starting Resonance Sanctuary..."
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python api_server.py &
API_PID=$!
sleep 2
cd electron-frontend
npx electron . &
ELEC_PID=$!
trap "kill $API_PID $ELEC_PID 2>/dev/null; exit" INT TERM
wait $ELEC_PID
kill $API_PID 2>/dev/null
