#!/bin/bash
echo "⏹️ Stopping CCTV System..."
pkill -f "python.*app.py"
echo "✅ CCTV System stopped"
