#!/bin/bash
echo "📊 CCTV System Status"
echo "===================="
echo ""

# Cek proses
if pgrep -f "python.*app.py" > /dev/null; then
    echo "🟢 CCTV System: RUNNING"
    echo "   PID: $(pgrep -f 'python.*app.py')"
    echo "   URL: http://localhost:5000"
else
    echo "🔴 CCTV System: STOPPED"
fi

echo ""
echo "📁 Files Status:"
for file in app.py templates/index.html yolov3.weights yolov3.cfg coco.names; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (missing)"
    fi
done

echo ""
echo "💾 Disk Usage:"
du -sh ~/cctv_system
