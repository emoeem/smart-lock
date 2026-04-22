#!/usr/bin/env bash
set -e

# =========================
# 配置区
# =========================
PROJECT_DIR="$HOME/smart_lock"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
MAIN_PY="$PROJECT_DIR/main.py"
URL="http://127.0.0.1:8000"
LOCK_FILE="/tmp/smart_lock_panel.lock"

export DISPLAY=:0
export XAUTHORITY=/home/emo/.Xauthority

# =========================
# 防止重复启动
# =========================
if [ -f "$LOCK_FILE" ]; then
    echo "门禁面板已在运行，拒绝重复启动"
    exit 1
fi
touch "$LOCK_FILE"

cleanup() {
    echo "正在关闭门禁系统..."

    [ -n "$PY_PID" ] && kill "$PY_PID" 2>/dev/null || true
    [ -n "$FIREFOX_PID" ] && kill "$FIREFOX_PID" 2>/dev/null || true

    rm -f "$LOCK_FILE"
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"

# =========================
# 启动 Python 后端
# =========================
echo "启动 Python 后端..."
"$PYTHON" "$MAIN_PY" &
PY_PID=$!

# 等待后端启动
sleep 5

# =========================
# 启动 Firefox（kiosk）
# =========================
echo "启动 Firefox 门禁面板..."
firefox \
  "$URL" &
FIREFOX_PID=$!

sleep 5
# =========================
# 任一进程退出就结束全部
# =========================
while true; do
    if ! kill -0 "$PY_PID" 2>/dev/null; then
        echo "Python 后端已退出，关闭浏览器"
        break
    fi

    if ! kill -0 "$FIREFOX_PID" 2>/dev/null; then
        echo "浏览器已关闭，停止后端"
        break
    fi

    sleep 1
done

clear

exit 0
