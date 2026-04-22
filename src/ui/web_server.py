from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
from pathlib import Path
import os

app = FastAPI()
templates = Jinja2Templates(directory="src/ui/templates")
BASE_DIR = Path(__file__).resolve().parents[2]

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "src/ui/static"),
    name="static"
)

# 共享状态引用
system_context = None

# 在主事件循环中保存引用（用于从监听线程向 WebSocket 发送）
keyboard_event_loop = None

# 存放所有连接的键盘 WebSocket 客户端
keyboard_ws_clients = set()


def set_keyboard_event_loop(loop: asyncio.AbstractEventLoop):
    """将运行 Web 服务的 asyncio 事件循环保存起来，供其它线程使用 `run_coroutine_threadsafe` 调度任务。"""
    global keyboard_event_loop
    keyboard_event_loop = loop

def set_context(ctx):
    global system_context
    system_context = ctx
    # 尝试将前端广播函数注册到 face_confirm，以便在检测到人脸时推送通知
    try:
        from src.core import face_confirm
        try:
            face_confirm.set_notify_callback(send_keyboard_to_clients)
        except Exception:
            pass
    except Exception:
        # 任何导入或注册错误均不应阻塞主流程
        pass

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    cfg = system_context.config_manager.config_data

    status_data = {
        "deviceId": cfg["deviceId"],
        "lock_status": "上锁中" if system_context.lock.is_locked else "未上锁",
        "mqtt_status": "已连接服务器" if system_context.comms.mqtt_connected else "未连接服务器",
        "reg_status": "已注册" if cfg.get("isRegister", False) else "离线模式",
        "features": cfg["lock_config"]["lock_features"]
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": status_data
        }
    )


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    """通过 WebSocket 实时推送最新日志"""
    await websocket.accept()
    log_path = "logs/doorlock.log"

    try:
        # 初次连接发送最后10行
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-10:]
                for line in lines:
                    await websocket.send_text(line.strip())

        # 记录文件指针位置，用于增量读取
        file_pointer = os.path.getsize(log_path) if os.path.exists(log_path) else 0

        while True:
            await asyncio.sleep(0.5)  # 半秒轮询，可根据需要调整频率

            # 实时增量读取日志
            if os.path.exists(log_path):
                current_size = os.path.getsize(log_path)
                if current_size > file_pointer:
                    with open(log_path, "r", encoding="utf-8") as f:
                        f.seek(file_pointer)
                        new_lines = f.readlines()
                        for line in new_lines:
                            await websocket.send_text(line.strip())
                    file_pointer = current_size

            # MQTT 掉线提示
            if not system_context.comms.mqtt_connected:
                await websocket.send_text("--- 实时状态检测: MQTT 掉线 ---")

    except Exception as e:
        print(f"WebSocket 断开: {e}")



@app.websocket("/ws/keyboard")
async def websocket_keyboard(websocket: WebSocket):
    """为页面提供键盘输入推送的 websocket 接口。客户端连接后会收到来自后台的输入字符串。"""
    await websocket.accept()
    keyboard_ws_clients.add(websocket)
    try:
        # 保持连接，若客户端发送数据可忽略或用作心跳
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        try:
            keyboard_ws_clients.remove(websocket)
        except KeyError:
            pass


def send_keyboard_to_clients(text: str):
    """从任意线程调用：将文本广播到所有已连接的键盘 websocket 客户端。

    如果当前在非 asyncio 事件循环线程，尝试使用保存的 keyboard_event_loop 进行调度。
    """
    if not keyboard_ws_clients:
        return

    async def _broadcast():
        dead = []
        for ws in list(keyboard_ws_clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for d in dead:
            try:
                keyboard_ws_clients.remove(d)
            except KeyError:
                pass

    try:
        loop = asyncio.get_running_loop()
        # 在事件循环中，直接创建任务
        loop.create_task(_broadcast())
    except RuntimeError:
        # 不在事件循环线程
        if keyboard_event_loop is not None:
            asyncio.run_coroutine_threadsafe(_broadcast(), keyboard_event_loop)
        else:
            # 作为后备，启动新的事件循环来发送（不推荐，但保证尽量发送）
            try:
                asyncio.run(_broadcast())
            except Exception:
                pass


