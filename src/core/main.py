import json
import time
import threading
import uvicorn
import signal
import sys
from src.core.logger import log
from src.core.config import ConfigManager
from src.core.auth import build_auth_provider_from_config
from src.hardware.lock_control import LockController
from src.hardware.keyboard_reader import USBKeyboardListener
from src.network.comms import CommunicationManager
from src.ui.web_server import app, set_context, send_keyboard_to_clients, set_keyboard_event_loop

class SmartLockSystem:
    def __init__(self):
        log.info("=== 智能门锁系统启动中 ===")
        
        # 1. 加载配置与身份识别
        self.config_manager = ConfigManager()
        self.device_id = self.config_manager.config_data['deviceId']
        
        # 2. 初始化硬件控制 (GPIO 21)
        self.lock = LockController(
            pin=21,
            on_unlock=self.report_unlock_event
        )

        
        # 3. 初始化认证器 (从配置读取 lock_config 中的特性)
        self.auth_provider = build_auth_provider_from_config(self.config_manager.config_data.get('lock_config', {}))
        # 人脸认证相关提供器（会在启动时绑定回调并启动监控）
        self._face_providers = []
        try:
            self._attach_face_providers(self.auth_provider)
        except Exception:
            log.debug("绑定人脸提供器时发生异常")
        
        # 4. 网络与远程通信初始化
        self.comms = CommunicationManager(self.config_manager)
        
        # 5. 键盘监听器初始化
        self.keyboard = USBKeyboardListener()

    def _handle_remote_command(self, topic, payload):
        """处理来自 MQTT 的远程指令"""
        try:
            data = json.loads(payload)
            if "command/open" in topic:
                self.lock.open_door(reason=data.get("reason"), user_id=data.get("userId"))
            elif "command/config" in topic:
                log.info("收到远程配置更新，正在同步本地缓存...")
                try:
                    # 支持两种形式：
                    # 1) 字典合并: {"lock_config": {"lock_password":"abc"}}
                    # 2) 单项点路径: {"key": "lock_config.lock_password", "value": "abc"}
                    updated = self.config_manager.apply_remote_update(data)
                    if updated:
                        log.info("远程配置已应用到本地配置文件。")
                        # 更新内存中的相关运行时状态：认证密码与门锁状态
                        try:
                            # 更新认证器密码（如果配置中含有该项）
                            new_pw = self.config_manager.config_data.get('lock_config', {}).get('lock_password')
                            if hasattr(self, 'auth_provider') and hasattr(self.auth_provider, 'set_password_hash'):
                                self.auth_provider.set_password_hash(new_pw)

                            # 更新门锁内存状态（非物理动作，仅同步状态值）
                            new_status = self.config_manager.config_data.get('lock_config', {}).get('lock_status')
                            if new_status in ('locked', 'unlocked'):
                                self.lock.is_locked = (new_status == 'locked')
                        except Exception as e:
                            log.debug(f"更新内存配置状态时发生异常: {e}")
                        if getattr(self.comms, 'client', None) and self.comms.mqtt_connected:
                            ack = {
                                "event": "config_update", 
                                "deviceId": self.device_id, 
                                "userId": data.get("userId"),
                                "reason": "remote_access"
                            }
                            try:
                                self.comms.client.publish(f"door/{self.device_id}/event", json.dumps(ack))
                            except Exception:
                                log.debug("发送配置更新确认失败")
                    else:
                        log.warning("远程配置应用失败。")
                except Exception as e:
                    log.error(f"应用远程配置失败: {e}")
        except Exception as e:
            log.error(f"解析远程指令失败: {e}")

    def on_key_submit(self, password):
        """本地键盘输入回调"""
        log.info(f"监听到按键提交，长度: {len(password)}")
        # 同步推送到 Web UI 的输入框（用于页面实时显示）
        try:
            send_keyboard_to_clients(password)
        except Exception:
            log.debug("向 Web UI 广播键盘输入时发生错误")
        if self.auth_provider.authenticate(password):
            self.lock.open_door(reason="password")
        else:
            log.warning("拒绝访问：密码错误")

    def _on_face_verified(self, user_id: str):
        """人脸验证成功时的回调：触发开门并上报事件。"""
        try:
            # log.info(f"通过回调收到人脸验证成功: userId={user_id}")
            # 调用门锁打开
            self.lock.open_door(reason="face", user_id=user_id)
        except Exception as e:
            log.debug(f"处理人脸验证成功回调时出错: {e}")

    def _trigger_face_verification(self):
        """在检测到 Enter 时触发面部认证提供器的手动验证。"""
        for p in getattr(self, '_face_providers', []):
            if hasattr(p, 'trigger_verification'):
                try:
                    p.trigger_verification()
                except Exception:
                    pass

    def _attach_face_providers(self, provider):
        """递归查找并启动所有支持人脸回调的子认证器。"""
        if provider is None:
            return

        # CompositeAuthProvider 有属性 'providers'
        if hasattr(provider, 'providers'):
            for p in getattr(provider, 'providers'):
                self._attach_face_providers(p)
            return

        # 单一提供器：如果支持 set_on_success，则绑定并启动
        if hasattr(provider, 'set_on_success'):
            try:
                provider.set_on_success(self._on_face_verified)
                # 启动后台监控（若有 start 方法）
                if hasattr(provider, 'start'):
                    provider.start()
                self._face_providers.append(provider)
                log.info("已绑定并启动 FaceAuthProvider")
            except Exception as e:
                log.debug(f"启动 FaceAuthProvider 失败: {e}")

    def start_web_ui(self):
        """启动 FastAPI Web 服务并将事件循环注入 `web_server` 模块供跨线程调度。"""
        set_context(self) # 将系统实例注入 Web 上下文
        log.info("正在启动 Web 管理界面 (Port: 8000)...")
        # 使用 programmatic API 启动 uvicorn，以便我们能拿到并传递事件循环
        import asyncio
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # 将事件循环传给 web_server，以便从其他线程调度 coroutine（如广播 websocket）
        try:
            set_keyboard_event_loop(loop)
        except Exception:
            log.debug("设置 keyboard_event_loop 失败")

        loop.run_until_complete(server.serve())

    def report_unlock_event(self, reason, user_id=None):
        if not self.comms.mqtt_connected:
            log.debug("MQTT 未连接，跳过开门事件上报")
            return

        payload = {
            "event": "unlock",
            "deviceId": self.device_id,
            "reason": reason,
            "userId": user_id,
        }

        topic = f"door/{self.device_id}/event"
        self.comms.client.publish(topic, json.dumps(payload))
        log.info(f"开门事件已上报: {payload}")


    def run(self):
        # A. 如果未注册，尝试注册
        if not self.config_manager.config_data.get("isRegister", False):
            self.comms.register_device()

        # B. 只要已注册，就启动 MQTT
        if self.config_manager.config_data.get("isRegister", False):
            self.comms.setup_mqtt()

            # 绑定远程指令回调
            if self.comms.client:
                self.comms.client.on_message = (
                    lambda c, u, m: self._handle_remote_command(
                        m.topic,
                        m.payload.decode()
                    )
                )

        
        # C. 启动硬件监听线程 (Daemon 模式)
        # 传入 on_update 用于将实时输入推送到 Web UI（避免页面一直显示固定6个掩码）
        self.keyboard.start_listening(on_submit=self.on_key_submit,
                  on_update=lambda buf: send_keyboard_to_clients(buf),
                  on_enter=self._trigger_face_verification)
        
        # D. 在子线程启动 Web UI，避免阻塞主循环
        ui_thread = threading.Thread(target=self.start_web_ui, daemon=True)
        ui_thread.start()

        log.info("系统各模块就绪，开始主循环运行。")
        
        # E. 主循环：监控健康状态
        try:
            while True:
                # 每 60 秒上报一次心跳状态
                if self.comms.mqtt_connected:
                    status_payload = {
                        "deviceId": self.device_id,
                        "lock_status": "locked" if self.lock.is_locked else "unlocked",
                    }
                    self.comms.client.publish(f"door/{self.device_id}/status", str(status_payload))
                time.sleep(60) 
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()

    def shutdown(self):
        log.info("正在关闭系统，释放硬件资源...")
        # 停止面部认证后台线程
        try:
            for p in getattr(self, '_face_providers', []):
                if hasattr(p, 'stop'):
                    p.stop()
        except Exception:
            log.debug("停止 FaceAuthProvider 时发生异常")

        self.lock.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    # 处理系统信号（如 Ctrl+C）
    lock_system = SmartLockSystem()
    
    def signal_handler(sig, frame):
        lock_system.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    
    lock_system.run()