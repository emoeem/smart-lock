import requests
import json
import paho.mqtt.client as mqtt
from src.core.logger import log

class CommunicationManager:
    def __init__(self, config_manager):
        self.cm = config_manager
        self.cfg = config_manager.config_data
        self.server_url = "http://192.168.200.136:8080"
        self.mqtt_broker = "192.168.200.136"
        self.mqtt_port = 1883
        self.mqtt_connected = False
        self.client = None


    def register_device(self):
        """向 Spring Boot 发送注册请求"""
        if self.cfg.get("isRegister", True):
            return

        url = f"{self.server_url}/device/register"
        payload = self.cm.get_registration_payload()
        try:
            log.info(f"正在尝试注册设备到: {url}")
            response = requests.post(url, json=payload, timeout=5)
            # 先保证 HTTP 层可用
            if response.status_code != 200:
                log.warning(f"HTTP 请求失败，状态码: {response.status_code}")
                return False
            # 解析业务返回码
            result = response.json()
            code = result.get("code")
            if code == 200:
                # 业务注册成功 → 写入配置文件
                self.cm.set_register_status(True)
                log.info("设备注册成功, 状态已写入配置文件。")
                return True
            else:
                log.warning(
                    f"设备注册失败，业务码: {code}, message: {result.get('message')}"
                )
        except Exception as e:
            log.error(f"无法连接到注册服务器: {e}")

        log.warning("注册失败，系统将以离线模式运行。")
        return False


    def setup_mqtt(self):
        """初始化 MQTT 连接并订阅主题"""
        if not self.cfg.get("isRegister", False):
            log.warning("设备尚未注册，MQTT 初始化被跳过。")
            return

        device_id = self.cfg["deviceId"]
        self.client = mqtt.Client(client_id=device_id)
        self.client.username_pw_set("admin", "admin")

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        try:
            self.client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.client.loop_start()
        except Exception as e:
            log.error(f"MQTT 连接初始化失败: {e}")


    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.mqtt_connected = True
            log.info("MQTT 已连接")
            # 订阅控制主题
            device_id = self.cfg['deviceId']
            client.subscribe(f"door/{device_id}/command/#")
        else:
            log.error(f"MQTT 连接失败，错误码: {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self.mqtt_connected = False
        log.warning("MQTT 已断开连接，正在等待自动重连...")

    def _on_message(self, client, userdata, msg):
        log.info(f"收到远程指令 [Topic: {msg.topic}]: {msg.payload.decode()}")
        # 这里的指令处理逻辑我们将在后续“远程控制”步骤中完善