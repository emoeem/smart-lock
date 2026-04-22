import json
import os
import uuid
import socket
import re
import subprocess
from .logger import log


class ConfigManager:
    def __init__(self, config_path="config/config.json"):
        self.config_path = config_path
        self.config_data = {
            "deviceId": "",
            "name": "",
            "model": "",
            "softwareVersion": "1.0.0",
            "isRegister": False,   # 默认未注册
            "lock_config": {
                "lock_password": "123456",
                "lock_status": "locked",
                "lock_features": ["PASSWORD", "REMOTE_ACCESS"],
            }
        }
        self.load_config()

    # -------------------- 基础信息获取 --------------------

    def get_mac_address(self):
        """获取物理网卡 MAC 地址"""
        try:
            mac = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
            return mac
        except Exception:
            return "00:00:00:00:00:00"

    def get_ip_address(self):
        """获取当前本地 IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def get_system_model(self):
        """
        获取系统/设备型号（不使用 platform）
        优先级：
        1. /proc/device-tree/model（Raspberry Pi）
        2. /etc/hostname
        3. uname
        """
        # 1. Raspberry Pi 官方方式
        try:
            model_path = "/proc/device-tree/model"
            if os.path.exists(model_path):
                with open(model_path, "r") as f:
                    return f.read().strip('\x00')
        except Exception:
            pass

        # 2. hostname
        try:
            with open("/etc/hostname", "r") as f:
                hostname = f.read().strip()
                if hostname:
                    return hostname
        except Exception:
            pass

        # 3. uname
        try:
            uname = os.uname()
            return f"{uname.sysname}-{uname.machine}"
        except Exception:
            pass

        return "UnknownDevice"

    def generate_device_name(self, device_id: str):
        """
        生成设备名称：SL-xxxxxxx
        """
        return f"SL-{device_id[:7].upper()}"

    # -------------------- 配置加载与初始化 --------------------

    def load_config(self):
        """从本地加载配置，若无则初始化"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config_data.update(json.load(f))
                self.save_config()
                log.info(f"成功加载本地配置。DeviceID: {self.config_data['deviceId']}")
            except Exception as e:
                log.error(f"加载配置文件失败: {e}")
        else:
            self.init_first_run()

    def init_first_run(self):
        """首次运行：生成设备 ID，并初始化 name / model"""
        new_id = self._generate_device_id()

        self.config_data["deviceId"] = new_id
        self.config_data["model"] = self.get_system_model()
        self.config_data["name"] = self.generate_device_name(new_id)

        self.save_config()
        log.info(
            f"首次运行初始化完成: "
            f"deviceId={new_id}, "
            f"name={self.config_data['name']}, "
            f"model={self.config_data['model']}"
        )

    def _generate_device_id(self):
        """生成 32 位设备 ID"""
        return str(uuid.uuid4()).replace("-", "")[:32]

    def save_config(self):
        """持久化配置到 JSON 文件"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config_data, f, indent=4)
        except Exception as e:
            log.error(f"保存配置文件失败: {e}")

    # -------------------- 对外接口 --------------------

    def get_registration_payload(self):
        """构建发送给 Spring Boot 的注册报文"""
        payload = self.config_data.copy()
        payload["ip"] = self.get_ip_address()
        payload["mac"] = self.get_mac_address()
        return payload

    def set_register_status(self, status: bool = True):
        """设置设备注册状态并持久化"""
        self.config_data["isRegister"] = status
        self.save_config()
        log.info(f"设备注册状态已更新: isRegister={status}")

    def apply_remote_update(self, update: dict) -> bool:
        """应用远程下发的配置更新（支持忽略字段、深度合并、点路径更新）"""
        try:
            if not isinstance(update, dict):
                log.error("远程配置更新必须为 JSON 对象。")
                return False

            # ---- 可配置的忽略字段（根级）----
            IGNORE_ROOT_KEYS = {
                "deviceId",
                "userId",
                "model",
                "isRegister",
            }

            # ---- 允许的根级字段 ----
            ALLOW_ROOT_KEYS = {
                "name",
                "softwareVersion",
                "lock_config",
            }

            def _is_ignored_root(key: str) -> bool:
                return key in IGNORE_ROOT_KEYS or key not in ALLOW_ROOT_KEYS

            def _set_by_path(d, path_parts, value):
                if not path_parts:
                    return False

                # 根级字段校验
                if _is_ignored_root(path_parts[0]):
                    log.warning(f"尝试远程修改受保护字段 {path_parts[0]}，已忽略。")
                    return False

                cur = d
                for p in path_parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]

                cur[path_parts[-1]] = value
                return True

            def _merge(dst, src, depth=0):
                for k, v in src.items():
                    # 仅在根级应用忽略/允许规则
                    if depth == 0 and _is_ignored_root(k):
                        log.warning(f"远程配置包含忽略字段 {k}，已跳过。")
                        continue

                    if isinstance(v, dict) and isinstance(dst.get(k), dict):
                        _merge(dst[k], v, depth + 1)
                    else:
                        dst[k] = v

            # ---- 情况 1：点路径更新 ----
            if "key" in update and "value" in update:
                path = str(update["key"]).split(".")
                ok = _set_by_path(self.config_data, path, update["value"])
                if ok:
                    self.save_config()
                    log.info(f"已应用远程单项配置更新: {update['key']}")
                return ok

            # ---- 情况 2：整体 / 部分字典合并 ----
            _merge(self.config_data, update)
            self.save_config()
            log.info(f"已合并远程配置更新: {list(update.keys())}")
            return True

        except Exception as e:
            log.error(f"应用远程配置更新失败: {e}")
            return False

