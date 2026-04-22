import time
from gpiozero import DigitalOutputDevice
from src.core.logger import log
import os
os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

class LockController:
    def __init__(self, pin=21, on_unlock=None, pulse_time=0.3):
        """
        :param pin: GPIO BCM 编号
        :param on_unlock: 开锁成功回调
        :param pulse_time: 开锁脉冲时间（秒）
        """
        self.lock_pin = pin
        self.pulse_time = pulse_time
        self.on_unlock = on_unlock
        self.is_locked = True
        self.lock_device = None

        self._setup_gpio()

    def _setup_gpio(self):
        try:
            self.lock_device = DigitalOutputDevice(
                self.lock_pin,
                initial_value=False
            )
            log.info(f"GPIO 初始化成功 (gpiozero)，引脚: GPIO{self.lock_pin}")
        except Exception as e:
            log.error(f"GPIO 初始化失败 (gpiozero): {e}")
            self.lock_device = None

    def open_door(self, reason="unknown", user_id="SYSTEM"):
        if not self.lock_device:
            log.error("开锁失败：GPIO 未正确初始化")
            return False

        try:
            log.info(f"执行开锁动作，触发原因: {reason}, user_id: {user_id}")

            # === 开锁脉冲 ===
            self.lock_device.on()
            self.is_locked = False

            time.sleep(self.pulse_time)

            self.lock_device.off()
            self.is_locked = True

            log.info("开锁指令执行完毕，GPIO 已拉低")

            # === 开锁成功回调 ===
            if callable(self.on_unlock):
                try:
                    try:
                        self.on_unlock(reason, user_id)
                    except TypeError:
                        self.on_unlock(reason)
                except Exception as e:
                    log.error(f"开锁事件回调执行失败: {e}")

            return True

        except Exception as e:
            log.error(f"硬件开锁执行异常: {e}")
            return False

    def cleanup(self):
        try:
            if self.lock_device:
                self.lock_device.off()
                self.lock_device.close()
                log.info("GPIO 资源已释放")
        except Exception as e:
            log.error(f"GPIO 清理失败: {e}")
