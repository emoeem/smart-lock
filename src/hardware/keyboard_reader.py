from evdev import InputDevice, categorize, ecodes, list_devices
from src.core.logger import log
from src.core import face_confirm
import threading


class USBKeyboardListener:
    def __init__(self, device_path=None):
        self.device_path = device_path
        self.input_buffer = ""
        self.callback = None
        self.running = True
        self.on_plus = None
        self.MAX_LEN = 6  # 门锁密码长度

    def find_keyboard(self):
        """
        自动寻找真正产生数字键和回车键的键盘设备
        """
        for path in list_devices():
            device = InputDevice(path)
            if ecodes.EV_KEY not in device.capabilities():
                continue
            keys = device.capabilities().get(ecodes.EV_KEY, [])
            # 同时具备数字键和回车键，基本可判定为主键盘
            if (ecodes.KEY_1 in keys or ecodes.KEY_KP1 in keys) and \
               (ecodes.KEY_ENTER in keys or ecodes.KEY_KPENTER in keys):
                log.info(f"选中键盘设备: {device.name} ({device.path})")
                return device.path
        log.warning("未自动找到合适键盘，使用默认路径")
        return self.device_path

    def start_listening(self, on_submit, on_update=None, on_plus=None):
        """
        on_submit: 输入完成（回车或满长度）后的回调
        on_update: 每次缓冲变化时调用，用于实时显示（可选）
        """
        self.callback = on_submit
        self.update_callback = on_update
        self.on_plus = on_plus
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        log.info("USB 键盘监听线程已启动")

    def stop(self):
        self.running = False

    def _run(self):
        try:
            path = self.find_keyboard()
            if not path:
                log.warning("未找到键盘设备路径，键盘监听线程退出")
                return
            device = InputDevice(path)
            log.info(f"监听输入设备: {device.name} ({device.path})")
            # 独占设备，防止输入泄露到控制台
            device.grab()
            for event in device.read_loop():
                if not self.running:
                    break
                if event.type != ecodes.EV_KEY:
                    continue
                key_event = categorize(event)
                # 只处理按下事件
                if key_event.keystate != 1:
                    continue
                key_code = key_event.keycode
                if isinstance(key_code, list):
                    key_code = key_code[0]
                
                # ========= 回车 =========
                if key_code in ("KEY_ENTER", "KEY_KPENTER"):
                    # 如果存在待确认的人脸，则按 Enter 触发人脸确认（优先），否则继续作为密码提交
                    try:
                        if face_confirm.has_pending():
                            face_confirm.trigger_pending()
                            # 清空输入缓冲避免误提交
                            self.input_buffer = ""
                            continue
                    except Exception:
                        pass
                    if self.input_buffer:
                        # 提交前也先把最终值推送一次（以保证前端最新）
                        if self.update_callback:
                            try:
                                self.update_callback(self.input_buffer)
                            except Exception:
                                pass
                        self.callback(self.input_buffer)
                        self.input_buffer = ""
                # ========= 退格 =========
                elif key_code == "KEY_BACKSPACE":
                    self.input_buffer = self.input_buffer[:-1]
                    if self.update_callback:
                        try:
                            self.update_callback(self.input_buffer)
                        except Exception:
                            pass

                # ========= 数字键处理 =========
                elif key_code.startswith("KEY_"):
                    key = key_code.replace("KEY_", "")
                    # 小键盘 '+ / KPPLUS' 或主键盘 'PLUS'
                    if key in ("KPPLUS", "PLUS"):
                        if self.on_plus:
                            try:
                                self.on_plus()
                            except Exception:
                                pass
                        continue
                    # 主键盘数字
                    if key.isdigit():
                        self.input_buffer += key
                    # 小键盘数字 KP0 ~ KP9
                    elif key.startswith("KP") and key[2:].isdigit():
                        self.input_buffer += key[2:]
                    else:
                        continue
                    log.debug(f"当前输入缓冲: {'*' * len(self.input_buffer)}")
                    # 达到最大长度自动提交
                    if self.update_callback:
                        try:
                            self.update_callback(self.input_buffer)
                        except Exception:
                            pass
                    # 达到最大长度自动提交
                    if len(self.input_buffer) >= self.MAX_LEN:
                        self.callback(self.input_buffer)
                        self.input_buffer = ""
                

        except Exception as e:
            log.error(f"键盘监听发生错误: {e}")