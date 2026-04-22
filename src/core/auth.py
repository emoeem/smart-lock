from abc import ABC, abstractmethod
from src.core.logger import log
from src.security.hash_util import verify_hash
from src.face.face_recognition_json import FaceCapture, FaceRecognitionJSON
import threading
import time
import cv2
import os
import subprocess
import sys
from src.core import face_confirm


class AuthProvider(ABC):
    """统一认证抽象类"""
    @abstractmethod
    def authenticate(self, credential) -> bool:
        pass


class PasswordAuthProvider(AuthProvider):
    """
    基于 hash 的本地密码认证实现
    correct_password 存储的是 hash，而不是明文
    """
    def __init__(self, password_hash: str):
        self.password_hash = password_hash

    def authenticate(self, input_pw) -> bool:
        if not self.password_hash:
            log.error("未配置认证密码 hash")
            return False
        is_valid = verify_hash(input_pw, self.password_hash)
        if is_valid:
            log.info("密码验证通过")
        else:
            log.warning("密码验证失败")

        return is_valid

    def set_password_hash(self, new_hash: str):
        self.password_hash = new_hash
        log.info("认证密码 hash 已更新到内存。")

class VoiceAuthProvider(AuthProvider):
    """
    基于语音识别的认证实现
    """
    def __init__(self):
        # 可选回调：当语音验证开始时触发，签名为 fn()
        self.on_start = None
        # 可选回调：当语音验证产生输出（stdout/stderr）时调用，签名为 fn(text:str, is_error:bool)
        self.on_output = None

    def set_on_start(self, callback):
        self.on_start = callback

    def set_on_output(self, callback):
        self.on_output = callback

    def authenticate(self, credential=None) -> bool:
        """
        触发语音验证流程的入口（当前为占位实现）。
        当被调用时记录日志并触发 `on_start` 回调（如果有）。
        返回值表示是否已认证通过（此处默认 False，实际实现可替换）。
        """
        # 触发 on_start 回调（如果配置）用于 UI/状态更新
        try:
            log.info("语音验证: 启动子进程进行说话人验证")
            if callable(self.on_start):
                try:
                    threading.Thread(target=self.on_start, daemon=True).start()
                except Exception:
                    log.debug("启动语音验证回调线程失败")
        except Exception as e:
            log.debug(f"触发语音验证时发生异常: {e}")

        # 寻找 speaker_verification.py 脚本路径
        try:
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'voice', 'speaker_verification.py'))
            if not os.path.exists(script_path):
                log.error(f"语音验证脚本不存在: {script_path}")
                return False
        except Exception as e:
            log.debug(f"定位语音验证脚本失败: {e}")
            return False

        # 以新的子进程运行脚本，并根据退出码判断是否验证通过
        try:
            cmd = [sys.executable or 'python3', "-u", script_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 以流式读取 stdout/stderr
            def _reader(stream, is_err=False):
                try:
                    for line in iter(stream.readline, ''):
                        if not line:
                            break
                        text = line.rstrip('\n')
                        try:
                            log.debug(f"语音验证输出: {text}")
                            if callable(self.on_output):
                                try:
                                    self.on_output(text, is_err)
                                except Exception:
                                    log.debug("调用 on_output 回调失败")
                        except Exception:
                            pass
                except Exception as e:
                    log.debug(f"读取语音子进程输出异常: {e}")

            t_out = threading.Thread(target=_reader, args=(proc.stdout, False), daemon=True)
            t_err = threading.Thread(target=_reader, args=(proc.stderr, True), daemon=True)
            t_out.start()
            t_err.start()

            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                log.warning("语音验证子进程超时并被终止")

            # 等待读线程结束
            t_out.join(timeout=1.0)
            t_err.join(timeout=1.0)

            ret = proc.returncode
            if ret == 0:
                log.info("语音验证: 通过 (子进程返回码 0)")
                return True
            else:
                log.warning(f"语音验证: 未通过 (返回码 {ret})")
                return False
        except Exception as e:
            log.debug(f"运行语音验证子进程失败: {e}")
            return False

class FaceAuthProvider(AuthProvider):
    """
    基于人脸识别的认证实现
    """
    def __init__(self, poll_interval: float = 5.0, camera_index: int = 0, recognizer_kwargs=None, cooldown_seconds: float = 3.0):
        """
        Args:
            poll_interval: 当检测到人脸后，验证的间隔（秒）
            camera_index: 摄像头索引
            recognizer_kwargs: 传递给 FaceRecognitionJSON 的配置字典（可选）
        """
        self.poll_interval = max(1.0, float(poll_interval))
        self.camera_index = camera_index
        self._stop_event = threading.Event()
        self._thread = None

        # 回调：识别成功时调用，签名为 fn(user_id)
        self.on_success = None

        # 初始化采集与识别器
        try:
            self.capture = FaceCapture(camera_index=self.camera_index)
        except Exception as e:
            log.error(f"初始化 FaceCapture 失败: {e}")
            self.capture = None

        try:
            kwargs = recognizer_kwargs or {}
            self.recognizer = FaceRecognitionJSON(**kwargs)
        except Exception as e:
            log.error(f"初始化 FaceRecognitionJSON 失败: {e}")
        # 冷却秒数（识别成功后进入冷却期，默认 0 不启用）
        try:
            self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        except Exception:
            self.cooldown_seconds = 0.0
            self.recognizer = None

    def set_on_success(self, callback):
        """设置识别成功回调，回调签名为 fn(user_id:str)。"""
        self.on_success = callback

    def start(self):
        """开启后台检测线程（daemon）。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log.info("FaceAuthProvider: 后台人脸检测已启动")

    def stop(self):
        """停止后台检测并释放摄像头。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            if self.capture:
                self.capture.close_camera()
        except Exception:
            pass
        log.info("FaceAuthProvider: 后台人脸检测已停止")

    def _monitor_loop(self):
        """后台循环：检测到人脸则每 N 秒进行一次人脸识别验证。"""
        if not self.capture:
            log.warning("FaceAuthProvider: 无法访问摄像头，停止监控")
            return

        if not self.capture.open_camera():
            log.warning("FaceAuthProvider: 打开摄像头失败，停止监控")
            return

        while not self._stop_event.is_set():
            try:
                ret, frame = self.capture.cap.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue

                # 使用 detector 检测人脸
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.capture.detector.detect_faces(rgb)

                if faces and len(faces) > 0:
                    # 检测到人脸，开始验证
                    log.info("存在人脸开始验证")
                    # 检测到人脸：不立即运行识别，而是注册一个 verifier，等待用户按 Enter 启动识别
                    try:
                        if self.recognizer:
                            frm = frame.copy()
                            def _verifier():
                                try:
                                    result = self.recognizer.identify_user(frm)
                                except Exception as e:
                                    log.debug(f"人脸识别异常 (verifier): {e}")
                                    return
                                if isinstance(result, dict) and result.get('identified'):
                                    best = result.get('best_match') or {}
                                    user_id = best.get('user_id')
                                    if user_id:
                                        log.info(f"userId:{user_id} 人脸验证成功 (由 Enter 触发)")
                                        try:
                                            if callable(self.on_success):
                                                threading.Thread(target=self.on_success, args=(user_id,), daemon=True).start()
                                        except Exception:
                                            log.debug("调用 on_success 回调失败")
                                        # 触发冷却（如果配置）
                                        try:
                                            if self.cooldown_seconds and self.cooldown_seconds > 0:
                                                face_confirm.set_cooldown(self.cooldown_seconds)
                                        except Exception:
                                            log.debug("设置识别冷却失败")
                            # 将 verifier 注册为待触发项
                            try:
                                face_confirm.set_pending_verifier(_verifier, ttl=self.poll_interval)
                            except Exception:
                                log.debug("设置待验证 verifier 失败")
                    except Exception as e:
                        log.debug(f"设置 verifier 时异常: {e}")
                    # 等待下一次循环或用户触发
                    time.sleep(self.poll_interval)
                    continue

                # 未检测到人脸或未识别成功，等待短时间后继续
                time.sleep(0.5)
            except Exception as e:
                log.debug(f"FaceAuthProvider 监控循环异常: {e}")
                time.sleep(1.0)

    def authenticate(self, credential=None) -> bool:
        """立即进行一次人脸检测与识别。如果识别出用户返回 True 并打印信息。"""
        if not self.capture:
            log.error("FaceAuthProvider: 未能初始化摄像头")
            return False

        try:
            if not self.capture.cap:
                if not self.capture.open_camera():
                    return False

            ret, frame = self.capture.cap.read()
            if not ret or frame is None:
                log.debug("FaceAuthProvider: 无法读取摄像头帧")
                return False

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.capture.detector.detect_faces(rgb)
            if not faces:
                log.debug("FaceAuthProvider: 未检测到人脸")
                return False

            log.info("存在人脸开始验证 (立即模式)")
            result = self.recognizer.identify_user(frame) if self.recognizer else None
            if isinstance(result, dict) and result.get('identified'):
                best = result.get('best_match') or {}
                user_id = best.get('user_id')
                if user_id:
                    log.info(f"userId:{user_id} 人脸验证成功 (立即模式)")
                    try:
                        if callable(self.on_success):
                            threading.Thread(target=self.on_success, args=(user_id,), daemon=True).start()
                    except Exception:
                        log.debug("调用 on_success 回调失败")
                    # 触发冷却（如果配置）
                    try:
                        if hasattr(self, 'cooldown_seconds') and self.cooldown_seconds and self.cooldown_seconds > 0:
                            face_confirm.set_cooldown(self.cooldown_seconds)
                    except Exception:
                        log.debug("设置识别冷却失败")
                    return True

        except Exception as e:
            log.debug(f"FaceAuthProvider.authenticate 异常: {e}")

        return False


class CompositeAuthProvider(AuthProvider):
    """
    组合认证器：可以包含多个子认证器，支持 'any' (任意通过) 或 'all' (全部通过) 模式。
    子认证器应实现 `authenticate(credential) -> bool`。
    """
    def __init__(self, providers, mode='any'):
        self.providers = providers or []
        self.mode = mode.lower() if isinstance(mode, str) else 'any'

    def authenticate(self, credential) -> bool:
        if not self.providers:
            log.warning("未配置任何子认证器，拒绝访问。")
            return False

        results = []
        for p in self.providers:
            try:
                ok = p.authenticate(credential)
            except Exception as e:
                log.debug(f"子认证器执行异常: {e}")
                ok = False
            results.append(bool(ok))

        if self.mode == 'all':
            return all(results)
        # 默认为 any
        return any(results)

    def set_password_hash(self, new_hash: str):
        """如果包含 `PasswordAuthProvider`，则将新的 hash 下发给所有子认证器。"""
        for p in self.providers:
            if hasattr(p, 'set_password_hash'):
                try:
                    p.set_password_hash(new_hash)
                except Exception:
                    log.debug("向子认证器更新密码 hash 失败")


def build_auth_provider_from_config(lock_config: dict):
    """
    根据 `lock_config` 字段构建认证器。
    - lock_config['lock_features'] : 列表，可能包含 'PASSWORD' 等标识
    - lock_config['lock_password'] : 存储密码 hash（base64 编码）

    返回一个 AuthProvider（可能是 CompositeAuthProvider 或 单一 PasswordAuthProvider）。
    """
    if not isinstance(lock_config, dict):
        return None

    features = lock_config.get('lock_features', []) or []
    pw_hash = lock_config.get('lock_password')

    providers = []
    for f in features:
        key = str(f).upper()
        if key == 'PASSWORD':
            providers.append(PasswordAuthProvider(pw_hash))
        if key == 'VOICE':
            providers.append(VoiceAuthProvider())
        if key == 'FACE':
            # 确保人脸特征文件使用 src/face/user_features.json（避免运行目录差异导致创建新空文件）
            try:
                base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'face'))
                features_path = os.path.join(base, 'user_features.json')
            except Exception:
                features_path = 'user_features.json'
            # 从配置中读取冷却秒数（可选）：face_cooldown_seconds
            cooldown = 0.0
            try:
                cooldown = float(lock_config.get('face_cooldown_seconds', 0) or 0)
            except Exception:
                cooldown = 0.0
            providers.append(FaceAuthProvider(recognizer_kwargs={'features_file': features_path}, cooldown_seconds=cooldown))

    if not providers:
        # 回退到单一基于密码的认证（即便密码可能为 None）
        return PasswordAuthProvider(pw_hash)

    if len(providers) == 1:
        return providers[0]

    # 默认使用任意通过（OR）策略，可在 config 中扩展为 'auth_mode'
    mode = lock_config.get('auth_mode', 'any')
    return CompositeAuthProvider(providers, mode=mode)