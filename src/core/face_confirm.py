import threading
import time
from src.core.logger import log

# 线程安全的全局待确认人脸状态
_lock = threading.Lock()
# _pending can hold either:
#  {'user_id': str, 'callback': callable, 'expires_at': float}
#  or {'verifier': callable, 'expires_at': float}
_pending = None
# 可选回调：当 pending 状态变化时调用，签名为 fn(text:str)
_notify_callback = None


def set_pending_user(user_id: str, callback, ttl: float = 10.0):
    """设置待确认的人脸信息（兼容旧接口），ttl 秒后自动过期并清除。"""
    global _pending
    if user_id is None:
        return

    expires_at = time.time() + float(ttl)
    with _lock:
        _pending = {'user_id': user_id, 'callback': callback, 'expires_at': expires_at}
        log.info(f"人脸识别结果已就绪，等待用户按 Enter 以确认 (userId={user_id})")
        # 通知前端（若已注册回调）
        try:
            if _notify_callback:
                import json
                msg = json.dumps({
                    'type': 'wait',
                    'message': '检测到人脸，请按 Enter 开始人脸验证，保持人脸稳定'
                })
                try:
                    _notify_callback(msg)
                except Exception:
                    log.debug('调用 notify_callback 失败')
        except Exception:
            pass

    # 启动后台线程在 ttl 后清除（非阻塞）
    def _expire(was_set_at):
        time.sleep(ttl)
        with _lock:
            global _pending
            if _pending and _pending.get('expires_at') == was_set_at:
                _pending = None
                log.info("人脸确认超时，已清除待确认状态")

    threading.Thread(target=_expire, args=(expires_at,), daemon=True).start()


def set_pending_verifier(verifier_callable, ttl: float = 10.0):
    """设置一个待触发的 verifier 回调（当用户按 Enter 时执行）。verifier_callable() 将在新线程中被调用。"""
    global _pending
    if not callable(verifier_callable):
        return
    expires_at = time.time() + float(ttl)
    with _lock:
        _pending = {'verifier': verifier_callable, 'expires_at': expires_at}
        log.info("检测到人脸，等待用户按 Enter 开始验证")
        # 通知前端（若已注册回调），提示用户按 Enter
        try:
            if _notify_callback:
                import json
                msg = json.dumps({
                    'type': 'wait',
                    'message': '检测到人脸，请按 Enter 开始人脸验证，保持人脸稳定'
                })
                try:
                    _notify_callback(msg)
                except Exception:
                    log.debug('调用 notify_callback 失败')
        except Exception:
            pass

    def _expire(was_set_at):
        time.sleep(ttl)
        with _lock:
            global _pending
            if _pending and _pending.get('expires_at') == was_set_at:
                _pending = None
                log.info("人脸验证等待超时，已清除待验证状态")

    threading.Thread(target=_expire, args=(expires_at,), daemon=True).start()


def has_pending() -> bool:
    with _lock:
        if not _pending:
            return False
        return _pending.get('expires_at', 0) > time.time()


def trigger_pending() -> bool:
    """触发并执行待确认回调（在新线程中调用），执行后清除 pending。返回是否触发成功。"""
    global _pending
    with _lock:
        if not _pending:
            return False
        if _pending.get('expires_at', 0) <= time.time():
            _pending = None
            return False
        data = _pending
        _pending = None

    # 优先支持 verifier
    verifier = data.get('verifier')
    if callable(verifier):
        try:
            threading.Thread(target=verifier, daemon=True).start()
            log.info("已通过按键触发人脸验证流程 (verifier)")
            return True
        except Exception as e:
            log.debug(f"触发 verifier 回调时出错: {e}")
            return False

    # 兼容老接口：callback(user_id)
    user_id = data.get('user_id')
    cb = data.get('callback')
    try:
        if callable(cb):
            threading.Thread(target=cb, args=(user_id,), daemon=True).start()
            log.info(f"已通过按键触发人脸确认回调 (userId={user_id})")
            return True
    except Exception as e:
        log.debug(f"触发人脸确认回调时出错: {e}")
    return False


def clear_pending():
    global _pending
    with _lock:
        _pending = None


def set_notify_callback(fn):
    """注册通知回调：当 pending 状态创建时，回调会接收到一个字符串（通常为 JSON）。"""
    global _notify_callback
    if fn and callable(fn):
        _notify_callback = fn
    else:
        _notify_callback = None
