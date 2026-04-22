#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能门锁系统 - 说话人验证（自动化运行）
=======================================
自动运行说话人验证流程，不经过菜单选择
"""

import sys
import os

os.environ["JACK_NO_START_SERVER"] = "1"
os.environ["JACK_DISABLE"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import SmartLockBaseSystem, PYAUDIO_AVAILABLE


class SpeakerVerificationSystem(SmartLockBaseSystem):
    """说话人验证系统 - 自动化运行版本"""
    
    def __init__(self):
        """初始化验证系统"""
        super().__init__()
    
    def run_verification(self):
        """运行验证流程"""
        try:
            # 检查必要组件
            if not PYAUDIO_AVAILABLE:
                print("❌ PyAudio不可用，无法进行录音验证")
                return False
            
            if not self.sv_system.pipeline:
                print("❌ CAM++系统未初始化")
                return False
            
            # 直接进行验证
            access_granted, similarity = self.verify_speaker()
            
            # 显示最终结果
            return access_granted
            
        except KeyboardInterrupt:
            print("\n⏹️ 验证被用户中断")
            return False
        except Exception as e:
            print(f"\n❌ 验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    try:
        system = SpeakerVerificationSystem()
        
        # 运行验证
        success = system.run_verification()
        # 使用退出码传递结果：0 表示通过，1 表示失败
        try:
            sys.exit(0 if success else 1)
        except SystemExit:
            raise
        
    except KeyboardInterrupt:
        print("\n⏹️ 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'system' in locals():
            system.cleanup()


if __name__ == '__main__':
    main()