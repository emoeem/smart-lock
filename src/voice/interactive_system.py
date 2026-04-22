#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能门锁系统 - 交互式管理
=======================================
包含用户注册、查看用户、系统信息等功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import SmartLockBaseSystem, PYAUDIO_AVAILABLE


class InteractiveSmartLockSystem(SmartLockBaseSystem):
    """交互式智能门锁系统"""
    
    def __init__(self):
        """初始化系统"""
        super().__init__()
        print("🔓 智能门锁交互式管理系统初始化完成")
    
    def run_interactive_mode(self):
        """运行交互模式"""
        print("\n" + "="*80)
        print("🎤 智能门锁集成系统 - 交互模式")
        print("="*80)
        
        # 检查必要组件
        if not self.check_components():
            return
        
        while True:
            print("\n🎯 请选择操作:")
            print("1. 用户注册")
            print("2. 查看已注册用户")
            print("3. 系统信息")
            print("4. 退出")
            
            try:
                choice = input("\n请输入选择 (1-4): ").strip()
                
                if choice == "1":
                    self.register_user()
                elif choice == "2":
                    self.list_users()
                elif choice == "3":
                    self.show_system_info()
                elif choice == "4":
                    print("\n👋 退出程序")
                    break
                else:
                    print("❌ 无效选择，请输入 1-4")
                    
            except KeyboardInterrupt:
                print("\n⏹️ 操作被用户中断")
                break
            except Exception as e:
                print(f"\n❌ 操作失败: {e}")
                
        print("\n🔚 程序结束")
    
    def check_components(self):
        """检查系统组件"""
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio不可用，无法进行录音测试")
            return False
            
        if not self.sv_system.pipeline:
            print("❌ CAM++系统初始化失败")
            return False
        
        print("✅ 系统组件检查通过")
        return True


def main():
    """主函数"""
    try:
        system = InteractiveSmartLockSystem()
        
        # 运行交互模式
        system.run_interactive_mode()
        
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