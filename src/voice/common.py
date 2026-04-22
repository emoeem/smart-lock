#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能门锁系统 - 公共组件
=======================================
包含所有公共类和功能
"""

import os
import sys
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

try:
    import pyaudio
    import wave
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from modelscope.pipelines import pipeline
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import ast
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.io import wavfile
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def resample_audio(audio_file, target_sample_rate=16000):
    """使用scipy重采样音频文件"""
    if not SCIPY_AVAILABLE:
        print("⚠️ scipy不可用，无法进行音频重采样")
        return audio_file
    
    try:
        # 读取原始音频文件
        sample_rate, audio_data = wavfile.read(audio_file)
        
        # 如果已经是目标采样率，直接返回原文件
        if sample_rate == target_sample_rate:
            return audio_file
        
        # 计算新的长度
        new_length = int(len(audio_data) * target_sample_rate / sample_rate)
        
        # 使用scipy进行重采样
        resampled_data = signal.resample(audio_data, new_length)
        
        # 确保数据类型正确
        if audio_data.dtype == np.int16:
            resampled_data = np.int16(resampled_data)
        elif audio_data.dtype == np.int32:
            resampled_data = np.int32(resampled_data)
        elif audio_data.dtype == np.float32:
            resampled_data = np.float32(resampled_data)
        elif audio_data.dtype == np.float64:
            resampled_data = np.float64(resampled_data)
        
        # 创建新文件名
        base_name, ext = os.path.splitext(audio_file)
        resampled_file = f"{base_name}_resampled{ext}"
        
        # 保存重采样后的音频
        wavfile.write(resampled_file, target_sample_rate, resampled_data)
        
        print(f"✅ 音频重采样完成: {sample_rate}Hz -> {target_sample_rate}Hz")
        return resampled_file
        
    except Exception as e:
        print(f"❌ 音频重采样失败: {e}")
        return audio_file


class SpeakerVerificationEngine:
    """CAM++说话人验证引擎"""
    
    def __init__(self):
        """初始化验证引擎"""
        self.config = {
            'sample_rate': 44100,
            'threshold': 0.65,
            'model_name': 'damo/speech_campplus_sv_zh-cn_16k-common',
            'record_seconds': 5
        }
        
        if MODELSCOPE_AVAILABLE:
            try:
                self.pipeline = pipeline(
                    task='speaker-verification',
                    model="/home/emo/.cache/modelscope/hub/models/damo/speech_campplus_sv_zh-cn_16k-common",
                    # model=self.config['model_name'],
                    model_revision='v1.0.0',
                    local_files_only=True
                )
                print("✅ CAM++模型初始化成功")
            except Exception as e:
                print(f"❌ CAM++模型初始化失败: {e}")
                self.pipeline = None
        else:
            self.pipeline = None
    
    def extract_embedding(self, audio_file):
        """提取说话人特征向量"""
        if not self.pipeline:
            print("❌ CAM++模型未初始化")
            return None
        
        try:
            print(f"🔍 提取音频特征向量: {audio_file}")
            # 使用scipy进行音频重采样，避免采样率不匹配问题
            if SCIPY_AVAILABLE:
                resampled_file = resample_audio(audio_file, 16000)
            else:
                resampled_file = audio_file
            # 调用pipeline进行特征提取
            result = self.pipeline([resampled_file], output_emb=True)
            if 'embs' in result and len(result['embs']) > 0:
                embedding_vector = result['embs'][0]
                print(f"✅ 成功提取特征向量，维度: {len(embedding_vector)}")
                
                # 如果创建了重采样文件，删除它以节省空间
                if resampled_file != audio_file and os.path.exists(resampled_file):
                    try:
                        os.remove(resampled_file)
                    except:
                        pass
                
                return embedding_vector
            else:
                print("❌ 无法提取特征向量")
                return None
                
        except Exception as e:
            print(f"❌ 特征向量提取失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def verify_speaker(self, audio_file, template_data):
        """验证说话人"""
        if not self.pipeline:
            print("❌ CAM++模型未初始化")
            return 0.0
        
        try:
            # print(f"🔍 开始验证说话人...")
            print(f"   输入音频: {audio_file}")
            recorded_emb = self.extract_embedding(audio_file)
            if recorded_emb is None or len(recorded_emb) == 0:
                print("❌ 无法提取录音的嵌入向量")
                return 0.0
            
            template_emb = self.extract_template_embedding(template_data)
            if template_emb is None or len(template_emb) == 0:
                print("❌ 无法获取模板嵌入向量")
                return 0.0
            
            similarity = self.calculate_similarity(recorded_emb, template_emb)
            
            print(f"🎯 相似度分数: {similarity:.4f}")
            print(f"📋 验证结果: {'✅ 通过' if similarity >= self.config['threshold'] else '❌ 失败'}")
            
            return similarity
            
        except Exception as e:
            print(f"❌ 验证过程错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def extract_template_embedding(self, template_data):
        """从模板数据中提取embedding向量"""
        try:
            if isinstance(template_data, dict):
                template_str = template_data.get('template_data')
                if isinstance(template_str, str):
                    try:
                        template_emb = json.loads(template_str)
                        print(f"✅ 成功解析数据库向量，长度: {len(template_emb)}")
                        return template_emb
                    except:
                        try:
                            template_emb = ast.literal_eval(template_str)
                            return template_emb
                        except Exception as parse_error:
                            print(f"❌ 向量解析失败: {parse_error}")
                            return None
                elif isinstance(template_str, bytes):
                    try:
                        data_str = template_str.decode('utf-8')
                        template_emb = json.loads(data_str)
                        print(f"✅ 成功解析数据库向量，长度: {len(template_emb)}")
                        return template_emb
                    except json.JSONDecodeError:
                        try:
                            template_emb = ast.literal_eval(data_str)
                            return template_emb
                        except:
                            print("❌ 无法解析二进制数据")
                            return None
                else:
                    print(f"❌ 不支持的模板数据类型: {type(template_str)}")
                    return None
                    
            elif isinstance(template_data, bytes) and template_data:
                try:
                    data_str = template_data.decode('utf-8')
                    template_emb = json.loads(data_str)
                    return template_emb
                except:
                    try:
                        template_emb = ast.literal_eval(data_str)
                        return template_emb
                    except Exception as parse_error:
                        print(f"⚠️ 向量解析失败: {parse_error}")
                        return None
                        
            elif isinstance(template_data, str) and os.path.exists(template_data):
                return self.extract_embedding(template_data)
            elif isinstance(template_data, (list, tuple)):
                return template_data
            else:
                print(f"❌ 不支持的模板数据类型: {type(template_data)}")
                return None
                
        except Exception as e:
            print(f"❌ 模板向量提取失败: {e}")
            return None
    
    def calculate_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        try:
            vec1_array = np.array(vec1).flatten()
            vec2_array = np.array(vec2).flatten()
            
            print(f"📏 向量1维度: {len(vec1_array)}")
            print(f"📏 向量2维度: {len(vec2_array)}")
            
            if len(vec1_array) != len(vec2_array):
                max_dim = max(len(vec1_array), len(vec2_array))
                min_dim = min(len(vec1_array), len(vec2_array))
                
                print(f"⚠️ 检测到维度不匹配: {len(vec1_array)} vs {len(vec2_array)}")
                
                if len(vec1_array) == 192 and len(vec2_array) == 256:
                    print("🔧 192维 -> 256维向量补齐")
                    padded_vec1 = np.zeros(max_dim)
                    padded_vec1[:min_dim] = vec1_array
                    vec1_array = padded_vec1
                    
                elif len(vec1_array) == 256 and len(vec2_array) == 192:
                    print("🔧 256维 -> 192维向量裁剪")
                    vec1_array = vec1_array[:min_dim]
                    
                else:
                    print(f"❌ 不支持的维度组合: {len(vec1_array)} vs {len(vec2_array)}")
                    return 0.0
            
            if SKLEARN_AVAILABLE:
                similarity = cosine_similarity([vec1_array], [vec2_array])[0][0]
                return float(similarity)
            else:
                dot_product = np.dot(vec1_array, vec2_array)
                norm1 = np.linalg.norm(vec1_array)
                norm2 = np.linalg.norm(vec2_array)
                
                if norm1 < 1e-8 or norm2 < 1e-8:
                    return 0.0
                
                similarity = dot_product / (norm1 * norm2)
                return float(similarity)
                
        except Exception as e:
            print(f"❌ 向量相似度计算错误: {e}")
            import traceback
            traceback.print_exc()
            return 0.0


class MySQLUserDatabaseManager:
    """MySQL用户数据库管理器"""
    
    def __init__(self):
        """初始化数据库管理器"""
        self.connected = False
        
        if MYSQL_AVAILABLE:
            try:
                self.config = {
                    'host': 'localhost',
                    'user': 'root',
                    'password': '123456',
                    'database': 'smart_lock_system',
                    'port': 3306
                }
                
                self.connection = mysql.connector.connect(**self.config)
                self.connected = True
                print("✅ MySQL数据库连接成功")
                self._check_tables()
                
            except Exception as e:
                print(f"⚠️ MySQL连接失败: {e}")
                print("💡 将使用模拟数据库")
                self.connection = None
        else:
            self.connection = None
        
        self.mock_users = self._create_mock_users()
    
    def _check_tables(self):
        """检查必要的表是否存在"""
        if not self.connected:
            return
            
        try:
            cursor = self.connection.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'smart_lock_system' 
                AND table_name = 'voice_templates'
            """)
            
            exists = cursor.fetchone()[0] > 0
            cursor.close()
            
            if not exists:
                print("⚠️ voice_templates表不存在，请执行SQL文件创建表")
            else:
                print("✅ 数据库表检查通过")
                
        except Exception as e:
            print(f"⚠️ 数据库表检查失败: {e}")
    
    def _create_mock_users(self):
        """创建模拟用户数据"""
        return {
            'user001': {
                'user_id': 'user001',
                'name': '张三',
                'access_level_id': 1,
                'is_active': True,
                'voice_templates': [
                    {
                        'template_id': 1,
                        'template_name': '张三_template_1',
                        'template_data': json.dumps([0.1] * 192).encode('utf-8'),
                        'sample_rate': 44100,
                        'feature_dim': 192
                    }
                ]
            },
            'user002': {
                'user_id': 'user002',
                'name': '李四',
                'access_level_id': 1,
                'is_active': True,
                'voice_templates': [
                    {
                        'template_id': 2,
                        'template_name': '李四_template_1',
                        'template_data': json.dumps([0.2] * 192).encode('utf-8'),
                        'sample_rate': 44100,
                        'feature_dim': 192
                    }
                ]
            }
        }
    
    def get_connection(self):
        """获取数据库连接"""
        return self.connection
    
    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            print("🔌 MySQL连接已关闭")

    def add_user(self, user_data, voice_embeddings):
        """添加用户及其声纹模板到数据库"""
        try:
            if self.connected:
                if self.connection.in_transaction:
                    self.connection.rollback()
                
                cursor = self.connection.cursor()
                
                query_user = """
                INSERT INTO users (
                    user_id, name, access_level_id, is_active,
                    allowed_start_hour, allowed_end_hour, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(query_user, (
                    user_data['user_id'],
                    user_data['name'],
                    user_data['access_level_id'],
                    user_data.get('is_active', 1),
                    user_data.get('allowed_start_hour', 0),
                    user_data.get('allowed_end_hour', 23),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                
                query_template = """
                INSERT INTO voice_templates (
                    user_id, template_name, template_data,
                    feature_dim, is_active
                ) VALUES (%s, %s, %s, %s, %s)
                """
                
                template_count = 0
                for i, embedding in enumerate(voice_embeddings):
                    if isinstance(embedding, np.ndarray):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = embedding
                    
                    embedding_json = json.dumps(embedding_list)
                    feature_dim = len(embedding_list)
                    
                    cursor.execute(query_template, (
                        user_data['user_id'],
                        f"{user_data['name']}_template_{i+1}",
                        embedding_json,
                        feature_dim,
                        1
                    ))
                    template_count += 1
                
                self.connection.commit()
                cursor.close()
                
                print(f"✅ 用户 {user_data['name']} 已添加到MySQL数据库")
                print(f"✅ 已保存 {template_count} 个声纹模板到voice_templates表")
                print(f"💾 用户ID: {user_data['user_id']}")
                return True
            else:
                user_id = user_data['user_id']
                self.mock_users[user_id] = user_data
                self.mock_users[user_id]['voice_templates'] = []
                
                for i, embedding in enumerate(voice_embeddings):
                    if isinstance(embedding, np.ndarray):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = embedding
                    
                    template_info = {
                        'template_id': i + 1,
                        'user_id': user_id,
                        'template_name': f"{user_data['name']}_template_{i+1}",
                        'template_data': json.dumps(embedding_list),
                        'feature_dim': len(embedding_list),
                        'is_active': True
                    }
                    self.mock_users[user_id]['voice_templates'].append(template_info)
                
                print(f"✅ 用户 {user_data['name']} 已添加到模拟数据库")
                print(f"✅ 已保存 {len(voice_embeddings)} 个声纹模板到模拟数据库")
                return True
                
        except mysql.connector.Error as err:
            if self.connected and self.connection.in_transaction:
                self.connection.rollback()
            print(f"❌ MySQL错误: {err}")
            return False
        except Exception as e:
            print(f"❌ 添加用户失败: {e}")
            import traceback
            traceback.print_exc()
            if self.connected and self.connection.in_transaction:
                self.connection.rollback()
            return False

    def get_all_users(self):
        """获取所有用户及其声纹模板"""
        if self.connected:
            try:
                cursor = self.connection.cursor(dictionary=True)
                
                query_users = """
                SELECT u.user_id, u.name, u.access_level_id, u.is_active,
                       u.allowed_start_hour, u.allowed_end_hour,
                       u.created_at, u.last_access_time
                FROM users u
                WHERE u.is_active = 1
                ORDER BY u.created_at DESC
                """
                cursor.execute(query_users)
                user_results = cursor.fetchall()
                
                users = {}
                for row in user_results:
                    user_id = row['user_id']
                    users[user_id] = {
                        'user_id': user_id,
                        'name': row['name'],
                        'access_level': row['access_level_id'],
                        'is_active': bool(row['is_active']),
                        'allowed_start_hour': row.get('allowed_start_hour', 0),
                        'allowed_end_hour': row.get('allowed_end_hour', 23),
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'last_access_time': row['last_access_time'].isoformat() if row['last_access_time'] else None,
                        'voice_templates': []
                    }
                
                if users:
                    user_ids = list(users.keys())
                    placeholders = ', '.join(['%s'] * len(user_ids))
                    
                    query_templates = f"""
                    SELECT template_id, user_id, template_name, 
                           template_data, feature_dim, created_at,
                           is_active
                    FROM voice_templates
                    WHERE user_id IN ({placeholders}) AND is_active = 1
                    ORDER BY user_id, template_id
                    """
                    
                    cursor.execute(query_templates, tuple(user_ids))
                    template_results = cursor.fetchall()
                    
                    for template_row in template_results:
                        user_id = template_row['user_id']
                        if user_id in users:
                            template_info = {
                                'template_id': template_row['template_id'],
                                'template_name': template_row['template_name'],
                                'template_data': template_row['template_data'],
                                'feature_dim': template_row['feature_dim'],
                                'created_at': template_row['created_at'].isoformat() if template_row['created_at'] else None,
                                'is_active': bool(template_row['is_active'])
                            }
                            users[user_id]['voice_templates'].append(template_info)
                
                cursor.close()
                print(f"📊 从MySQL数据库加载了 {len(users)} 个用户")
                return users
                
            except Exception as e:
                print(f"MySQL查询错误: {e}")
                import traceback
                traceback.print_exc()
                return self.mock_users
        else:
            return self.mock_users

    def log_access(self, user_id, user_name, similarity_score, access_granted, failure_reason=None):
        """记录访问日志"""
        if not self.connected:
            return True
            
        try:
            cursor = self.connection.cursor()
            
            query = """
            INSERT INTO access_logs (
                user_id, user_name, similarity_score, access_granted,
                failure_reason, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                user_id,
                user_name,
                similarity_score,
                1 if access_granted else 0,
                failure_reason,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            self.connection.commit()
            cursor.close()
            
            if access_granted and user_id:
                self._update_last_access(user_id)
            
            return True
        except Exception as e:
            print(f"❌ 记录访问日志失败: {e}")
            return False

    def _update_last_access(self, user_id):
        """更新用户最后访问时间"""
        try:
            cursor = self.connection.cursor()
            query = """
            UPDATE users 
            SET last_access_time = %s 
            WHERE user_id = %s
            """
            
            cursor.execute(query, (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                user_id
            ))
            
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"⚠️ 更新用户最后访问时间失败: {e}")


class SmartLockBaseSystem:
    """智能门锁基础系统"""
    
    def __init__(self):
        """初始化系统"""
        self.config = {
            'sample_rate': 44100,
            'record_seconds': 10,
            'threshold': 0.31
        }
        
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
        self.channels = 1
        self.chunk = 1024
        
        self.sv_system = SpeakerVerificationEngine()
        self.db_manager = MySQLUserDatabaseManager()
        
        if PYAUDIO_AVAILABLE:
            self.audio = pyaudio.PyAudio()
        
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            "data/registrations",
            "data/voice_templates", 
            "data/audio_recordings"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        print("📁 必要的目录结构已创建")
    
    def find_usb_microphone(self):
        """查找USB麦克风设备"""
        if not PYAUDIO_AVAILABLE:
            return None
            
        print("🔍 查找USB麦克风设备...")
        
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 and ('USB' in info['name'] or 'Usb' in info['name']):
                device_name = info['name']
                print(f"✅ 找到USB麦克风: 设备{i} ({device_name})")
                return i
        
        print("⚠️ 未找到USB麦克风，使用默认输入设备")
        try:
            default_device = self.audio.get_default_input_device_info()
            print(f"🎤 使用默认设备: {default_device['name']}")
            return default_device['index']
        except:
            print("❌ 无法获取默认输入设备")
            return None
    
    def record_audio(self, duration=None, sample_name="sample"):
        """录制音频样本"""
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio不可用，无法录音")
            return None
            
        if duration is None:
            duration = self.config['record_seconds']
            
        device_index = self.find_usb_microphone()
        if device_index is None:
            print("❌ 未找到可用的录音设备")
            return None
        
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.config['sample_rate'],
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk
            )
            
            print(f"\n🎤 开始录音 {duration} 秒...")
            print(f"📢 请对着麦克风清晰地说出: '{sample_name}'")
            print("🔴 录音中...")
            
            frames = []
            total_chunks = int(self.config['sample_rate'] / self.chunk * duration)
            
            for i in range(total_chunks):
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)
                
                if i % 10 == 0:
                    progress = (i / total_chunks) * 100
                    print(f"⏳ 录音进度: {progress:.1f}%", end='\r')
            
            stream.stop_stream()
            stream.close()
            
            print("\n✅ 录音完成")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = f"data/audio_recordings/voice_{sample_name}_{timestamp}.wav"
            
            wf = wave.open(audio_file, 'wb')
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
            wf.setframerate(self.config['sample_rate'])
            wf.writeframes(b''.join(frames))
            wf.close()
            
            print(f"💾 音频保存到: {audio_file}")
            return audio_file
            
        except Exception as e:
            print(f"❌ 录音失败: {e}")
            return None
    
    def register_user(self):
        """注册用户"""
        print("\n🔓 开始用户注册流程")
        print("=" * 60)
        
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio不可用，无法进行录音")
            return False
        
        if not self.sv_system.pipeline:
            print("❌ CAM++系统未初始化")
            return False
        
        user_info = self._get_user_input()
        if not user_info:
            return False
        
        user_name = user_info['name']
        
        print(f"\n🎯 开始收集语音样本...")
        voice_samples = self._collect_voice_samples(user_name, num_samples=3)
        
        if len(voice_samples) < 2:
            print("❌ 语音样本数量不足，无法完成注册")
            return False
        
        embeddings = []
        for sample in voice_samples:
            if 'embedding' in sample and sample['embedding'] is not None:
                embeddings.append(sample['embedding'])
        
        if not embeddings:
            print("❌ 无法提取有效的声纹向量")
            return False
        
        user_data = {
            'user_id': f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'name': user_name,
            'access_level_id': user_info['access_level'],
            'is_active': True,
            'allowed_start_hour': 0,
            'allowed_end_hour': 23
        }
        
        success = self.db_manager.add_user(user_data, embeddings)
        
        if success:
            print(f"\n🎉 用户 '{user_name}' 注册成功!")
            print(f"✅ 已保存 {len(embeddings)} 个声纹模板到数据库")
            print(f"🔐 用户ID: {user_data['user_id']}")
            print(f"🔐 访问级别: {user_info['access_level']}")
            
            if embeddings:
                print(f"📏 特征向量维度: {len(embeddings[0])}")
                print(f"💾 存储到表: voice_templates")
                print(f"📊 数据类型: BLOB (JSON二进制)")
            return True
        else:
            print(f"\n❌ 用户注册失败")
            return False
    
    def _get_user_input(self):
        """获取用户输入信息"""
        print(f"\n📝 请输入用户注册信息")
        print("=" * 40)
        
        while True:
            user_name = input("👤 请输入用户姓名: ").strip()
            if user_name:
                break
            else:
                print("❌ 用户姓名不能为空，请重新输入")
        
        while True:
            try:
                access_level = input("🔐 请输入访问级别 (1-10, 默认5): ").strip()
                if not access_level:
                    access_level = 5
                else:
                    access_level = int(access_level)
                
                if 1 <= access_level <= 10:
                    break
                else:
                    print("❌ 访问级别必须在 1-10 之间")
            except ValueError:
                print("❌ 请输入有效的数字")
        
        description = input("📄 用户描述 (可选): ").strip()
        
        user_info = {
            'name': user_name,
            'access_level': access_level,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
        
        return user_info

    def _collect_voice_samples(self, user_name, num_samples=3):
        """收集多个语音样本"""
        print(f"\n📢 开始为用户 '{user_name}' 收集语音样本")
        print(f"📊 计划收集 {num_samples} 个样本")
        print("=" * 60)
        
        samples = []
        sample_texts = [
            f"我的名字是{user_name}",
            f"{user_name}正在注册",
            "欢迎使用语音门锁系统",
            "我是授权用户",
            "语音验证测试"
        ]
        
        for i in range(num_samples):
            print(f"\n🎯 样本 {i+1}/{num_samples}")
            print("-" * 40)
            
            if i < len(sample_texts):
                sample_text = sample_texts[i]
            else:
                sample_text = f"语音样本{i+1} - {user_name}"
            
            audio_file = self.record_audio(sample_name=f"{user_name}_sample_{i+1}")
            
            if audio_file:
                embedding = self.sv_system.extract_embedding(audio_file)
                
                if embedding is not None and len(embedding) > 0:
                    sample_info = {
                        'audio_file': audio_file,
                        'embedding': embedding,
                        'sample_text': sample_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    samples.append(sample_info)
                    print(f"✅ 样本 {i+1} 收集成功")
                else:
                    print(f"❌ 样本 {i+1} 嵌入向量提取失败")
            else:
                print(f"❌ 样本 {i+1} 录音失败")
            
            if i < num_samples - 1:
                print("⏳ 等待 2 秒...")
                time.sleep(2)
        
        print(f"\n📊 语音样本收集完成:")
        print(f"   成功收集: {len(samples)}/{num_samples} 个样本")
        
        return samples

    def verify_speaker(self):
        """验证说话人（门锁验证）"""
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio不可用，无法进行录音验证")
            return False, 0.0
        
        if not self.sv_system.pipeline:
            print("❌ CAM++系统未初始化")
            return False, 0.0
        
        audio_file = self.record_audio(sample_name="verification")
        if not audio_file:
            print("❌ 验证录音失败")
            return False, 0.0
        
        users = self.db_manager.get_all_users()
        if not users:
            print("❌ 数据库中没有已注册用户")
            if audio_file and os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"🗑️ 已删除验证失败音频: {audio_file}")
            return False, 0.0
        
        best_user_id = None
        best_similarity = 0.0
        best_user_name = None
        
        for user_id, user_info in users.items():
            if not user_info.get('is_active', True):
                continue
            
            voice_templates = user_info.get('voice_templates', [])
            if not voice_templates:
                continue
            
            template_dict = voice_templates[0]
            similarity = self.sv_system.verify_speaker(audio_file, template_dict)
            if similarity > best_similarity:
                best_similarity = similarity
                best_user_id = user_id
                best_user_name = user_info.get('name', 'Unknown')
        
        access_granted = False
        failure_reason = None
        
        if best_user_id and best_similarity >= self.sv_system.config['threshold']:
            access_granted = True
            failure_reason = None
            print(f"👤 识别用户: {best_user_name}")
            print(f"🎯 相似度: {best_similarity:.4f}")
            self.unlock_door()
            
        else:
            access_granted = False
            if not best_user_id:
                failure_reason = "未找到匹配用户"
                print(f"\n🚫 验证失败!")
                print(f"🎯 最高相似度: {best_similarity:.4f}")
                print(f"🔒 拒绝开门 - 未找到授权用户")
            else:
                failure_reason = f"相似度 {best_similarity:.4f} 低于阈值 {self.sv_system.config['threshold']}"
                print(f"\n🚫 验证失败!")
                print(f"👤 最相似用户: {best_user_name}")
                print(f"🎯 相似度: {best_similarity:.4f}")
                print(f"🔒 拒绝开门 - 相似度不足")
            
            self.deny_access()
            
            # 删除验证失败的音频文件
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    print(f"🗑️ 已删除验证失败音频: {audio_file}")
                except Exception as e:
                    print(f"⚠️ 删除音频文件失败: {e}")
        
        self.db_manager.log_access(
            best_user_id,
            best_user_name,
            best_similarity,
            access_granted,
            failure_reason
        )
        
        return access_granted, best_similarity

    def list_users(self):
        """列出已注册的用户"""
        try:
            users = self.db_manager.get_all_users()
            if not users:
                print("📭 数据库中没有已注册的用户")
                return
            
            print(f"\n📋 已注册用户列表 (共 {len(users)} 个用户)")
            print("=" * 80)
            
            for i, (user_id, user_info) in enumerate(users.items(), 1):
                name = user_info.get('name', '未知')
                access_level = user_info.get('access_level', user_info.get('access_level_id', 0))
                is_active = user_info.get('is_active', True)
                created_at = user_info.get('created_at', '')
                last_access_time = user_info.get('last_access_time', '')
                voice_templates = user_info.get('voice_templates', [])
                
                print(f"\n👤 用户 #{i}")
                print(f"   📝 ID: {user_id}")
                print(f"   👤 姓名: {name}")
                print(f"   🔐 访问级别: {access_level}")
                print(f"   {'✅ 激活' if is_active else '❌ 禁用'}")
                print(f"   📅 注册时间: {created_at}")
                print(f"   ⏰ 最后访问: {last_access_time if last_access_time else '从未访问'}")
                print(f"   🎵 声音模板: {len(voice_templates)} 个")
                
                for j, template in enumerate(voice_templates, 1):
                    print(f"     📄 模板{j}: {template.get('template_name', '未知')}")
                    print(f"       📏 特征维度: {template.get('feature_dim', '未知')}")
                    print(f"       ⭐ 置信度: {template.get('confidence_score', 1.0):.3f}")
                
            print(f"\n{'='*80}")
            
        except Exception as e:
            print(f"❌ 获取用户列表时出错: {e}")

    def show_system_info(self):
        """显示系统信息"""
        print("\n📊 系统信息:")
        print(f"   PyAudio状态: {'✅ 可用' if PYAUDIO_AVAILABLE else '❌ 不可用'}")
        print(f"   ModelScope状态: {'✅ 可用' if MODELSCOPE_AVAILABLE else '❌ 不可用'}")
        print(f"   MySQL状态: {'✅ 可用' if MYSQL_AVAILABLE else '❌ 不可用'}")
        print(f"   采样率: {self.config['sample_rate']} Hz")
        print(f"   录音时长: {self.config['record_seconds']} 秒")
        print(f"   验证阈值: {self.sv_system.config['threshold']}")
        print(f"   CAM++模型: {'✅ 已加载' if self.sv_system.pipeline else '❌ 未加载'}")
        print(f"   数据库连接: {'✅ 已连接' if self.db_manager.connected else '❌ 未连接 (使用模拟数据)'}")
        
        if self.db_manager.connected:
            try:
                cursor = self.db_manager.connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM voice_templates")
                template_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM access_logs")
                log_count = cursor.fetchone()[0]
                
                cursor.close()
                
                print(f"\n📊 数据库统计:")
                print(f"   用户数量: {user_count}")
                print(f"   声纹模板数量: {template_count}")
                print(f"   访问日志数量: {log_count}")
            except:
                pass

    def unlock_door(self):
        """执行开门操作"""
        try:
            print("🔐 执行开门指令...")
            time.sleep(1)
            print("🔓 门锁已解锁 - 门已打开")
            print("✅ 开门操作完成")
            
        except Exception as e:
            print(f"❌ 开门操作失败: {e}")

    def deny_access(self):
        """执行拒绝访问操作"""
        try:
            print("🔐 执行拒绝访问指令...")
            time.sleep(0.5)
            print("🔒 门锁保持锁定状态")
            print("✅ 拒绝访问操作完成")
            
        except Exception as e:
            print(f"❌ 拒绝访问操作失败: {e}")

    def cleanup(self):
        """清理资源"""
        if PYAUDIO_AVAILABLE:
            try:
                self.audio.terminate()
                print("\n🔌 音频资源已清理")
            except:
                pass
        
        if self.db_manager:
            try:
                self.db_manager.close()
            except:
                pass