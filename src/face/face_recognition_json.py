#!/usr/bin/env python3
"""
人脸识别系统 - 使用JSON文件存储人脸特征
基于用户提供的方案实现：
1. 使用MTCNN进行人脸检测
2. 使用DeepFace提取人脸特征
3. 将特征向量存储在JSON文件中
4. 提供用户注册和验证功能
"""

import os
# 设置Qt平台插件环境变量，解决Wayland显示问题
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import json
import numpy as np
import cv2
import time
from datetime import datetime
from deepface import DeepFace
from mtcnn import MTCNN
import sys
import random

class FaceRecognitionJSON:
    def __init__(self, features_file='user_features.json', use_multiple_models=False, 
                 primary_model='ArcFace', enable_dynamic_threshold=True):
        """
        初始化人脸识别系统
        
        Args:
            features_file: 用户特征JSON文件路径
            use_multiple_models: 是否使用多模型融合
            primary_model: 主要模型名称 ('ArcFace', 'Facenet', 'VGG-Face', 'OpenFace')
            enable_dynamic_threshold: 是否启用动态阈值
        """
        self.features_file = features_file
        self.detector = MTCNN()  # MTCNN人脸检测器
        
        # 加载已有的用户特征
        self.user_features = self.load_user_features()
        
        # 模型配置
        self.primary_model = primary_model  # 主要模型
        self.use_multiple_models = use_multiple_models  # 是否使用多模型融合
        self.enable_dynamic_threshold = enable_dynamic_threshold  # 是否启用动态阈值
        
        # 阈值配置
        # 根据模型设置不同的基础阈值
        self.model_thresholds = {
            'ArcFace': 1.0,    # ArcFace建议阈值1.0
            'Facenet': 0.6,    # Facenet建议阈值0.6
            'VGG-Face': 0.6,   # VGG-Face建议阈值0.6
            'OpenFace': 0.6    # OpenFace建议阈值0.6
        }
        
        self.base_threshold = self.model_thresholds.get(self.primary_model, 0.6)  # 基础阈值
        self.threshold = self.base_threshold  # 当前阈值
        self.threshold_adjustment = 0.0  # 阈值调整值
        
        # 支持的模型列表
        self.supported_models = ['ArcFace', 'Facenet', 'VGG-Face', 'OpenFace']
        
        # 模型维度信息
        self.model_dimensions = {
            'ArcFace': 512,
            'Facenet': 128,
            'VGG-Face': 4096,
            'OpenFace': 128
        }
        
        # 验证模型选择
        if self.primary_model not in self.supported_models:
            print(f"⚠ 警告: 模型 {self.primary_model} 不在支持列表中，使用默认模型 ArcFace")
            self.primary_model = 'ArcFace'
        
        print("=" * 60)
        print("人脸识别系统 - JSON特征存储版 (优化版)")
        print("=" * 60)
        print(f"特征文件: {self.features_file}")
        print(f"已注册用户: {len(self.user_features)}")
        print(f"主要模型: {self.primary_model} ({self.model_dimensions[self.primary_model]}维)")
        print(f"多模型融合: {'启用' if self.use_multiple_models else '禁用'}")
        print(f"动态阈值: {'启用' if self.enable_dynamic_threshold else '禁用'}")
        print(f"当前阈值: {self.threshold}")
        print("=" * 60)
        
        # 检查特征维度是否匹配
        self.check_and_warn_dimension_mismatch()
        
        # 记录系统启动日志
        self.log_event("system_start", {
            "users_count": len(self.user_features),
            "primary_model": self.primary_model,
            "use_multiple_models": self.use_multiple_models,
            "threshold": self.threshold
        })
    
    def load_user_features(self):
        """
        加载已有的用户特征
        
        Returns:
            dict: 用户特征字典
        """
        try:
            if os.path.exists(self.features_file):
                with open(self.features_file, 'r') as f:
                    data = json.load(f)
                    print(f"✓ 从 {self.features_file} 加载了 {len(data)} 个用户特征")
                    return data
            else:
                print(f"⚠ 特征文件 {self.features_file} 不存在，创建新文件")
                return {}
        except Exception as e:
            print(f"✗ 加载特征文件失败: {e}")
            return {}
    
    def save_user_features(self):
        """
        保存用户特征到JSON文件
        """
        try:
            with open(self.features_file, 'w') as f:
                json.dump(self.user_features, f, indent=2, ensure_ascii=False)
            print(f"✓ 用户特征已保存到 {self.features_file}")
            return True
        except Exception as e:
            print(f"✗ 保存特征文件失败: {e}")
            return False
    
    def log_event(self, event_type, event_data):
        """
        记录系统事件
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "data": event_data
        }
        
        # 打印到控制台
        print(f"[{timestamp}] {event_type}: {event_data}")
        
        # 这里可以添加将日志保存到文件的功能
        # 例如：self.save_log_to_file(log_entry)
    
    def enhance_image(self, image):
        """
        增强图像质量，提高识别准确性
        
        Args:
            image: OpenCV BGR图像
            
        Returns:
            numpy.ndarray: 增强后的图像
        """
        if image is None:
            return None
        
        try:
            # 转换为灰度图像
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 直方图均衡化 - 提高对比度
            enhanced = cv2.equalizeHist(gray)
            
            # 高斯模糊去噪
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            # 自适应直方图均衡化 (CLAHE) - 更好的对比度增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            
            # 转换为BGR格式（如果需要）
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            print("✓ 图像增强完成")
            return enhanced
            
        except Exception as e:
            print(f"✗ 图像增强失败: {e}")
            return image
    
    def augment_image(self, image, augmentations=None):
        """
        数据增强：对图像进行变换以增加多样性
        
        Args:
            image: OpenCV BGR图像
            augmentations: 增强类型列表，可选值: ['rotate', 'flip', 'brightness', 'contrast']
            
        Returns:
            list: 增强后的图像列表
        """
        if image is None:
            return []
        
        if augmentations is None:
            augmentations = ['rotate', 'flip', 'brightness', 'contrast']
        
        augmented_images = [image.copy()]
        
        try:
            # 旋转增强
            if 'rotate' in augmentations:
                # 小角度旋转
                for angle in [-10, -5, 5, 10]:
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h))
                    augmented_images.append(rotated)
            
            # 翻转增强
            if 'flip' in augmentations:
                # 水平翻转
                flipped_h = cv2.flip(image, 1)
                augmented_images.append(flipped_h)
                
                # 垂直翻转
                flipped_v = cv2.flip(image, 0)
                augmented_images.append(flipped_v)
            
            # 亮度调整
            if 'brightness' in augmentations:
                # 增加亮度
                brightened = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
                augmented_images.append(brightened)
                
                # 降低亮度
                darkened = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
                augmented_images.append(darkened)
            
            # 对比度调整
            if 'contrast' in augmentations:
                # 增加对比度
                high_contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
                augmented_images.append(high_contrast)
                
                # 降低对比度
                low_contrast = cv2.convertScaleAbs(image, alpha=0.7, beta=0)
                augmented_images.append(low_contrast)
            
            print(f"✓ 数据增强完成，生成 {len(augmented_images)} 张图像")
            return augmented_images
            
        except Exception as e:
            print(f"✗ 数据增强失败: {e}")
            return [image]
    
    def extract_features(self, image, model_name=None):
        """
        从图像中提取人脸特征（修复版）
        
        Args:
            image: OpenCV BGR图像或图像路径
            model_name: 指定模型名称，如果为None则使用主要模型
            
        Returns:
            numpy.ndarray: 特征向量，如果失败返回None
        """
        try:
            if model_name is None:
                model_name = self.primary_model
            
            # 记录特征提取开始
            self.log_event("feature_extraction_start", {"model": model_name})
            
            result = DeepFace.represent(
                img_path=image,
                model_name=model_name,
                enforce_detection=False,
                align=True
            )
            
            if result:
                features = np.array(result[0]['embedding'])
                
                # 归一化特征（关键！）
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
                    print(f"✓ 特征提取成功并归一化，模型: {model_name}, 范数: {norm:.4f} -> 1.0")
                else:
                    print(f"⚠ 特征范数为零，模型: {model_name}")
                
                print(f"  特征维度: {features.shape}")
                
                # 记录特征提取成功
                self.log_event("feature_extraction_success", {
                    "model": model_name,
                    "dimension": features.shape[0],
                    "norm": float(norm) if norm > 0 else 0
                })
                
                return features
            else:
                print("✗ 特征提取失败: 未检测到人脸")
                self.log_event("feature_extraction_failed", {"reason": "no_face_detected"})
                return None
                
        except Exception as e:
            error_msg = str(e)
            print(f"✗ 特征提取失败 ({model_name}): {error_msg}")
            self.log_event("feature_extraction_error", {
                "model": model_name,
                "error": error_msg
            })
            return None
    
    def extract_multiple_models_features(self, image):
        """
        使用多个模型提取特征并融合
        
        Args:
            image: OpenCV BGR图像或图像路径
            
        Returns:
            numpy.ndarray: 融合后的特征向量，如果失败返回None
        """
        if not self.use_multiple_models:
            # 如果不使用多模型融合，直接使用默认模型
            return self.extract_features(image)
        
        # 使用所有支持的模型
        models = self.supported_models
        all_features = []
        successful_models = []
        
        print(f"使用多模型融合提取特征（使用{len(models)}个模型）...")
        for model in models:
            try:
                features = self.extract_features(image, model_name=model)
                if features is not None:
                    all_features.append(features)
                    successful_models.append(model)
                    print(f"  ✓ {model}: 特征提取成功，维度: {features.shape}")
                else:
                    print(f"  ✗ {model}: 特征提取失败")
            except Exception as e:
                print(f"  ✗ {model}: 错误 - {e}")
        
        if not all_features:
            print("✗ 所有模型特征提取失败，尝试使用单个模型...")
            return self.extract_features(image)
        
        # 对特征进行标准化处理（L2归一化）
        normalized_features = []
        for features in all_features:
            norm = np.linalg.norm(features)
            if norm > 0:
                normalized_features.append(features / norm)
            else:
                normalized_features.append(features)
        
        # 对特征进行加权平均融合
        # 给ArcFace和Facenet更高的权重，因为它们通常性能更好
        weights = {
            'ArcFace': 0.4,
            'Facenet': 0.3,
            'VGG-Face': 0.2,
            'OpenFace': 0.1
        }
        
        weighted_sum = np.zeros_like(normalized_features[0])
        total_weight = 0
        
        for i, model in enumerate(successful_models):
            weight = weights.get(model, 0.1)
            weighted_sum += normalized_features[i] * weight
            total_weight += weight
        
        if total_weight > 0:
            fused_features = weighted_sum / total_weight
        else:
            fused_features = np.mean(normalized_features, axis=0)
        
        print(f"✓ 多模型融合成功，使用模型: {successful_models}")
        print(f"  融合后特征维度: {fused_features.shape}")
        
        # 记录多模型融合事件
        self.log_event("multi_model_fusion", {
            "successful_models": successful_models,
            "weights": {model: weights.get(model, 0.1) for model in successful_models},
            "dimension": fused_features.shape[0]
        })
        
        return fused_features
    
    def compare_features(self, features1, features2):
        """
        比较两个特征向量的相似度（修复版）
        
        Args:
            features1: 第一个特征向量
            features2: 第二个特征向量
            
        Returns:
            float: 距离（越小越相似）
        """
        # 确保都是numpy数组
        if isinstance(features1, list):
            features1 = np.array(features1)
        if isinstance(features2, list):
            features2 = np.array(features2)
        
        # 检查特征维度
        if features1.shape != features2.shape:
            print(f"⚠ 特征维度不匹配: {features1.shape} != {features2.shape}")
            # 尝试调整维度
            min_len = min(len(features1), len(features2))
            features1 = features1[:min_len]
            features2 = features2[:min_len]
        
        # 归一化特征向量（关键修复！）
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 > 0:
            features1_norm = features1 / norm1
        else:
            features1_norm = features1.copy()
            print("⚠ 特征1范数为零")
        
        if norm2 > 0:
            features2_norm = features2 / norm2
        else:
            features2_norm = features2.copy()
            print("⚠ 特征2范数为零")
        
        # 计算余弦距离（更稳定）
        cosine_similarity = np.dot(features1_norm, features2_norm)
        cosine_distance = 1 - cosine_similarity
        
        # 计算欧氏距离（归一化后）
        euclidean_distance = np.linalg.norm(features1_norm - features2_norm)
        
        print(f"  范数: {norm1:.4f} vs {norm2:.4f}")
        print(f"  余弦距离: {cosine_distance:.4f}")
        print(f"  欧氏距离: {euclidean_distance:.4f}")
        
        # 返回较小的距离（更宽松）
        return min(cosine_distance, euclidean_distance)
    
    def register_user(self, user_id, name, image, use_enhancement=True):
        """
        注册新用户
        
        Args:
            user_id: 用户ID
            name: 用户姓名
            image: OpenCV BGR图像或图像路径
            use_enhancement: 是否使用图像增强
            
        Returns:
            bool: 是否注册成功
        """
        # 检查用户是否已存在
        if user_id in self.user_features:
            print(f"✗ 用户 {user_id} 已存在!")
            return False
        
        # 图像增强
        if use_enhancement and isinstance(image, np.ndarray):
            print("应用图像增强...")
            enhanced_image = self.enhance_image(image)
            if enhanced_image is not None:
                image = enhanced_image
        
        # 提取特征
        features = self.extract_features(image)
        if features is None:
            print("✗ 无法提取人脸特征，注册失败")
            return False
        
        # 保存用户特征
        self.user_features[user_id] = {
            'name': name,
            'features': features.tolist(),  # 转换为列表以便JSON序列化
            'register_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_count': 1,  # 样本数量
            'model_used': self.primary_model  # 记录使用的模型
        }
        
        # 保存到文件
        if self.save_user_features():
            print(f"✓ 用户 {user_id} ({name}) 注册成功!")
            print(f"  使用模型: {self.primary_model}, 特征维度: {len(features)}")
            return True
        else:
            print(f"✗ 用户 {user_id} 注册失败: 无法保存特征")
            return False
    
    def register_user_with_multiple_images(self, user_id, name, images):
        """
        使用多张图像注册用户（取特征均值）
        
        Args:
            user_id: 用户ID
            name: 用户姓名
            images: 图像列表（OpenCV BGR图像或图像路径）
            
        Returns:
            bool: 是否注册成功
        """
        # 检查用户是否已存在
        if user_id in self.user_features:
            print(f"✗ 用户 {user_id} 已存在!")
            return False
        
        if not images:
            print("✗ 图像列表为空")
            return False
        
        all_features = []
        successful_count = 0
        
        print(f"开始使用 {len(images)} 张图像注册用户...")
        for i, image in enumerate(images):
            print(f"  处理图像 {i+1}/{len(images)}...")
            
            # 提取特征
            if self.use_multiple_models:
                features = self.extract_multiple_models_features(image)
            else:
                features = self.extract_features(image)
            
            if features is not None:
                all_features.append(features)
                successful_count += 1
                print(f"    ✓ 图像 {i+1} 特征提取成功")
            else:
                print(f"    ✗ 图像 {i+1} 特征提取失败")
        
        if not all_features:
            print("✗ 所有图像特征提取失败")
            return False
        
        # 计算特征均值
        mean_features = np.mean(all_features, axis=0)
        print(f"✓ 成功提取 {successful_count}/{len(images)} 张图像的特征")
        print(f"  特征均值计算完成，维度: {mean_features.shape}")
        
        # 保存用户特征
        self.user_features[user_id] = {
            'name': name,
            'features': mean_features.tolist(),  # 转换为列表以便JSON序列化
            'register_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_count': successful_count,  # 样本数量
            'original_samples': successful_count  # 原始样本数量
        }
        
        # 保存到文件
        if self.save_user_features():
            print(f"✓ 用户 {user_id} ({name}) 使用多张图像注册成功!")
            print(f"  使用 {successful_count} 张图像的特征均值")
            return True
        else:
            print(f"✗ 用户 {user_id} 注册失败: 无法保存特征")
            return False
    
    def update_user_features(self, user_id, new_image, learning_rate=0.1):
        """
        增量学习：使用新图像更新用户特征
        
        Args:
            user_id: 用户ID
            new_image: 新图像（OpenCV BGR图像或图像路径）
            learning_rate: 学习率（0-1），控制新特征的影响程度
            
        Returns:
            bool: 是否更新成功
        """
        # 检查用户是否存在
        if user_id not in self.user_features:
            print(f"✗ 用户 {user_id} 不存在!")
            return False
        
        # 提取新特征
        if self.use_multiple_models:
            new_features = self.extract_multiple_models_features(new_image)
        else:
            new_features = self.extract_features(new_image)
        
        if new_features is None:
            print("✗ 无法提取新图像的特征")
            return False
        
        # 获取当前特征
        current_features = np.array(self.user_features[user_id]['features'])
        
        # 获取样本数量
        sample_count = self.user_features[user_id].get('sample_count', 1)
        
        # 增量更新：加权平均
        # 新特征权重 = learning_rate
        # 旧特征权重 = 1 - learning_rate
        updated_features = (1 - learning_rate) * current_features + learning_rate * new_features
        
        # 更新用户特征
        self.user_features[user_id]['features'] = updated_features.tolist()
        self.user_features[user_id]['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.user_features[user_id]['sample_count'] = sample_count + 1
        
        # 保存到文件
        if self.save_user_features():
            print(f"✓ 用户 {user_id} 特征更新成功!")
            print(f"  学习率: {learning_rate}, 样本数量: {sample_count + 1}")
            return True
        else:
            print(f"✗ 用户 {user_id} 特征更新失败")
            return False
    
    def adjust_threshold(self, distance, is_verified):
        """
        动态调整阈值
        
        Args:
            distance: 当前距离
            is_verified: 是否验证成功
            
        Returns:
            float: 调整后的阈值
        """
        if not self.enable_dynamic_threshold:
            return self.threshold
        
        # 根据距离动态调整阈值
        if is_verified:
            # 验证成功，但距离接近阈值，可以稍微降低阈值以提高安全性
            if distance > self.threshold * 0.8:
                adjustment = -0.02  # 稍微降低阈值
            else:
                adjustment = 0.0  # 保持阈值
        else:
            # 验证失败，如果距离接近阈值，可以稍微提高阈值以提高识别率
            if distance < self.threshold * 1.2:
                adjustment = 0.05  # 稍微提高阈值
            else:
                adjustment = 0.0  # 保持阈值
        
        # 应用调整，但保持在合理范围内
        self.threshold_adjustment += adjustment
        self.threshold_adjustment = max(-0.2, min(0.2, self.threshold_adjustment))  # 限制调整范围
        
        new_threshold = self.base_threshold + self.threshold_adjustment
        new_threshold = max(0.3, min(1.5, new_threshold))  # 确保阈值在合理范围内
        
        if adjustment != 0:
            print(f"  动态阈值调整: {adjustment:+.3f}, 新阈值: {new_threshold:.3f}")
        
        return new_threshold
    
    def verify_user(self, user_id, image):
        """
        验证用户身份
        
        Args:
            user_id: 要验证的用户ID
            image: OpenCV BGR图像或图像路径
            
        Returns:
            dict: 验证结果
        """
        # 检查用户是否存在
        if user_id not in self.user_features:
            print(f"✗ 用户 {user_id} 不存在!")
            return {
                'verified': False,
                'error': f'用户 {user_id} 不存在',
                'distance': None,
                'user_id': user_id
            }
        
        # 提取当前图像的特征
        current_features = self.extract_features(image)
        if current_features is None:
            return {
                'verified': False,
                'error': '无法提取人脸特征',
                'distance': None,
                'user_id': user_id
            }
        
        # 获取存储的特征
        stored_features = np.array(self.user_features[user_id]['features'])
        
        # 计算距离
        distance = self.compare_features(current_features, stored_features)
        
        # 判断是否验证成功
        # 使用双重阈值：
        # - accept_threshold: 接受阈值（宽松）
        # - reject_threshold: 拒绝阈值（严格）
        accept_threshold = self.threshold  # 当前阈值
        reject_threshold = 0.3  # 调整：更严格的拒绝阈值，防止用户间互相识别
        
        verified = distance < accept_threshold and distance < reject_threshold
        
        # 动态调整阈值
        if self.enable_dynamic_threshold:
            old_threshold = self.threshold
            self.threshold = self.adjust_threshold(distance, verified)
        
        result = {
            'verified': verified,
            'user_id': user_id,
            'user_name': self.user_features[user_id]['name'],
            'distance': distance,
            'threshold': self.threshold,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if verified:
            print(f"✓ 验证成功: {self.user_features[user_id]['name']} (距离: {distance:.4f})")
        else:
            print(f"✗ 验证失败: 距离 {distance:.4f} 大于阈值 {self.threshold}")
        
        return result
    
    def check_feature_dimension(self, user_id=None):
        """
        检查特征维度是否与当前模型匹配
        
        Args:
            user_id: 要检查的用户ID，如果为None则检查所有用户
            
        Returns:
            bool: 是否所有特征维度都匹配
        """
        expected_dimension = self.model_dimensions[self.primary_model]
        
        if user_id:
            users_to_check = [(user_id, self.user_features[user_id])] if user_id in self.user_features else []
        else:
            users_to_check = self.user_features.items()
        
        all_match = True
        for uid, user_info in users_to_check:
            actual_dimension = len(user_info['features'])
            if actual_dimension != expected_dimension:
                print(f"⚠ 警告: 用户 {uid} 的特征维度不匹配!")
                print(f"   当前模型 ({self.primary_model}): {expected_dimension}维")
                print(f"   存储特征: {actual_dimension}维")
                print(f"   可能原因: 特征是用不同模型提取的")
                all_match = False
        
        return all_match
    
    def check_and_warn_dimension_mismatch(self):
        """
        检查并警告特征维度不匹配问题
        """
        if not self.user_features:
            return
        
        expected_dimension = self.model_dimensions[self.primary_model]
        mismatched_users = []
        
        for uid, user_info in self.user_features.items():
            actual_dimension = len(user_info['features'])
            if actual_dimension != expected_dimension:
                mismatched_users.append({
                    'user_id': uid,
                    'name': user_info['name'],
                    'actual_dim': actual_dimension,
                    'expected_dim': expected_dimension,
                    'model_used': user_info.get('model_used', '未知')
                })
        
        if mismatched_users:
            print("\n" + "=" * 60)
            print("⚠ 特征维度不匹配警告")
            print("=" * 60)
            print(f"当前主要模型: {self.primary_model} ({expected_dimension}维)")
            print(f"发现 {len(mismatched_users)} 个用户的特征维度不匹配:")
            
            for user in mismatched_users:
                print(f"\n  用户: {user['user_id']} ({user['name']})")
                print(f"    存储特征维度: {user['actual_dim']}维")
                print(f"    预期维度: {user['expected_dim']}维")
                print(f"    记录使用的模型: {user['model_used']}")
                
                # 根据维度猜测可能的模型
                if user['actual_dim'] == 128:
                    possible_models = "Facenet 或 OpenFace"
                elif user['actual_dim'] == 512:
                    possible_models = "ArcFace"
                elif user['actual_dim'] == 4096:
                    possible_models = "VGG-Face"
                else:
                    possible_models = "未知模型"
                
                print(f"    可能使用的模型: {possible_models}")
            
            print("\n建议解决方案:")
            print("  1. 将主要模型设置为与用户特征匹配的模型")
            print("  2. 使用 reextract_user_features() 方法重新提取用户特征")
            print("  3. 使用 migrate_to_arcface.py 脚本迁移用户特征")
            print("=" * 60)
    
    def reextract_user_features(self, user_id, image_paths=None):
        """
        重新提取用户特征（使用当前主要模型）
        
        Args:
            user_id: 要重新提取特征的用户ID
            image_paths: 图像路径列表，如果为None则尝试查找用户图像
            
        Returns:
            bool: 是否重新提取成功
        """
        if user_id not in self.user_features:
            print(f"✗ 用户 {user_id} 不存在")
            return False
        
        user_info = self.user_features[user_id]
        print(f"\n重新提取用户 {user_id} ({user_info['name']}) 的特征...")
        print(f"  当前特征维度: {len(user_info['features'])}维")
        print(f"  目标模型: {self.primary_model} ({self.model_dimensions[self.primary_model]}维)")
        
        # 查找用户图像
        if image_paths is None:
            image_paths = self.find_user_images(user_id)
        
        if not image_paths:
            print("  ⚠ 未找到用户图像，无法重新提取特征")
            print("  请提供图像路径或确保用户图像目录存在")
            return False
        
        print(f"  找到 {len(image_paths)} 张用户图像")
        
        # 使用所有图像提取特征
        all_features = []
        successful_count = 0
        
        for i, img_path in enumerate(image_paths):
            print(f"    处理图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
            
            # 提取特征
            features = self.extract_features(img_path)
            if features is not None:
                all_features.append(features)
                successful_count += 1
                print(f"      ✓ 特征提取成功，维度: {features.shape}")
            else:
                print(f"      ✗ 特征提取失败")
        
        if not all_features:
            print("  ✗ 所有图像特征提取失败")
            return False
        
        # 计算特征均值
        mean_features = np.mean(all_features, axis=0)
        
        # 更新用户特征
        user_info['features'] = mean_features.tolist()
        user_info['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_info['model_used'] = self.primary_model
        user_info['reextracted'] = True
        user_info['reextraction_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_info['sample_count'] = successful_count
        
        print(f"  ✓ 特征重新提取成功，新特征维度: {mean_features.shape}")
        print(f"    使用 {successful_count} 张图像的特征均值")
        
        # 保存更新
        self.user_features[user_id] = user_info
        return self.save_user_features()
    
    def find_user_images(self, user_id):
        """
        查找用户的图像文件
        
        Args:
            user_id: 用户ID
            
        Returns:
            list: 图像文件路径列表
        """
        # 尝试在face_database目录下查找用户图像
        user_dir = os.path.join('face_database', user_id)
        if os.path.exists(user_dir):
            images = []
            for file in os.listdir(user_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    images.append(os.path.join(user_dir, file))
            return images
        
        # 尝试在当前目录下查找
        images = []
        for file in os.listdir('.'):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 检查文件名是否包含用户ID
                if user_id in file:
                    images.append(file)
        
        return images
    
    def identify_user(self, image):
        """
        识别图像中的用户（与所有用户比较）
        
        Args:
            image: OpenCV BGR图像或图像路径
            
        Returns:
            dict: 识别结果
        """
        if not self.user_features:
            print("✗ 没有注册用户，无法识别")
            return {
                'identified': False,
                'error': '没有注册用户',
                'best_match': None,
                'all_distances': []
            }
        
        # 检查特征维度是否匹配
        if not self.check_feature_dimension():
            print("⚠ 警告: 特征维度不匹配，识别结果可能不准确!")
            print("   建议: 使用相同模型重新注册用户或调整主要模型设置")
        
        # 提取当前图像的特征
        current_features = self.extract_features(image)
        if current_features is None:
            return {
                'identified': False,
                'error': '无法提取人脸特征',
                'best_match': None,
                'all_distances': []
            }
        
        # 与所有用户比较
        distances = []
        for user_id, user_info in self.user_features.items():
            stored_features = np.array(user_info['features'])
            distance = self.compare_features(current_features, stored_features)
            distances.append({
                'user_id': user_id,
                'name': user_info['name'],
                'distance': distance
            })
        
        # 找到最佳匹配
        distances.sort(key=lambda x: x['distance'])
        best_match = distances[0]
        
        # 判断是否识别成功
        identified = best_match['distance'] < self.threshold
        
        # 动态调整阈值
        if self.enable_dynamic_threshold:
            old_threshold = self.threshold
            self.threshold = self.adjust_threshold(best_match['distance'], identified)
        
        result = {
            'identified': identified,
            'best_match': best_match if identified else None,
            'all_distances': distances,
            'threshold': self.threshold,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if identified:
            print(f"✓ 识别成功: {best_match['name']} (距离: {best_match['distance']:.4f})")
        else:
            print(f"✗ 识别失败: 最佳匹配距离 {best_match['distance']:.4f} 大于阈值 {self.threshold}")
        
        return result
    
    def list_users(self):
        """
        列出所有注册用户
        """
        if not self.user_features:
            print("没有注册用户")
            return
        
        print("\n注册用户列表:")
        print("-" * 60)
        for user_id, user_info in self.user_features.items():
            print(f"ID: {user_id}")
            print(f"  姓名: {user_info['name']}")
            print(f"  注册时间: {user_info.get('register_time', '未知')}")
            print(f"  最后更新: {user_info.get('last_update', '未知')}")
            print(f"  特征维度: {len(user_info['features'])}")
            print(f"  使用模型: {user_info.get('model_used', '未知')}")
            print(f"  样本数量: {user_info.get('sample_count', 1)}")
            print("-" * 60)
    
    def fix_existing_features(self):
        """
        修复已有特征（归一化处理）
        """
        print("\n" + "=" * 60)
        print("修复已有特征（归一化处理）")
        print("=" * 60)
        
        fixed_count = 0
        for user_id, user_info in self.user_features.items():
            features = user_info['features']
            
            if isinstance(features, list):
                features_array = np.array(features)
                norm = np.linalg.norm(features_array)
                
                print(f"\n用户: {user_id} ({user_info['name']})")
                print(f"  原始范数: {norm:.4f}")
                
                if norm > 0:
                    # 归一化
                    normalized = features_array / norm
                    user_info['features'] = normalized.tolist()
                    user_info['normalized'] = True
                    user_info['original_norm'] = float(norm)
                    fixed_count += 1
                    print(f"  ✓ 已归一化 -> 范数: {np.linalg.norm(normalized):.4f}")
                else:
                    print(f"  ✗ 范数为零，无法归一化")
        
        if fixed_count > 0:
            self.save_user_features()
            print(f"\n✓ 修复完成: {fixed_count}个用户特征已归一化")
        else:
            print("\n没有需要修复的特征")
        
        return fixed_count
    
    def delete_user(self, user_id):
        """
        删除用户
        
        Args:
            user_id: 要删除的用户ID
            
        Returns:
            bool: 是否删除成功
        """
        if user_id not in self.user_features:
            print(f"✗ 用户 {user_id} 不存在!")
            return False
        
        user_name = self.user_features[user_id]['name']
        del self.user_features[user_id]
        
        if self.save_user_features():
            print(f"✓ 用户 {user_id} ({user_name}) 已删除")
            return True
        else:
            print(f"✗ 删除用户 {user_id} 失败")
            return False


class FaceCapture:
    """
    人脸捕获类，使用MTCNN进行人脸检测
    """
    def __init__(self, camera_index=0):
        """
        初始化人脸捕获
        
        Args:
            camera_index: 摄像头索引
        """
        self.camera_index = camera_index
        self.detector = MTCNN()
        self.cap = None
        
    def open_camera(self):
        """打开摄像头"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"✗ 无法打开摄像头 {self.camera_index}")
            return False
        
        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"✓ 摄像头 {self.camera_index} 已打开 (640x480)")
        return True
    
    def close_camera(self):
        """关闭摄像头"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            print("✓ 摄像头已关闭")
    
    def capture_face(self, show_preview=True):
        """
        捕获人脸图像
        
        Args:
            show_preview: 是否显示预览
            
        Returns:
            numpy.ndarray: 人脸图像，如果失败返回None
        """
        if not self.cap:
            if not self.open_camera():
                return None
        
        print("\n准备捕获人脸...")
        print("按 SPACE 键捕获，按 'q' 键退出")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("✗ 无法读取摄像头帧")
                    return None
                
                # 使用MTCNN检测人脸
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb_frame)
                
                display_frame = frame.copy()
                
                if len(faces) > 0:
                    # 检测到人脸，绘制框
                    x, y, w, h = faces[0]['box']
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Face Detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    face_status = "✓ 检测到人脸"
                else:
                    face_status = "✗ 未检测到人脸"
                
                # 添加状态信息
                cv2.putText(display_frame, face_status, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if show_preview:
                    cv2.imshow('Face Capture', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("用户取消捕获")
                    # 关闭摄像头窗口
                    if show_preview:
                        cv2.destroyWindow('Face Capture')
                    return None
                elif key == 32:  # SPACE键
                    if len(faces) > 0:
                        # 截取人脸区域
                        x, y, w, h = faces[0]['box']
                        face_image = frame[y:y+h, x:x+w]
                        
                        if face_image.size > 0:
                            print("✓ 人脸捕获成功")
                            # 关闭摄像头窗口
                            if show_preview:
                                cv2.destroyWindow('Face Capture')
                            return face_image
                        else:
                            print("✗ 人脸区域无效")
                    else:
                        print("✗ 未检测到人脸，请重新尝试")
        except KeyboardInterrupt:
            print("\n捕获被用户中断")
            # 关闭摄像头窗口
            if show_preview:
                cv2.destroyWindow('Face Capture')
            return None
        
        return None
    
    def capture_from_file(self, file_path):
        """
        从文件加载图像并检测人脸
        
        Args:
            file_path: 图像文件路径
            
        Returns:
            numpy.ndarray: 人脸图像，如果失败返回None
        """
        if not os.path.exists(file_path):
            print(f"✗ 文件不存在: {file_path}")
            return None
        
        frame = cv2.imread(file_path)
        if frame is None:
            print(f"✗ 无法读取图像文件: {file_path}")
            return None
        
        # 使用MTCNN检测人脸
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb_frame)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            face_image = frame[y:y+h, x:x+w]
            
            if face_image.size > 0:
                print(f"✓ 从文件检测到人脸: {file_path}")
                return face_image
            else:
                print(f"✗ 人脸区域无效: {file_path}")
                return None
        else:
            print(f"✗ 文件中未检测到人脸: {file_path}")
            return None


def main():
    """
    主函数 - 交互式界面
    """
    print("=" * 60)
    print("人脸识别系统 - JSON特征存储版 (增强版)")
    print("=" * 60)
    
    # 初始化系统
    features_file = input(f"请输入特征文件路径 (默认: user_features.json): ").strip()
    if not features_file:
        features_file = "user_features.json"
    
    # 询问是否启用多模型融合
    use_multiple_models_input = input("是否启用多模型融合? (y/n, 默认: n): ").strip().lower()
    use_multiple_models = use_multiple_models_input == 'y'
    
    recognizer = FaceRecognitionJSON(features_file, use_multiple_models=use_multiple_models)
    face_capture = FaceCapture()
    
    while True:
        print("\n" + "=" * 60)
        print("主菜单")
        print("=" * 60)
        print("1. 注册新用户 (单张图像)")
        print("2. 注册新用户 (多张图像)")
        print("3. 验证用户")
        print("4. 验证并增量学习 (验证成功后更新特征)")
        print("5. 识别用户")
        print("6. 列出所有用户")
        print("7. 删除用户")
        print("8. 从图像文件注册用户")
        print("9. 从图像文件验证用户")
        print("10. 设置相似度阈值 (当前: {})".format(recognizer.threshold))
        print("11. 手动更新用户特征 (增量学习)")
        print("12. 退出")
        print("=" * 60)
        
        choice = input("请选择操作 (1-12): ").strip()
        
        if choice == '1':
            # 注册新用户 (单张图像)
            user_id = input("请输入用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            name = input("请输入用户姓名 (可选): ").strip()
            if not name:
                name = user_id
            
            print("\n请面对摄像头准备捕获人脸...")
            face_image = face_capture.capture_face()
            
            if face_image is not None:
                recognizer.register_user(user_id, name, face_image)
        
        elif choice == '2':
            # 注册新用户 (多张图像)
            user_id = input("请输入用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            name = input("请输入用户姓名 (可选): ").strip()
            if not name:
                name = user_id
            
            num_images = input("请输入要捕获的图像数量 (建议3-5张): ").strip()
            try:
                num_images = int(num_images)
                if num_images < 1:
                    print("图像数量必须大于0")
                    continue
            except ValueError:
                print("请输入有效的数字")
                continue
            
            images = []
            print(f"\n准备捕获 {num_images} 张人脸图像...")
            for i in range(num_images):
                print(f"\n第 {i+1}/{num_images} 张图像:")
                face_image = face_capture.capture_face()
                if face_image is not None:
                    images.append(face_image)
                    print(f"✓ 第 {i+1} 张图像捕获成功")
                else:
                    print(f"✗ 第 {i+1} 张图像捕获失败")
            
            if images:
                recognizer.register_user_with_multiple_images(user_id, name, images)
            else:
                print("✗ 没有成功捕获任何图像")
        
        elif choice == '3':
            # 验证用户
            user_id = input("请输入要验证的用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            print("\n请面对摄像头准备验证...")
            face_image = face_capture.capture_face()
            
            if face_image is not None:
                result = recognizer.verify_user(user_id, face_image)
                if result['verified']:
                    print(f"✓ 验证成功! 欢迎 {result['user_name']}")
                else:
                    print(f"✗ 验证失败: {result.get('error', '未知错误')}")
        
        elif choice == '4':
            # 验证并增量学习
            user_id = input("请输入要验证的用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            print("\n请面对摄像头准备验证...")
            face_image = face_capture.capture_face()
            
            if face_image is not None:
                result = recognizer.verify_user(user_id, face_image)
                if result['verified']:
                    print(f"✓ 验证成功! 欢迎 {result['user_name']}")
                    
                    # 询问是否进行增量学习
                    update_choice = input("是否使用此图像更新用户特征? (y/n): ").strip().lower()
                    if update_choice == 'y':
                        try:
                            learning_rate = float(input("请输入学习率 (0.01-0.5, 默认0.1): ").strip() or "0.1")
                            if 0 < learning_rate <= 1:
                                recognizer.update_user_features(user_id, face_image, learning_rate)
                            else:
                                print("✗ 学习率必须在0到1之间")
                        except ValueError:
                            print("✗ 请输入有效的数字")
                else:
                    print(f"✗ 验证失败: {result.get('error', '未知错误')}")
        
        elif choice == '5':
            # 识别用户
            print("\n请面对摄像头准备识别...")
            face_image = face_capture.capture_face()
            
            if face_image is not None:
                result = recognizer.identify_user(face_image)
                if result['identified']:
                    print(f"✓ 识别成功! 用户: {result['best_match']['name']}")
                    print(f"   距离: {result['best_match']['distance']:.4f}")
                else:
                    print(f"✗ 识别失败: {result.get('error', '未知错误')}")
        
        elif choice == '6':
            # 列出所有用户
            recognizer.list_users()
        
        elif choice == '7':
            # 删除用户
            user_id = input("请输入要删除的用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            confirm = input(f"确定要删除用户 {user_id} 吗? (y/n): ").strip().lower()
            if confirm == 'y':
                recognizer.delete_user(user_id)
            else:
                print("取消删除操作")
        
        elif choice == '8':
            # 从图像文件注册用户
            user_id = input("请输入用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            name = input("请输入用户姓名 (可选): ").strip()
            if not name:
                name = user_id
            
            file_path = input("请输入图像文件路径: ").strip()
            if not file_path:
                print("文件路径不能为空!")
                continue
            
            face_image = face_capture.capture_from_file(file_path)
            if face_image is not None:
                recognizer.register_user(user_id, name, face_image)
        
        elif choice == '9':
            # 从图像文件验证用户
            user_id = input("请输入要验证的用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            file_path = input("请输入图像文件路径: ").strip()
            if not file_path:
                print("文件路径不能为空!")
                continue
            
            face_image = face_capture.capture_from_file(file_path)
            if face_image is not None:
                result = recognizer.verify_user(user_id, face_image)
                if result['verified']:
                    print(f"✓ 验证成功! 欢迎 {result['user_name']}")
                else:
                    print(f"✗ 验证失败: {result.get('error', '未知错误')}")
        
        elif choice == '10':
            # 设置相似度阈值
            try:
                new_threshold = float(input(f"请输入新的相似度阈值 (当前: {recognizer.threshold}): ").strip())
                if 0 < new_threshold < 2:  # 合理的阈值范围
                    recognizer.threshold = new_threshold
                    print(f"✓ 相似度阈值已更新为: {new_threshold}")
                else:
                    print("✗ 阈值必须在0到2之间")
            except ValueError:
                print("✗ 请输入有效的数字")
        
        elif choice == '11':
            # 手动更新用户特征 (增量学习)
            user_id = input("请输入要更新特征的用户ID: ").strip()
            if not user_id:
                print("用户ID不能为空!")
                continue
            
            if user_id not in recognizer.user_features:
                print(f"✗ 用户 {user_id} 不存在!")
                continue
            
            print("\n请面对摄像头准备捕获新图像...")
            face_image = face_capture.capture_face()
            
            if face_image is not None:
                try:
                    learning_rate = float(input("请输入学习率 (0.01-0.5, 默认0.1): ").strip() or "0.1")
                    if 0 < learning_rate <= 1:
                        recognizer.update_user_features(user_id, face_image, learning_rate)
                    else:
                        print("✗ 学习率必须在0到1之间")
                except ValueError:
                    print("✗ 请输入有效的数字")
        
        elif choice == '12':
            # 退出
            print("感谢使用人脸识别系统!")
            face_capture.close_camera()
            break
        
        else:
            print("无效的选择，请重新输入!")
        
        # 每次操作后暂停一下
        input("\n按Enter键继续...")
    
    print("程序结束")
    
if __name__ == "__main__":
    main()
