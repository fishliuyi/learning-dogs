"""
OpenVINO 推理工具类
用于将 PyTorch 模型转换为 OpenVINO IR 格式并使用 Intel GPU/NPU 加速
"""

import openvino as ov
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import onnx


class OpenVINOInferencer:
    """OpenVINO 推理器 - 支持 Intel GPU/NPU 加速"""
    
    def __init__(self, 
                 model_path: str,
                 device_name: str = "GPU",
                 input_shape: tuple = (1, 3, 224, 224)):
        """
        初始化 OpenVINO 推理器
        
        Args:
            model_path: PyTorch 模型路径 (.pth)
            device_name: 推理设备 ("GPU", "NPU", "CPU")
            input_shape: 输入张量形状 (B, C, H, W)
        """
        self.device_name = device_name
        self.input_shape = input_shape
        self.compiled_model = None
        self.infer_request = None
        
        # 加载模型
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """加载并转换模型为 OpenVINO IR 格式"""
        try:
            from reference.model import PetNet50
            
            # 1. 加载 PyTorch 模型
            print(f"📦 加载 PyTorch 模型：{model_path}")
            pet_model = PetNet50(model_path)
            pet_model.model.eval()
            
            # 2. 导出为 ONNX
            print("🔄 转换为 ONNX 格式...")
            dummy_input = torch.randn(self.input_shape)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_onnx:
                onnx_path = tmp_onnx.name
                
                torch.onnx.export(
                    pet_model.model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=None
                )
                print(f"✅ ONNX 导出成功：{onnx_path}")
            
            # 3. 使用 OpenVINO 读取 ONNX
            print(f"🚀 编译模型到 {self.device_name}...")
            core = ov.Core()
            onnx_model = core.read_model(onnx_path)
            
            # 4. 编译到指定设备
            self.compiled_model = core.compile_model(
                model=onnx_model,
                device_name=self.device_name
            )
            
            # 5. 创建推理请求
            self.infer_request = self.compiled_model.create_infer_request()
            
            print(f"✅ OpenVINO 模型加载完成，使用设备：{self.device_name}")
            
            # 清理临时文件
            import os
            try:
                os.remove(onnx_path)
            except:
                pass
                
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            raise
    
    def infer(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        执行推理
        
        Args:
            input_tensor: 输入张量 (B, C, H, W)
            
        Returns:
            特征向量 numpy 数组
        """
        if self.infer_request is None:
            raise RuntimeError("模型未初始化")
        
        # 转换为 numpy
        input_numpy = input_tensor.detach().cpu().numpy()
        
        # 执行推理
        self.infer_request.set_input_tensor(ov.Tensor(array=input_numpy))
        self.infer_request.start_async()
        self.infer_request.wait()
        
        # 获取结果
        result = self.infer_request.get_output_tensor().data
        
        return result
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        try:
            core = ov.Core()
            device_info = {
                'device_name': self.device_name,
                'full_name': core.get_property(self.device_name, 'FULL_DEVICE_NAME'),
                'device_type': core.get_property(self.device_name, 'DEVICE_TYPE')
            }
            return device_info
        except:
            return {'device_name': self.device_name}


def create_openvino_inferencer(model_path: str, prefer_gpu: bool = True):
    """
    创建 OpenVINO 推理器的便捷函数
    
    Args:
        model_path: PyTorch 模型路径
        prefer_gpu: 是否优先使用GPU
        
    Returns:
        OpenVINOInferencer 实例
    """
    import openvino as ov
    
    # 检测可用设备
    try:
        core = ov.Core()
        devices = core.available_devices
        
        if prefer_gpu and 'GPU' in devices:
            device = "GPU"
            print(f"🎯 使用 Intel GPU 加速")
        elif 'NPU' in devices:
            device = "NPU"
            print(f"🎯 使用 Intel NPU 加速")
        else:
            device = "CPU"
            print(f"⚠️ 使用 CPU (GPU/NPU 不可用)")
            
    except Exception as e:
        print(f"⚠️ OpenVINO 检测失败：{e}")
        device = "CPU"
    
    # 创建推理器
    inferencer = OpenVINOInferencer(
        model_path=model_path,
        device_name=device
    )
    
    return inferencer
