import torch
import os
import tempfile
import torchaudio
import json
import numpy as np
import random
# 修改导入路径，从 utils_asr 导入
from .utils_asr import load_asr_model, unload_asr_model

# ASR 语言列表
ASR_LANGUAGES = {
    "自动识别 (Auto)": None,
    "中文 (Chinese)": "Chinese",
    "英语 (English)": "English",
    "粤语 (Cantonese)": "Cantonese",
    "日语 (Japanese)": "Japanese",
    "韩语 (Korean)": "Korean",
    "法语 (French)": "French",
    "德语 (German)": "German",
    "西班牙语 (Spanish)": "Spanish",
    "俄语 (Russian)": "Russian",
    "意大利语 (Italian)": "Italian",
    "葡萄牙语 (Portuguese)": "Portuguese",
    "泰语 (Thai)": "Thai",
    "越南语 (Vietnamese)": "Vietnamese",
    "阿拉伯语 (Arabic)": "Arabic",
    "印尼语 (Indonesian)": "Indonesian",
    "土耳其语 (Turkish)": "Turkish",
    "印地语 (Hindi)": "Hindi",
    "马来语 (Malay)": "Malay",
    "荷兰语 (Dutch)": "Dutch",
    "瑞典语 (Swedish)": "Swedish",
    "丹麦语 (Danish)": "Danish",
    "芬兰语 (Finnish)": "Finnish",
    "波兰语 (Polish)": "Polish",
    "捷克语 (Czech)": "Czech",
    "菲律宾语 (Filipino)": "Filipino",
    "波斯语 (Persian)": "Persian",
    "希腊语 (Greek)": "Greek",
    "匈牙利语 (Hungarian)": "Hungarian",
    "马其顿语 (Macedonian)": "Macedonian",
    "罗马尼亚语 (Romanian)": "Romanian"
}

class Qwen_ASR_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"]
        return {
            "required": {
                # 全中文组件名称
                "音频": ("AUDIO", ),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(ASR_LANGUAGES.keys()), {"default": "自动识别 (Auto)"}),
                
                # --- 推理参数 ---
                "最大生成长度": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
                "批处理大小": ("INT", {"default": 1, "min": 1, "max": 32}),
                "生成时间戳": ("BOOLEAN", {"default": False, "label": "生成时间戳 (需下载额外模型)"}),
                
                # --- 下载设置 ---
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("文本输出", "JSON详细数据")
    FUNCTION = "transcribe_audio"
    CATEGORY = "💬 AI人工智能"
    DESCRIPTION = "使用 Qwen3-ASR 进行多语言语音识别。开启时间戳将自动下载 Qwen3-ForcedAligner。"

    def _save_temp_wav(self, audio_input):
        """将 ComfyUI 的音频 Tensor 保存为临时 WAV 文件"""
        waveform = audio_input['waveform'] 
        sample_rate = audio_input['sample_rate']
        
        if waveform.dim() == 3:
            wav_tensor = waveform[0]
        else:
            wav_tensor = waveform

        if wav_tensor.shape[0] > wav_tensor.shape[1]: 
             wav_tensor = wav_tensor.t()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        
        torchaudio.save(temp_file.name, wav_tensor.cpu(), sample_rate)
        return temp_file.name

    def transcribe_audio(self, 音频, 模型名称, 语言, 最大生成长度, 批处理大小, 生成时间戳, 下载源, 自动下载模型):
        
        temp_wav_path = None
        try:
            # 1. 准备音频文件
            temp_wav_path = self._save_temp_wav(音频)
            
            # 2. 加载模型
            model = load_asr_model(
                模型名称, 
                self.device, 
                自动下载模型, 
                source=下载源, 
                use_aligner=生成时间戳
            )

            # 3. 准备参数
            target_lang = ASR_LANGUAGES.get(语言, None)
            
            print(f"[Qwen ASR] Transcribing... Lang: {target_lang if target_lang else 'Auto'} | Timestamps: {生成时间戳}")
            
            # 4. 执行推理
            results = model.transcribe(
                audio=[temp_wav_path],
                language=[target_lang] if target_lang else None,
                return_time_stamps=生成时间戳,
                max_new_tokens=最大生成长度,
                batch_size=批处理大小
            )

            # 5. 处理结果
            result = results[0]
            text_output = result.text
            
            # 构建详细 JSON 输出
            json_data = {
                "language": result.language,
                "text": result.text,
            }
            if 生成时间戳 and hasattr(result, 'time_stamps'):
                json_data["timestamps"] = result.time_stamps

            print(f"[Qwen ASR] Detected: {result.language}")
            print(f"[Qwen ASR] Text: {text_output[:50]}...")

            return (text_output, json.dumps(json_data, ensure_ascii=False, indent=2))

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"ASR Error: {str(e)}")
            
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            
            unload_asr_model()