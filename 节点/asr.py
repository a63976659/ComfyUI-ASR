import torch
import os
import tempfile
import torchaudio
import json
import numpy as np
import random
import gc
# 引入 ComfyUI 的工具以支持进度条
from comfy.utils import ProgressBar

# 从 utils_asr 导入
from .utils_asr import load_asr_model, unload_asr_model

# ================= 配置与常量 =================

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

# ================= 辅助函数 =================

def save_audio_to_temp(audio_input):
    """
    将 ComfyUI 的音频 Tensor 保存为临时 WAV 文件。
    """
    try:
        waveform = audio_input['waveform'] 
        sample_rate = audio_input['sample_rate']
        
        # 处理维度
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
    except Exception as e:
        print(f"[ASR Error] Failed to save temp audio: {e}")
        return None

# ================= 节点 1: 标准 ASR 节点 (已精简输出) =================

class Qwen_ASR_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"]
        return {
            "required": {
                "音频": ("AUDIO", ),
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(ASR_LANGUAGES.keys()), {"default": "自动识别 (Auto)"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "可选：输入上下文或提示词"}),
                "生成时间戳": ("BOOLEAN", {"default": False, "label": "生成时间戳 (仅内部计算, 暂不输出)"}),
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
            }
        }

    # 修改：只输出一个 STRING
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本输出",)
    FUNCTION = "transcribe_audio"
    CATEGORY = "💬 AI人工智能/音频"
    DESCRIPTION = "使用 Qwen3-ASR 进行语音识别。"

    def transcribe_audio(self, 音频, 模型名称, 语言, 提示词, 生成时间戳, 下载源, 自动下载模型):
        
        temp_wav_path = None
        try:
            # 1. 准备音频
            temp_wav_path = save_audio_to_temp(音频)
            if not temp_wav_path:
                raise ValueError("Audio processing failed.")

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
            context_prompt = 提示词.strip() if 提示词.strip() else None

            print(f"[Qwen ASR] Transcribing... Lang: {target_lang if target_lang else 'Auto'}")
            
            # 4. 执行推理
            kwargs = {
                "audio": [temp_wav_path],
                "language": [target_lang] if target_lang else None,
                "return_time_stamps": 生成时间戳
            }
            if context_prompt:
                kwargs["context"] = [context_prompt]

            results = model.transcribe(**kwargs)

            # 5. 处理结果
            result = results[0]
            text_output = result.text
            
            print(f"[Qwen ASR] Detected: {result.language}")
            print(f"[Qwen ASR] Text: {text_output[:50]}...")

            # 只返回文本
            return (text_output,)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"ASR Error: {str(e)}")
            
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

# ================= 节点 2: 批量 ASR 节点 (已精简输出) =================

class Qwen_ASR_Batch_Node:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def INPUT_TYPES(cls):
        presets = ["Qwen3-ASR-1.7B", "Qwen3-ASR-0.6B"]
        return {
            "required": {
                "音频列表": ("AUDIO", ), 
                "模型名称": (presets, {"default": presets[0]}),
                "语言": (list(ASR_LANGUAGES.keys()), {"default": "自动识别 (Auto)"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "placeholder": "批量提示词"}),
                "生成时间戳": ("BOOLEAN", {"default": False}),
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
            }
        }

    # 修改：只输出一个 STRING
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("合并文本",)
    FUNCTION = "batch_transcribe"
    CATEGORY = "💬 AI人工智能/音频"
    DESCRIPTION = "批量处理多个音频片段，输出合并后的文本。"

    def batch_transcribe(self, 音频列表, 模型名称, 语言, 提示词, 生成时间戳, 下载源, 自动下载模型):
        temp_files = []
        try:
            audio_inputs = []
            if isinstance(音频列表, list):
                audio_inputs = 音频列表
            else:
                audio_inputs = [音频列表]

            total_files = len(audio_inputs)
            if total_files == 0:
                return ("",)

            # 进度条
            pbar = ProgressBar(total_files)

            model = load_asr_model(
                模型名称, 
                self.device, 
                自动下载模型, 
                source=下载源, 
                use_aligner=生成时间戳
            )
            
            target_lang = ASR_LANGUAGES.get(语言, None)
            context_prompt = 提示词.strip() if 提示词.strip() else None
            
            full_text_list = []

            print(f"[Qwen ASR Batch] Processing {total_files} files...")

            for i, audio_item in enumerate(audio_inputs):
                temp_path = None
                try:
                    temp_path = save_audio_to_temp(audio_item)
                    if not temp_path:
                        continue
                    
                    kwargs = {
                        "audio": [temp_path],
                        "language": [target_lang] if target_lang else None,
                        "return_time_stamps": 生成时间戳
                    }
                    if context_prompt:
                        kwargs["context"] = [context_prompt]

                    results = model.transcribe(**kwargs)
                    res = results[0]

                    # 收集结果
                    full_text_list.append(res.text)

                except Exception as inner_e:
                    print(f"[Error] Batch processing failed at index {i}: {inner_e}")
                    full_text_list.append(f"[Error in file {i+1}]")
                
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                    pbar.update(1)

            # 只返回合并后的文本
            return ("\n".join(full_text_list),)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Batch ASR Error: {str(e)}")
        finally:
            unload_asr_model()
