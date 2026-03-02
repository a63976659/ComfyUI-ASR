import torch
import os
import tempfile
import torchaudio
import json
import numpy as np
import random
import gc
# 引入 ComfyUI 的工具以支持进度条
import nodes
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

def format_time(seconds):
    """将秒数格式化为 分:秒.毫秒 (MM:SS.xx)"""
    if seconds is None:
        return "00:00.00"
    seconds = float(seconds)
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

def merge_timestamps_by_text(full_text, time_stamps):
    """
    将字级时间戳严格按照 [文本输出] 中的标点符号进行对齐和分段。
    包含：严格获取原生第一个字的 start_time 作为起始，最后一个字的 end_time 作为结束。
    """
    merged_list = []
    if not time_stamps or not full_text:
        return merged_list

    # 定义所有作为断句依据的标点符号
    punctuations = set('。！？.!?，,；;、\n')
    
    text_ptr = 0
    text_len = len(full_text)

    current_start = None
    current_end = None
    current_segment_text = ""

    for ts in time_stamps:
        ts_text = getattr(ts, 'text', '').strip()
        # 完全提取原生时间戳，不做任何大小判断
        start = getattr(ts, 'start_time', None)
        end = getattr(ts, 'end_time', None)

        if not ts_text:
            continue

        # 记录该段落第一个能拿到原声时间戳的起始点
        if current_start is None and start is not None:
            current_start = start
        
        # 持续覆盖为当前字的结束点
        if end is not None:
            current_end = end

        # 在带有标点的完整输出文本中，查找当前 token 的位置
        found_idx = full_text.find(ts_text, text_ptr)
        if found_idx == -1:
            found_idx = full_text.lower().find(ts_text.lower(), text_ptr)
        
        if found_idx != -1:
            # 拼接从当前指针到这个词结束的所有字符
            chunk = full_text[text_ptr : found_idx + len(ts_text)]
            current_segment_text += chunk
            text_ptr = found_idx + len(ts_text)
        else:
            current_segment_text += ts_text

        lookahead_ptr = text_ptr
        has_punctuation = False

        while lookahead_ptr < text_len:
            c = full_text[lookahead_ptr]
            if c in punctuations:
                has_punctuation = True
                current_segment_text += c
                lookahead_ptr += 1
            elif c == ' ':
                current_segment_text += c
                lookahead_ptr += 1
            else:
                break 

        text_ptr = lookahead_ptr

        # 探测到标点，封口当前这一句
        if has_punctuation:
            merged_list.append({
                "start": current_start if current_start is not None else 0.0,
                "end": current_end if current_end is not None else 0.0,
                "text": current_segment_text.strip()
            })
            # 重置状态
            current_start = None
            current_end = None
            current_segment_text = ""

    # 兜底：处理最后一段文本
    if current_segment_text.strip() or current_start is not None:
         if text_ptr < text_len:
             current_segment_text += full_text[text_ptr:]

         merged_list.append({
             "start": current_start if current_start is not None else 0.0,
             "end": current_end if current_end is not None else 0.0,
             "text": current_segment_text.strip()
         })

    return merged_list

def format_timestamps(merged_stamps):
    """将合并后的时间戳字典列表格式化，并自动换行"""
    if not merged_stamps:
        return ""
    
    lines = []
    for ts in merged_stamps:
        start = ts.get('start', 0.0)
        end = ts.get('end', 0.0)
        text = ts.get('text', '')
        # 转换为分钟格式，例如 00:00.00 - 01:05.36
        lines.append(f"{format_time(start)} - {format_time(end)}: {text}")
        
    return "\n".join(lines)


# ================= 节点 1: 标准 ASR 节点 =================

class Qwen语音识别_Node:
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
                "生成时间戳": ("BOOLEAN", {"default": False, "label": "生成时间戳 (需下载额外模型)"}),
                "下载源": (["ModelScope", "HuggingFace", "HF Mirror"], {"default": "ModelScope"}),
                "自动下载模型": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("文本输出", "JSON详细数据", "带时间戳文本")
    FUNCTION = "transcribe_audio"
    CATEGORY = "💬 AI人工智能/语音识别"
    DESCRIPTION = "使用 Qwen3-ASR 进行语音识别。"

    def transcribe_audio(self, 音频, 模型名称, 语言, 提示词, 生成时间戳, 下载源, 自动下载模型):
        
        temp_wav_path = None
        try:
            temp_wav_path = save_audio_to_temp(音频)
            if not temp_wav_path:
                raise ValueError("Audio processing failed.")

            model = load_asr_model(
                模型名称, 
                self.device, 
                自动下载模型, 
                source=下载源, 
                use_aligner=生成时间戳
            )

            target_lang = ASR_LANGUAGES.get(语言, None)
            context_prompt = 提示词.strip() if 提示词.strip() else None

            print(f"[Qwen ASR] Transcribing... Lang: {target_lang if target_lang else 'Auto'}")
            
            kwargs = {
                "audio": [temp_wav_path],
                "language": [target_lang] if target_lang else None,
                "return_time_stamps": 生成时间戳
            }
            if context_prompt:
                kwargs["context"] = [context_prompt]

            results = model.transcribe(**kwargs)

            result = results[0]
            text_output = result.text
            
            json_data = {
                "language": result.language,
                "text": text_output,
                "timestamps": []
            }
            
            formatted_ts = ""
            if 生成时间戳:
                if hasattr(result, 'time_stamps') and result.time_stamps:
                    merged_ts = merge_timestamps_by_text(text_output, result.time_stamps)
                    json_data["timestamps"] = merged_ts
                    formatted_ts = format_timestamps(merged_ts)

            print(f"[Qwen ASR] Detected: {result.language}")
            print(f"[Qwen ASR] Text: {text_output[:50]}...")

            return (text_output, json.dumps(json_data, ensure_ascii=False, indent=2), formatted_ts)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"ASR Error: {str(e)}")
            
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

# ================= 节点 2: 批量 ASR 节点 =================

class Qwen批量语音识别_Node:
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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("合并文本", "详细日志文本", "带时间戳文本")
    FUNCTION = "batch_transcribe"
    CATEGORY = "💬 AI人工智能/语音识别"
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
                return ("", "", "")

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
            log_lines = []
            timestamp_text_list = []

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

                    full_text_list.append(res.text)
                    
                    current_ts_str = ""
                    if 生成时间戳 and hasattr(res, 'time_stamps'):
                        merged_ts = merge_timestamps_by_text(res.text, res.time_stamps)
                        current_ts_str = format_timestamps(merged_ts)
                    
                    filename = audio_item.get("filename", f"Audio_{i+1}")

                    if current_ts_str:
                        timestamp_text_list.append(f"--- {filename} ---")
                        timestamp_text_list.append(current_ts_str)
                        timestamp_text_list.append("")
                    
                    log_lines.append(f"--- [{i+1}/{total_files}] {filename} ({res.language}) ---")
                    log_lines.append(res.text)
                    if current_ts_str:
                        log_lines.append("[Timestamps]")
                        log_lines.append(current_ts_str)
                    log_lines.append("")

                except Exception as inner_e:
                    print(f"[Error] Batch processing failed at index {i}: {inner_e}")
                    full_text_list.append(f"[Error in file {i+1}]")
                
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                    pbar.update(1)

            return ("\n".join(full_text_list), "\n".join(log_lines), "\n".join(timestamp_text_list))

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Batch ASR Error: {str(e)}")
        finally:
            unload_asr_model()