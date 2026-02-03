
from .节点.asr import Qwen_ASR_Node  

NODE_CLASS_MAPPINGS = {
    # ASR 类
    "Qwen_ASR": Qwen_ASR_Node  
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen_ASR": "🎤 Qwen 语音识别 (ASR)"  
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]