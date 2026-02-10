from .节点.asr import Qwen语音识别_Node

NODE_CLASS_MAPPINGS = {
    "Qwen语音识别": Qwen语音识别_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen语音识别": "🎤 Qwen 语音识别 (ASR)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]