from .节点.translator import LLM_Translator_Node
from .节点.chat import LLM_Chat_Node
from .节点.tts import Qwen_TTS_Node, Qwen_TTS_VoiceDesign_Node, Qwen_TTS_VoiceClone_Node
from .节点.cosyvoice import Fun_CosyVoice3_Node 

NODE_CLASS_MAPPINGS = {
    "LLM_Translator": LLM_Translator_Node,
    "LLM_Chat": LLM_Chat_Node,
    "Qwen_TTS": Qwen_TTS_Node,
    "Qwen_TTS_VoiceDesign": Qwen_TTS_VoiceDesign_Node,
    "Qwen_TTS_VoiceClone": Qwen_TTS_VoiceClone_Node,
    "Fun_CosyVoice3": Fun_CosyVoice3_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Translator": "🧠 LLM 智能翻译 (Qwen)",
    "LLM_Chat": "💬 LLM 智能对话 (Qwen)",
    "Qwen_TTS": "🔊 Qwen 语音合成 (CustomVoice)",
    "Qwen_TTS_VoiceDesign": "🔊 Qwen 语音设计 (VoiceDesign)",
    "Qwen_TTS_VoiceClone": "🔊 Qwen 语音克隆 (VoiceClone)",
    "Fun_CosyVoice3": "🎤 CosyVoice 3.0 语音合成"  # <--- 已修改为中文
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]