import server
from aiohttp import web
import tkinter as tk
from tkinter import filedialog
import os

# 导入节点 (假设你的目录结构是 节点/asr.py 和 节点/加载音频.py)
# 如果文件在根目录，请改为 from .asr import ...
from .节点.asr import Qwen_ASR_Node, Qwen_ASR_Batch_Node
from .节点.加载音频 import Load_Audio_Folder_Node, Load_Audio_Node

# ================= API 1: 浏览文件夹 =================
@server.PromptServer.instance.routes.post("/qwen/browse_folder")
async def browse_folder(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        folder_path = filedialog.askdirectory()
        root.destroy()
        if folder_path:
            return web.json_response({"path": folder_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})

# ================= API 2: 浏览文件 (新增) =================
@server.PromptServer.instance.routes.post("/qwen/browse_file")
async def browse_file(request):
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        # 弹出文件选择框，限制音频格式
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path:
            return web.json_response({"path": file_path.replace("\\", "/")})
        return web.json_response({"path": ""})
    except Exception as e:
        return web.json_response({"error": str(e)})

# ================= 节点映射 =================

NODE_CLASS_MAPPINGS = {
    "Qwen_ASR": Qwen_ASR_Node,
    "Qwen_ASR_Batch": Qwen_ASR_Batch_Node,
    "Load_Audio_Folder": Load_Audio_Folder_Node,
    "Load_Audio": Load_Audio_Node  # <--- 新增
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen_ASR": "🎤 Qwen 语音识别 (ASR)",
    "Qwen_ASR_Batch": "🎤 Qwen 批量语音识别 (Batch)",
    "Load_Audio_Folder": "📂 批量加载音频文件夹",
    "Load_Audio": "🎵 加载音频 (Load Audio)" # <--- 新增
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]