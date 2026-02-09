import os
import torchaudio

# ================= 节点 1: 文件夹音频加载器 =================

class Load_Audio_Folder_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # [汉化] Key 名称改为中文，必须与 JS 中的 widgetName 对应
                "文件夹路径": ("STRING", {"default": "./input/audio", "multiline": False, "label": "文件夹路径"}),
            },
            "optional": {
                "文件扩展名": ("STRING", {"default": "wav,mp3,flac,m4a,ogg", "multiline": False, "label": "文件扩展名"}),
                "递归搜索": ("BOOLEAN", {"default": False, "label": "递归搜索子文件夹"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("音频列表", "文件数量")
    FUNCTION = "load_batch_audio"
    CATEGORY = "💬 AI人工智能/IO"
    DESCRIPTION = "从指定文件夹批量加载音频文件。"

    # [汉化] 函数参数名必须与 INPUT_TYPES 中的 Key 保持一致
    def load_batch_audio(self, 文件夹路径, 文件扩展名, 递归搜索):
        path = 文件夹路径.strip()
        if not os.path.isabs(path): path = os.path.abspath(path)
        if not os.path.isdir(path): return ([], 0)
        
        extensions = tuple([f".{ext.strip().lower()}" for ext in 文件扩展名.split(",")])
        audio_files = []
        
        if 递归搜索:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(extensions):
                        audio_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path) and file.lower().endswith(extensions):
                    audio_files.append(file_path)

        audio_files.sort()
        if not audio_files: return ([], 0)

        batch_audio_data = []
        for file_path in audio_files:
            try:
                waveform, sample_rate = torchaudio.load(file_path)
                audio_item = {
                    "waveform": waveform,
                    "sample_rate": sample_rate,
                    "filename": os.path.basename(file_path),
                    "path": file_path
                }
                batch_audio_data.append(audio_item)
            except Exception as e:
                print(f"[Error] Failed to load {file_path}: {e}")

        return (batch_audio_data, len(batch_audio_data))

# ================= 节点 2: 单个音频加载器 =================

class Load_Audio_Node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # [汉化] Key 改为中文
                "文件路径": ("STRING", {"default": "./input/audio/example.wav", "multiline": False, "label": "文件路径"}),
            },
            "optional": {
                "开始时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "label": "开始时间(秒)"}),
                "持续时间": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.1, "label": "持续时间(0=全长)"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    FUNCTION = "load_audio"
    CATEGORY = "💬 AI人工智能/IO"
    DESCRIPTION = "加载单个音频文件，支持指定开始时间和持续时间。"

    # [汉化] 参数名对应修改
    def load_audio(self, 文件路径, 开始时间, 持续时间):
        path = 文件路径.strip()
        if not os.path.isabs(path):
            path = os.path.abspath(path)
            
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            info = torchaudio.info(path)
            sr = info.sample_rate
            total_frames = info.num_frames
            
            frame_offset = int(开始时间 * sr)
            num_frames = int(持续时间 * sr) if 持续时间 > 0 else -1
            
            if frame_offset >= total_frames:
                frame_offset = 0
            
            waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
            
            audio_data = {
                "waveform": waveform.unsqueeze(0) if waveform.dim() == 2 else waveform, 
                "sample_rate": sample_rate,
                "filename": os.path.basename(path),
                "path": path
            }
            
            return (audio_data,)

        except Exception as e:
            raise Exception(f"Failed to load audio: {str(e)}")