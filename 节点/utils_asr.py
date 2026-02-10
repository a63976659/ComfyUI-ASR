import os
import torch
import gc
import folder_paths
from huggingface_hub import snapshot_download as hf_snapshot_download

# --- 尝试导入 ModelScope ---
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

# --- 尝试导入 Qwen-ASR ---
try:
    from qwen_asr import Qwen3ASRModel
    HAS_QWEN_ASR = True
except ImportError:
    HAS_QWEN_ASR = False

# ================= 路径配置 =================

# 路径固定为 models/TTS
ASR_MODELS_DIR = os.path.join(folder_paths.models_dir, "TTS")
if not os.path.exists(ASR_MODELS_DIR):
    os.makedirs(ASR_MODELS_DIR)

# 全局缓存
LOADED_ASR_MODELS = {}

def _download_model_logic(repo_id, local_dir, source="ModelScope"):
    """ASR 专用下载逻辑"""
    if source == "ModelScope":
        if not HAS_MODELSCOPE:
            raise ImportError("请先安装 modelscope: pip install modelscope")
        print(f"\n[ASR Download] ModelScope: {repo_id} -> {local_dir}")
        ms_snapshot_download(model_id=repo_id, local_dir=local_dir)
    elif source == "HF Mirror":
        print(f"\n[ASR Download] HF Mirror: {repo_id} -> {local_dir}")
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        hf_snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True, max_workers=4)
    else: # HuggingFace
        print(f"\n[ASR Download] HuggingFace: {repo_id} -> {local_dir}")
        if "HF_ENDPOINT" in os.environ: del os.environ["HF_ENDPOINT"]
        hf_snapshot_download(repo_id=repo_id, local_dir=local_dir, resume_download=True, max_workers=4)

def load_asr_model(model_name, device, auto_download=False, source="ModelScope", use_aligner=False):
    """
    加载 ASR 模型
    use_aligner: 是否加载 ForcedAligner (用于时间戳)
    """
    if not HAS_QWEN_ASR:
        raise ImportError("Critical Dependency Missing: Please run 'pip install qwen-asr'")

    # --- 1. 准备主模型路径 ---
    target_folder_name = model_name.split("/")[-1] if "/" in model_name else model_name
    model_path = os.path.join(ASR_MODELS_DIR, target_folder_name)
    
    # 下载主模型
    if not os.path.exists(os.path.join(model_path, "config.json")):
        if auto_download:
            repo_id = model_name if "/" in model_name else f"Qwen/{model_name}"
            try:
                _download_model_logic(repo_id, model_path, source)
            except Exception as e:
                raise Exception(f"ASR Main Model Download failed: {e}")
        else:
            raise FileNotFoundError(f"ASR Model {model_name} not found at {model_path}. Please enable auto_download.")

    # --- 2. 准备 Aligner 路径 (如果开启时间戳) ---
    aligner_path = None
    if use_aligner:
        # 对齐模型固定使用 Qwen3-ForcedAligner-0.6B
        aligner_repo = "Qwen/Qwen3-ForcedAligner-0.6B"
        aligner_folder = "Qwen3-ForcedAligner-0.6B"
        aligner_local_path = os.path.join(ASR_MODELS_DIR, aligner_folder)
        
        # 检查对齐模型是否存在
        if not os.path.exists(os.path.join(aligner_local_path, "config.json")):
            if auto_download:
                print(f"[ASR] Timestamps requested. Downloading ForcedAligner...")
                try:
                    _download_model_logic(aligner_repo, aligner_local_path, source)
                    aligner_path = aligner_local_path
                except Exception as e:
                    raise Exception(f"ForcedAligner Download failed: {e}")
            else:
                # 关键修改：如果需要时间戳但没模型且没开下载，直接报错，不能静默跳过
                raise FileNotFoundError(
                    f"Timestamps generation requires 'Qwen3-ForcedAligner-0.6B'. "
                    f"Model not found at {aligner_local_path}. "
                    f"Please enable 'auto_download_model' to fetch it."
                )
        else:
            aligner_path = aligner_local_path

    # --- 3. 加载模型 (缓存键包含 aligner 状态) ---
    # 如果之前加载过无时间戳版，现在要用有时间戳版，必须重新加载
    cache_key = f"{model_path}_aligner:{use_aligner}"
    global LOADED_ASR_MODELS

    if cache_key not in LOADED_ASR_MODELS:
        # 如果显存不够，可以先卸载旧模型
        # unload_asr_model() 
        
        print(f"[ASR] Loading {model_name} (Aligner={use_aligner}) to {device}...")
        torch.cuda.empty_cache()
        
        # 根据设备选择精度
        dtype = torch.bfloat16 if "cuda" in str(device) and torch.cuda.is_bf16_supported() else torch.float16
        if device == "cpu": dtype = torch.float32

        try:
            load_kwargs = {
                "dtype": dtype,
                "device_map": device, 
            }
            
            # 关键修复：明确传入 forced_aligner 参数
            if use_aligner and aligner_path:
                print(f"[ASR] Attaching ForcedAligner from: {aligner_path}")
                load_kwargs["forced_aligner"] = aligner_path
                load_kwargs["forced_aligner_kwargs"] = {
                    "dtype": dtype,
                    "device_map": device
                }

            # Qwen3ASRModel 是 wrapper，不支持 eval()，直接初始化即可
            model = Qwen3ASRModel.from_pretrained(model_path, **load_kwargs)
            
            LOADED_ASR_MODELS[cache_key] = model
            print("[ASR] Model loaded successfully.")
        except Exception as e:
            # 加载失败时清理缓存键（如果有）
            if cache_key in LOADED_ASR_MODELS:
                del LOADED_ASR_MODELS[cache_key]
            raise Exception(f"Failed to load ASR model: {e}")

    return LOADED_ASR_MODELS[cache_key]

def unload_asr_model():
    """强制卸载所有 ASR 模型"""
    global LOADED_ASR_MODELS
    if LOADED_ASR_MODELS:
        print(f"[ASR] Unloading models to free VRAM...")
        LOADED_ASR_MODELS.clear()
        gc.collect()
        torch.cuda.empty_cache()