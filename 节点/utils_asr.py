import os
import torch
import gc
import functools
import folder_paths
from huggingface_hub import snapshot_download as hf_snapshot_download

# ========== 兼容性补丁 1：check_model_inputs 装饰器 ==========
# 修复 qwen-asr 0.0.6 与 transformers 4.57.3+ 的不兼容问题。
# qwen-asr 中使用 @check_model_inputs（不带括号）或 @check_model_inputs()（带括号），
# 但 transformers 4.57.3+ 中 check_model_inputs 是装饰器工厂，且内部 wrapper 可能过滤参数。
# 为确保 qwen-asr 的模型 forward 方法能正确接收任意参数，将其替换为完全透明版本。

def _patch_check_model_inputs():
    """将 transformers 的 check_model_inputs 替换为完全透明的装饰器"""
    try:
        import transformers.utils.generic as _tg

        def _transparent_decorator(func):
            """完全透明的装饰器，不做任何参数过滤，直接透传"""
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper

        # 兼容 @check_model_inputs、@check_model_inputs()、@check_model_inputs(tie_last_hidden_states=False) 等所有用法
        def _compat_wrapper(func=None, *args, **kwargs):
            if func is not None and callable(func):
                # @check_model_inputs 不带括号
                return _transparent_decorator(func)
            # @check_model_inputs() 带括号，或 @check_model_inputs(True) 等
            return _transparent_decorator

        _tg.check_model_inputs = _compat_wrapper
    except (ImportError, AttributeError):
        pass

_patch_check_model_inputs()


# ========== 兼容性补丁 3：ROPE_INIT_FUNCTIONS 缺少 'default' 键 ==========
# qwen-asr 的模型配置中 rope_type 为 'default'（原始 RoPE 实现，无缩放），
# 但 transformers 5.6.2 的 ROPE_INIT_FUNCTIONS 字典中未包含该键。

def _patch_rope_init_functions():
    """修复 ROPE_INIT_FUNCTIONS 缺少 'default' 键的问题

    qwen-asr 的 Qwen3ASRThinkerTextRotaryEmbedding.__init__ 中使用
    ROPE_INIT_FUNCTIONS[self.rope_type]，当 rope_type 为 'default' 时触发 KeyError。
    'default' 表示原始 RoPE 实现（无缩放），不是 linear scaling。
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if 'default' not in ROPE_INIT_FUNCTIONS:
            import torch

            def _compute_default_rope_parameters(config, device=None, seq_len=None, layer_type=None):
                """原始 RoPE 参数计算（无缩放）"""
                config.standardize_rope_params()
                rope_parameters_dict = (
                    config.rope_parameters[layer_type]
                    if layer_type is not None
                    else config.rope_parameters
                )
                base = rope_parameters_dict["rope_theta"]
                partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                head_dim = (
                    getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads
                )
                dim = int(head_dim * partial_rotary_factor)
                attention_factor = 1.0

                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dim, 2, dtype=torch.int64).to(
                            device=device, dtype=torch.float
                        )
                        / dim
                    )
                )
                return inv_freq, attention_factor

            ROPE_INIT_FUNCTIONS['default'] = _compute_default_rope_parameters
    except (ImportError, AttributeError):
        pass

_patch_rope_init_functions()

# --- 尝试导入 ModelScope ---
try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    HAS_MODELSCOPE = True
except ImportError:
    HAS_MODELSCOPE = False

# --- 尝试导入 Qwen-ASR ---
try:
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    HAS_QWEN_ASR = True
except ImportError:
    HAS_QWEN_ASR = False
    Qwen3ASRConfig = None

# ========== 兼容性补丁 2：Qwen3ASRConfig thinker_config ==========
# transformers 5.6.2 的 @strict 装饰器会在 super().__init__() 中触发
# validate_token_ids() 验证，该验证需要访问 self.thinker_config。
# 但原始 __init__ 在 super().__init__() 之后才设置 thinker_config，
# 导致验证时属性不存在。修复方案：在调用 super().__init__() 之前先初始化子配置。

def _patch_qwen3_asr_config():
    """修复 Qwen3ASRConfig.thinker_config 属性缺失问题

    Qwen3ASRThinkerConfig 也有相同的代码模式（在 super().__init__() 之后才设置
    audio_config 和 text_config），出于防御性考虑一并修复。
    """
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
            Qwen3ASRConfig as _Cfg,
            Qwen3ASRThinkerConfig as _ThinkerCfg,
            Qwen3ASRAudioEncoderConfig as _AudioCfg,
            Qwen3ASRTextConfig as _TextCfg,
        )
        try:
            from transformers import PreTrainedConfig
        except ImportError:
            from transformers import PretrainedConfig as PreTrainedConfig

        # --- 补丁 1: Qwen3ASRConfig ---
        def _patched_cfg_init(self, thinker_config=None, support_languages=None, **kwargs):
            # 在调用 super().__init__() 之前先设置 thinker_config
            # 这样 @strict 验证器访问时属性已存在
            if thinker_config is None:
                thinker_config = {}
            if isinstance(thinker_config, dict):
                self.thinker_config = _ThinkerCfg(**thinker_config)
            elif isinstance(thinker_config, _ThinkerCfg):
                self.thinker_config = thinker_config
            else:
                self.thinker_config = thinker_config

            self.support_languages = support_languages

            # 现在安全调用 super().__init__()
            PreTrainedConfig.__init__(self, **kwargs)

        _Cfg.__init__ = _patched_cfg_init

        # --- 补丁 2: Qwen3ASRThinkerConfig ---
        # 虽然 Qwen3ASRThinkerConfig 在当前版本不会触发 AttributeError
        # （因为 validate_token_ids 会安全回退），但其代码模式与 Qwen3ASRConfig
        # 完全一致，出于防御性考虑一并修复。
        def _patched_thinker_init(
            self,
            audio_config=None,
            text_config=None,
            audio_token_id=151646,
            audio_start_token_id=151647,
            user_token_id=872,
            initializer_range=0.02,
            **kwargs,
        ):
            # 在调用 super().__init__() 之前先设置子配置
            if isinstance(audio_config, dict):
                self.audio_config = _AudioCfg(**audio_config)
            elif audio_config is None:
                self.audio_config = _AudioCfg()
            else:
                self.audio_config = audio_config

            if isinstance(text_config, dict):
                self.text_config = _TextCfg(**text_config)
            elif text_config is None:
                self.text_config = _TextCfg()
            else:
                self.text_config = text_config

            self.audio_token_id = audio_token_id
            self.audio_start_token_id = audio_start_token_id
            self.user_token_id = user_token_id
            self.initializer_range = initializer_range

            # 现在安全调用 super().__init__()
            PreTrainedConfig.__init__(self, **kwargs)

        _ThinkerCfg.__init__ = _patched_thinker_init

    except (ImportError, AttributeError) as e:
        print(f"[ComfyUI-ASR] Warning: Failed to patch Qwen3ASRConfig: {e}")

_patch_qwen3_asr_config()

# ========== 兼容性补丁 4：Qwen3ASRThinkerTextRotaryEmbedding rope_scaling 空指针 ==========
# qwen-asr 0.0.6 的 modeling_qwen3_asr.py 第 800 行：
#   self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
# 当 config.rope_scaling 为 None 时，None.get() 会抛出 AttributeError。
# 修复方案：monkey-patch __init__，在访问 rope_scaling 前进行 None 检查。

def _patch_rotary_embedding_init():
    """修复 Qwen3ASRThinkerTextRotaryEmbedding 中 rope_scaling 为 None 时的崩溃"""
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerTextRotaryEmbedding as _RotaryEmbed,
        )
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        def _patched_rotary_init(self, config, device=None):
            # 不能用无参 super()，必须通过 type(self) 定位正确的父类
            super(type(self), self).__init__()
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get("rope_type", "default")
            else:
                self.rope_type = "default"

            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

            # 修复：安全访问 rope_scaling，避免 None.get() 崩溃
            if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
                self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])
            else:
                self.mrope_section = [24, 20, 20]

        _RotaryEmbed.__init__ = _patched_rotary_init

        # 修复：transformers 5.6.2 的 _init_weights 在 rope_type='default' 时
        # 会调用 module.compute_default_rope_parameters 而非 ROPE_INIT_FUNCTIONS['default']
        if not hasattr(_RotaryEmbed, 'compute_default_rope_parameters'):
            _RotaryEmbed.compute_default_rope_parameters = property(lambda self: self.rope_init_fn)
    except (ImportError, AttributeError) as e:
        print(f"[ComfyUI-ASR] Warning: Failed to patch Qwen3ASRThinkerTextRotaryEmbedding: {e}")

_patch_rotary_embedding_init()

# ========== 兼容性补丁 5：Qwen3ASRThinkerForConditionalGeneration 属性访问 ==========
# qwen-asr 0.0.6 的 modeling_qwen3_asr.py 第 1089 行：
#   self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
# Qwen3ASRThinkerConfig 没有 pad_token_id 属性，transformers 5.6.2 严格检查下抛出 AttributeError。
# 同样 config.classify_num 在 forced_aligner 模型中可能不存在。
# 修复方案：monkey-patch __init__，使用 getattr 安全访问。

def _patch_thinker_for_cg_init():
    """修复 Qwen3ASRThinkerForConditionalGeneration.__init__ 中的属性访问问题"""
    try:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
            Qwen3ASRThinkerForConditionalGeneration as _ThinkerCG,
            Qwen3ASRAudioEncoder,
            Qwen3ASRThinkerTextModel,
        )
        import torch.nn as nn

        def _patched_thinker_cg_init(self, config):
            # 不能用无参 super()，必须通过 type(self) 定位正确的父类
            super(type(self), self).__init__(config)

            self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
            self.vocab_size = config.text_config.vocab_size
            self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)

            if "forced_aligner" in config.model_type:
                # 安全访问 classify_num，如果不存在则回退到 vocab_size
                classify_num = getattr(config, "classify_num", config.text_config.vocab_size)
                self.lm_head = nn.Linear(config.text_config.hidden_size, classify_num, bias=False)
            else:
                self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

            # 修复：安全访问 pad_token_id
            self.pad_token_id = getattr(self.config, "pad_token_id", None)
            if self.pad_token_id is None:
                self.pad_token_id = -1

            self.rope_deltas = None
            self.post_init()

        _ThinkerCG.__init__ = _patched_thinker_cg_init
    except (ImportError, AttributeError) as e:
        print(f"[ComfyUI-ASR] Warning: Failed to patch Qwen3ASRThinkerForConditionalGeneration: {e}")

_patch_thinker_for_cg_init()

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