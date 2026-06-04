"""
语音转文字模块 - 使用 sherpa-onnx SenseVoice Small 模型
支持并行解码，大幅提升长视频转写速度
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import soundfile as sf
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

# 模型目录（可配置）
_MODEL_ENV_VAR = "BILIBILI_SUMMARY_MODEL_DIR"
_LOCAL_MODEL = BASE_DIR / "models" / "sherpa-onnx-sense-voice-small"
_FALLBACK_MODEL = Path.home() / "VoiceDiscern" / "sherpa-onnx-sense-voice-small"

# 全局识别器（线程安全：只读推理）
_recognizer = None
_recognizer_lock = threading.Lock()

# 并行 worker 数：CPU 核心数的一半（留余量给系统和 ffmpeg），最少 2
_NUM_WORKERS = max(2, multiprocessing.cpu_count() // 2)


def _get_model_dir() -> Path:
    """获取模型目录（每次调用时检查，下载模型后无需重启即可生效）"""
    _env_model = os.environ.get(_MODEL_ENV_VAR, "").strip()
    if _env_model:
        p = Path(_env_model)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"环境变量 {_MODEL_ENV_VAR} 指定的模型目录不存在: {p}"
        )
    if _LOCAL_MODEL.exists():
        return _LOCAL_MODEL
    if _FALLBACK_MODEL.exists():
        return _FALLBACK_MODEL
    raise FileNotFoundError(
        f"模型目录未找到。请通过以下任一方式配置：\n"
        f"1. 设置环境变量 {_MODEL_ENV_VAR}\n"
        f"2. 将模型放在 {_LOCAL_MODEL}\n"
        f"3. 将模型放在 {_FALLBACK_MODEL}"
    )


def _load_recognizer():
    """延迟加载识别器"""
    global _recognizer
    if _recognizer is not None:
        return _recognizer

    with _recognizer_lock:
        if _recognizer is not None:
            return _recognizer

        import sherpa_onnx

        model_dir = _get_model_dir()
        model_path = str(model_dir / "model_q8.onnx")
        tokens_path = str(model_dir / "tokens.txt")

        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        logger.info("加载语音识别模型 %s", model_path)

        _recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            provider="cpu",
            num_threads=max(2, multiprocessing.cpu_count() // _NUM_WORKERS),
            sample_rate=16000,
            use_itn=True,
            language="",
            debug=False,
        )

        return _recognizer


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """简易重采样"""
    if len(data) == 0:
        return data
    scale = orig_sr / target_sr
    n_samples = int(len(data) / scale)
    indices = np.arange(n_samples) * scale
    return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


def _decode_chunk(recognizer, chunk: np.ndarray, sample_rate: int) -> str:
    """解码单个音频块，返回文本（线程安全：每个 stream 独立）"""
    stream = recognizer.create_stream()
    try:
        stream.accept_waveform(sample_rate, chunk)
        recognizer.decode_stream(stream)
        return stream.result.text or ""
    finally:
        del stream


def transcribe(wav_path: str, on_progress=None) -> dict:
    """
    对 WAV 文件进行语音转文字（并行解码）

    on_progress(current, total) - 进度回调
    返回 {"text": "...", "duration": float, "status": "success"|"error"}
    """
    try:
        # 1. 加载音频
        samples, sample_rate = sf.read(wav_path, dtype="int16")
        samples = samples.astype(np.float32) / 32768.0
        if len(samples.shape) > 1 and samples.shape[1] > 1:
            samples = samples.mean(axis=1)
        if sample_rate != 16000:
            samples = _resample(samples, sample_rate, 16000)
        samples = np.ascontiguousarray(samples)

        audio_duration = len(samples) / 16000

        # 2. 分块
        chunk_duration = 30.0
        chunk_samples = int(chunk_duration * 16000)
        total_chunks = max(1, (len(samples) + chunk_samples - 1) // chunk_samples)

        chunks = []
        for i in range(total_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, len(samples))
            chunks.append(np.ascontiguousarray(samples[start:end]))

        # 3. 加载识别器
        recognizer = _load_recognizer()

        # 4. 并行解码
        results = [""] * total_chunks
        completed = 0
        lock = threading.Lock()

        # 短音频（≤ 3 块）直接串行，避免线程开销
        if total_chunks <= 3:
            for i, chunk in enumerate(chunks):
                results[i] = _decode_chunk(recognizer, chunk, 16000)
                if on_progress:
                    on_progress(i + 1, total_chunks)
        else:
            workers = min(_NUM_WORKERS, total_chunks)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_idx = {
                    executor.submit(_decode_chunk, recognizer, chunk, 16000): i
                    for i, chunk in enumerate(chunks)
                }

                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception:
                        results[idx] = ""

                    with lock:
                        completed += 1
                        if on_progress:
                            on_progress(completed, total_chunks)

        # 5. 合并结果
        text = " ".join(t for t in results if t).strip()

        return {
            "text": text,
            "duration": round(audio_duration, 1),
            "status": "success",
            "chunks": total_chunks,
        }

    except FileNotFoundError as e:
        return {"text": "", "duration": 0, "status": "error", "error": str(e)}
    except Exception as e:
        return {"text": "", "duration": 0, "status": "error", "error": str(e)}


def load_recognizer_for_status() -> bool:
    """尝试加载识别器，返回是否成功"""
    try:
        _load_recognizer()
        return True
    except Exception:
        return False
