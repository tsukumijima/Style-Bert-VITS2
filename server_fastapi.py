import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote

import GPUtil
import modal
import modal.gpu
import psutil
import torch
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from modal import Image
from scipy.io import wavfile

from config import get_config
from style_bert_vits2.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.user_dict import update_dict
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

app = modal.App("style-bert-vits2-server")
configs_mount = modal.Mount.from_local_dir(".", remote_path="/root/.")
image = (
    Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "torch",
        "scipy",
        "GPUtil",
        "psutil",
        "PyYAML",
        "loguru",
        "cmudict",
        "cn2an",
        "g2p_en",
        "jieba",
        "num2words",
        "numba",
        "numpy",
        "pydantic>=2.0",
        "pyopenjtalk-dict",
        "pypinyin",
        "pyworld-prebuilt",
        "safetensors",
        "transformers",
        "sentencepiece",
    )
    .copy_mount(configs_mount)
)
model_loading_sequence = ["ima_004_whisperE001", "ima_blend_004_tsukuE015", "ima_blend_004_tsukuN004"]


@app.function(image=image, gpu=[modal.gpu.H100()], keep_warm=2, container_idle_timeout=1800)
@modal.asgi_app()
def create_app():
    config = get_config()
    ln = config.server_config.language

    # Initialize pyopenjtalk and update dictionary
    pyopenjtalk.initialize_worker()
    update_dict()

    # Load BERT models and tokenizers
    for lang in [Languages.JP, Languages.EN, Languages.ZH]:
        bert_models.load_model(lang)
        bert_models.load_tokenizer(lang)

    fastapi_app = FastAPI()

    # Add CORS middleware if needed
    if config.server_config.origins:
        fastapi_app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Load models
    logger.info("Loading models...")

    model_dir = Path(config.assets_root)
    model_holder = TTSModelHolder(model_dir, "cuda")
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    loaded_models = []
    for model_name in model_loading_sequence:
        if model_name in model_holder.model_files_dict:
            model_paths = model_holder.model_files_dict[model_name]
            model = TTSModel(
                model_path=model_paths[0],
                config_path=model_holder.root_dir / model_name / "config.json",
                style_vec_path=model_holder.root_dir / model_name / "style_vectors.npy",
                device=model_holder.device,
            )
            loaded_models.append(model)
            logger.info(f"Loaded model: {model_name}")
        else:
            logger.warning(f"Model {model_name} not found in {model_dir}.")

    logger.info("All specified models loaded successfully.")

    class AudioResponse(Response):
        media_type = "audio/wav"

    def raise_validation_error(msg: str, param: str):
        logger.warning(f"Validation error: {msg}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
        )

    @fastapi_app.api_route("/voice", methods=["GET", "POST"], response_class=AudioResponse)
    async def voice(
        request: Request,
        text: str = Query(..., min_length=1, max_length=config.server_config.limit, description="セリフ"),
        encoding: Optional[str] = Query(None, description="textをURLデコードする(ex, `utf-8`)"),
        model_name: Optional[str] = Query(None, description="モデル名(model_idより優先)"),
        model_id: int = Query(0, description="モデルID"),
        speaker_name: Optional[str] = Query(None, description="話者名(speaker_idより優先)"),
        speaker_id: int = Query(0, description="話者ID"),
        sdp_ratio: float = Query(DEFAULT_SDP_RATIO, description="SDP/DP混合比"),
        noise: float = Query(DEFAULT_NOISE, description="サンプルノイズの割合"),
        noisew: float = Query(DEFAULT_NOISEW, description="SDPノイズ"),
        length: float = Query(DEFAULT_LENGTH, description="話速"),
        language: Languages = Query(ln, description="textの言語"),
        auto_split: bool = Query(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Query(DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"),
        assist_text: Optional[str] = Query(None, description="補助テキスト"),
        assist_text_weight: float = Query(DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"),
        style: Optional[str] = Query(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Query(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Query(None, description="スタイルを音声ファイルで行う"),
    ):
        # logger.info(f"{request.client.host}:{request.client.port}/voice  {unquote(str(request.query_params))}")

        if model_id >= len(loaded_models):
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        if model_name:
            model_ids = [i for i, model in enumerate(loaded_models) if model.config_path.parent.name == model_name]
            if not model_ids:
                raise_validation_error(f"model_name={model_name} not found", "model_name")
            if len(model_ids) > 1:
                raise_validation_error(f"model_name={model_name} is ambiguous", "model_name")
            model_id = model_ids[0]

        model = loaded_models[model_id]

        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(f"speaker_id={speaker_id} not found", "speaker_id")
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(f"speaker_name={speaker_name} not found", "speaker_name")
            speaker_id = model.spk2id[speaker_name]

        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")

        if encoding is not None:
            text = unquote(text, encoding=encoding)

        sr, audio = model.infer(
            text=text,
            language=language,
            speaker_id=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noise_w=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )

        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            return Response(content=wavContent.getvalue(), media_type="audio/wav")

    @fastapi_app.get("/models/info")
    def get_loaded_models_info():
        result: Dict[str, Dict[str, Any]] = {}
        for model_id, model in enumerate(loaded_models):
            result[str(model_id)] = {
                "config_path": str(model.config_path),
                "model_path": str(model.model_path),
                "device": model.device,
                "spk2id": model.spk2id,
                "id2spk": model.id2spk,
                "style2id": model.style2id,
            }
        return result

    # @fastapi_app.post("/models/refresh")
    # def refresh():
    #     nonlocal loaded_models
    #     loaded_models = load_models.remote()
    #     return get_loaded_models_info()

    @fastapi_app.get("/health")
    def health():
        return {"status": "ok"}

    @fastapi_app.get("/status")
    def get_status():
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        memory_total = memory_info.total
        memory_available = memory_info.available
        memory_used = memory_info.used
        memory_percent = memory_info.percent
        gpuInfo = []
        devices = ["cpu"]
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpuInfo.append(
                {
                    "gpu_id": gpu.id,
                    "gpu_load": gpu.load,
                    "gpu_memory": {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed,
                        "free": gpu.memoryFree,
                    },
                }
            )
        return {
            "devices": devices,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_used": memory_used,
            "memory_percent": memory_percent,
            "gpu": gpuInfo,
        }

    return fastapi_app
