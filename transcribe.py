import argparse
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from tqdm import tqdm

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.paths import add_model_argument, get_paths_config
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# faster-whisperは並列処理しても速度が向上しないので、単一モデルでループ処理する
def transcribe_with_faster_whisper(
    model: "WhisperModel",
    audio_file: Path,
    initial_prompt: str | None = None,
    language: str = "ja",
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
):
    segments, _ = model.transcribe(
        str(audio_file),
        beam_size=num_beams,
        language=language,
        initial_prompt=initial_prompt,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    texts = [segment.text for segment in segments]
    return "".join(texts)


# HF pipelineで進捗表示をするために必要なDatasetクラス
class StrListDataset(Dataset[str]):
    def __init__(self, original_list: list[str]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> str:
        return self.original_list[i]


def _write_transcription_results(
    output_file: Path,
    input_dir: Path,
    model_name: str,
    language_id: str,
    audio_files: list[Path],
    results: list[str],
) -> None:
    """
    書き起こし結果をファイルへ保存する。

    Args:
        output_file (Path): 出力ファイルパス
        input_dir (Path): 入力音声のルートディレクトリ
        model_name (str): モデル名
        language_id (str): 言語 ID
        audio_files (list[Path]): 音声ファイル一覧
        results (list[str]): 書き起こし結果一覧
    """

    lines: list[str] = []
    for audio_file, text in zip(audio_files, results):
        wav_rel_path = audio_file.relative_to(input_dir)
        lines.append(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")
    with open(output_file, "a", encoding="utf-8") as f:
        f.writelines(lines)


# HFのWhisperはファイルリストを与えるとバッチ処理ができて速い
def transcribe_files_with_hf_whisper(
    audio_files: list[Path],
    model_id: str,
    input_dir: Path,
    model_name: str,
    language_id: str,
    output_file: Path,
    initial_prompt: str | None = None,
    language: str = "ja",
    batch_size: int = 16,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
    device: str = "cuda",
    pbar: tqdm[Any] | None = None,  # type: ignore[type-arg]
) -> list[str]:
    import torch
    from transformers import WhisperProcessor, pipeline

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_id)  # type: ignore[assignment]
    generate_kwargs: dict[str, Any] = {
        "language": language,
        "do_sample": False,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    logger.info(f"generate_kwargs: {generate_kwargs}, loading pipeline...")
    pipe = pipeline(  # pyright: ignore[reportCallIssue]
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        trust_remote_code=True,
        # generate_kwargs=generate_kwargs,
    )
    logger.info("Loaded pipeline")
    if initial_prompt is not None:
        if pipe.tokenizer is None:
            raise ValueError("Pipeline tokenizer is None")
        prompt_ids: torch.Tensor = pipe.tokenizer.get_prompt_ids(  # type: ignore[union-attr]
            initial_prompt, return_tensors="pt"
        ).to(device)
        generate_kwargs["prompt_ids"] = prompt_ids

    dataset = StrListDataset([str(f) for f in audio_files])

    results: list[str] = []
    pipe_results = pipe(dataset, generate_kwargs=generate_kwargs)
    if pipe_results is None:
        raise ValueError("Pipeline returned None")
    for whisper_result in pipe_results:
        if not isinstance(whisper_result, dict) or "text" not in whisper_result:
            raise ValueError(f"Unexpected pipeline result format: {whisper_result}")
        text: str = str(whisper_result["text"])
        # なぜかテキストの最初に" {initial_prompt}"が入るので、文字の最初からこれを削除する
        # cf. https://github.com/huggingface/transformers/issues/27594
        if text.startswith(f" {initial_prompt}"):
            text = text[len(f" {initial_prompt}") :]
        results.append(text)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    _write_transcription_results(
        output_file=output_file,
        input_dir=input_dir,
        model_name=model_name,
        language_id=language_id,
        audio_files=audio_files,
        results=results,
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
    )
    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )
    parser.add_argument("--whisper-model", type=str, default="large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compute_type", type=str, default="bfloat16")
    parser.add_argument("--use_hf_whisper", action="store_true")
    parser.add_argument("--hf_repo_id", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    args = parser.parse_args()

    paths_config = get_paths_config()
    model_name = str(args.model)

    input_dir = paths_config.dataset_root / model_name / "raw"
    output_file = paths_config.dataset_root / model_name / "esd.list"
    initial_prompt: str = args.initial_prompt
    initial_prompt = initial_prompt.strip('"')
    language: str = args.language
    device: str = args.device
    compute_type: str = args.compute_type
    batch_size: int = args.batch_size
    num_beams: int = args.num_beams
    no_repeat_ngram_size: int = args.no_repeat_ngram_size

    output_file.parent.mkdir(parents=True, exist_ok=True)

    wav_files = [f for f in input_dir.rglob("*.wav") if f.is_file()]
    wav_files = sorted(wav_files, key=lambda x: str(x))
    logger.info(f"Found {len(wav_files)} WAV files")
    if len(wav_files) == 0:
        logger.warning(f"No WAV files found in {input_dir}")
        sys.exit(1)

    if output_file.exists():
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        backup_path = output_file.with_name(output_file.name + ".bak")
        if backup_path.exists():
            logger.warning(f"{output_file}.bak exists, deleting...")
            backup_path.unlink()
        output_file.rename(backup_path)

    if language == "ja":
        language_id = Languages.JP.value
    elif language == "en":
        language_id = Languages.EN.value
    elif language == "zh":
        language_id = Languages.ZH.value
    else:
        raise ValueError(f"{language} is not supported.")

    if not args.use_hf_whisper:
        from faster_whisper import WhisperModel

        logger.info(
            f"Loading faster-whisper model ({args.whisper_model}) with compute_type: {compute_type}"
        )
        try:
            model = WhisperModel(
                args.whisper_model, device=device, compute_type=compute_type
            )
        except ValueError as e:
            logger.warning(f"Failed to load model, so use `auto` compute_type: {e}")
            model = WhisperModel(args.whisper_model, device=device)
        for wav_file in tqdm(wav_files, file=SAFE_STDOUT, dynamic_ncols=True):
            text = transcribe_with_faster_whisper(
                model=model,
                audio_file=wav_file,
                initial_prompt=initial_prompt,
                language=language,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            wav_rel_path = wav_file.relative_to(input_dir)
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")
    else:
        model_id = args.hf_repo_id
        logger.info(f"Loading HF Whisper model ({model_id})")
        pbar = tqdm(total=len(wav_files), file=SAFE_STDOUT, dynamic_ncols=True)
        results = transcribe_files_with_hf_whisper(
            audio_files=wav_files,
            model_id=model_id,
            output_file=output_file,
            input_dir=input_dir,
            model_name=model_name,
            language_id=language_id,
            initial_prompt=initial_prompt,
            language=language,
            batch_size=batch_size,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            device=device,
            pbar=pbar,
        )
        # with open(output_file, "w", encoding="utf-8") as f:
        #     for wav_file, text in zip(wav_files, results):
        #         wav_rel_path = wav_file.relative_to(input_dir)
        #         f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")

    sys.exit(0)
