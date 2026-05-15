"""
学習データの読み込み、前処理、バッチ処理機能を提供する。
"""

import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons
from style_bert_vits2.models.hyper_parameters import HyperParametersData
from style_bert_vits2.nlp import (
    cleaned_text_to_sequence,
    convert_unsupported_phones_for_current_model,
    phone_symbols_to_duration_token_types,
)
from training.mel_processing import mel_spectrogram_torch, spectrogram_torch
from training.utils import load_filepaths_and_text, load_wav_to_torch


"""Multi speaker version"""


class TextAudioSpeakerLoader(Dataset[tuple[Any, ...]]):
    """
    1) loads audio, speaker_id, text pairs
    2) normalizes text and converts them to sequences of integers
    3) computes spectrograms from audio files.
    """

    def __init__(
        self,
        audiopaths_sid_text: str,
        hparams: HyperParametersData,
        wavs_dir: Path,
        spec_cache: bool = True,
        return_duration_token_types: bool = False,
    ):
        """
        TextAudioSpeakerLoader を初期化する。

        Args:
            audiopaths_sid_text (str): 音声パス・話者 ID・テキストを含むファイルのパス
            hparams (HyperParametersData): ハイパーパラメータ
            wavs_dir (Path): wavs ディレクトリのパス (train.list 内の相対パスの基準)
            spec_cache (bool): スペクトログラムをキャッシュするかどうか
            return_duration_token_types (bool): Nanairo duration 用のトークン種別を返すかどうか
        """

        self.wavs_dir = wavs_dir
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.spec_cache = spec_cache
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length
        self.sampling_rate = hparams.sampling_rate
        self.spk_map = hparams.spk2id
        self.hparams = hparams
        self.use_jp_extra = getattr(hparams, "use_jp_extra", False)
        self.use_nanairo = getattr(hparams, "use_nanairo", False)
        self.return_duration_token_types = return_duration_token_types

        self.use_mel_spec_posterior = getattr(
            hparams, "use_mel_posterior_encoder", False
        )
        if self.use_mel_spec_posterior:
            self.n_mel_channels = getattr(hparams, "n_mel_channels", 80)

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.use_speaker_embedding = getattr(hparams, "use_speaker_embedding", False)
        self.speaker_to_audio_paths: dict[str, list[str]] = {}

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 384)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()
        if self.use_speaker_embedding:
            self._build_speaker_to_audio_paths()

    def _filter(self) -> None:
        """
        Filter text & store spec lengths
        """

        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length
        # NOTE: OGG Vorbis 等の圧縮形式ではファイルサイズから spec 長を推定できないため、
        # WAV は従来どおりファイルサイズヒューリスティックを維持し、
        # それ以外は soundfile で実際のフレーム数を取得する

        audiopaths_sid_text_new = []
        lengths = []
        skipped = 0
        not_found = 0
        sr_mismatch = 0
        spk_unknown = 0
        logger.info("Init dataset...")
        for line_index, fields in enumerate(
            tqdm(self.audiopaths_sid_text, file=sys.stdout, dynamic_ncols=True)
        ):
            # train.list は 7 カラム (id|spk|language|text|phones|tone|word2ph) を期待する
            if len(fields) != 7:
                logger.warning(
                    f"Skipping malformed line {line_index}: "
                    f"expected 7 fields, got {len(fields)}"
                )
                skipped += 1
                continue
            _id, spk, language, text, phones, tone, word2ph = fields

            # _id は wavs_dir からの相対パスなので、フルパスを構築
            audiopath_path = self.wavs_dir / _id
            audiopath = str(audiopath_path)
            audiopath_lower = audiopath.lower()

            # ファイル存在チェック
            if not audiopath_path.exists():
                logger.warning(f"Audio file not found, skipping: {audiopath}")
                not_found += 1
                skipped += 1
                continue

            # 話者 ID が spk2id に存在するか確認
            if spk not in self.spk_map:
                logger.warning(
                    f"Unknown speaker '{spk}' not in spk2id, skipping: {audiopath}"
                )
                spk_unknown += 1
                skipped += 1
                continue

            phones = phones.split(" ")
            tone = [int(i) for i in tone.split(" ")]
            word2ph = [int(i) for i in word2ph.split(" ")]
            # サンプルレートの事前チェック
            ## get_audio() でも検証されるが、ここで早期に検出してスキップする方が
            ## 学習開始後にバッチの途中でクラッシュするより遥かにデバッグしやすい
            audio_info = sf.info(audiopath)
            if audio_info.samplerate != self.sampling_rate:
                logger.warning(
                    f"Sample rate mismatch, skipping: {audiopath} "
                    f"(expected: {self.sampling_rate}, "
                    f"got: {audio_info.samplerate})"
                )
                sr_mismatch += 1
                skipped += 1
                continue

            audiopaths_sid_text_new.append(
                [audiopath, spk, language, text, phones, tone, word2ph]
            )
            # WAV (16-bit PCM mono) はファイルサイズから spec 長を高速に推定する
            # これは従来の SBV2 と同一の挙動であり、既存 WAV データセットとの互換性を維持する
            if audiopath_lower.endswith(".wav"):
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            else:
                # 非 WAV (OGG Vorbis / FLAC 等) は soundfile.info() でヘッダから
                # 実際のフレーム数を取得する
                ## 圧縮形式ではファイルサイズとサンプル数が比例しないためヒューリスティックでは取得できない
                ## sf.info() はファイルヘッダのみ読むため十分高速
                lengths.append(audio_info.frames // self.hop_length)
        if not_found > 0:
            logger.warning(f"Audio files not found: {not_found}")
        if sr_mismatch > 0:
            logger.warning(f"Sample rate mismatches: {sr_mismatch}")
        if spk_unknown > 0:
            logger.warning(f"Unknown speakers: {spk_unknown}")
        logger.info(
            f"Dataset initialized. "
            f"total: {len(self.audiopaths_sid_text)}, "
            f"valid: {len(audiopaths_sid_text_new)}, "
            f"skipped: {skipped}"
        )
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def _build_speaker_to_audio_paths(self) -> None:
        """
        話者ごとの音声パス一覧を構築する。
        """

        self.speaker_to_audio_paths = {}
        for (
            audiopath,
            spk,
            _language,
            _text,
            _phones,
            _tone,
            _word2ph,
        ) in self.audiopaths_sid_text:
            self.speaker_to_audio_paths.setdefault(spk, []).append(audiopath)

    def _select_speaker_embedding_audio_path(self, audiopath: str, spk: str) -> str:
        """
        speaker embedding の参照元を選択する。

        Args:
            audiopath (str): 対象音声のパス
            spk (str): 話者名

        Returns:
            str: 参照元の音声パス
        """

        candidates = self.speaker_to_audio_paths.get(spk, [])
        if len(candidates) <= 1:
            return audiopath
        unique_candidates = list(dict.fromkeys(candidates))
        candidates_excluding = [
            candidate for candidate in unique_candidates if candidate != audiopath
        ]
        if not candidates_excluding:
            return audiopath
        # Avoid selecting the same utterance when possible
        return random.choice(candidates_excluding)

    def get_audio_text_speaker_pair(
        self,
        audiopath_sid_text: list[Any],
    ) -> tuple[Any, ...]:
        """
        音声・テキスト・話者のペアを取得する。

        Args:
            audiopath_sid_text (list[Any]): 音声パス、話者 ID、テキスト情報のリスト

        Returns:
            tuple[Any, ...]: 処理済みのデータタプル
        """

        # separate filename, speaker_id and text
        audiopath, sid, language, text, phones, tone, word2ph = audiopath_sid_text
        speaker_name = sid

        (
            bert,
            ja_bert,
            en_bert,
            phones,
            tone,
            language,
            duration_token_types,
        ) = self.get_text(text, word2ph, phones, tone, language, audiopath)

        spec, wav = self.get_audio(audiopath)
        sid = torch.LongTensor([int(self.spk_map[sid])])
        style_vec = torch.FloatTensor(np.load(f"{audiopath}.npy"))
        speaker_embedding: torch.Tensor | None = None
        if self.use_speaker_embedding:
            reference_audio_path = self._select_speaker_embedding_audio_path(
                audiopath, speaker_name
            )
            speaker_embedding_path = Path(f"{reference_audio_path}.spk.npy")
            if not speaker_embedding_path.exists():
                raise FileNotFoundError(
                    "Speaker embedding not found. "
                    f"speaker: {speaker_name}, audio: {audiopath}, "
                    f"expected: {speaker_embedding_path}"
                )
            try:
                speaker_embedding = torch.FloatTensor(
                    np.load(speaker_embedding_path)
                ).reshape(-1)
            except Exception as ex:
                raise RuntimeError(
                    "Failed to load speaker embedding. "
                    f"speaker: {speaker_name}, audio: {audiopath}, "
                    f"path: {speaker_embedding_path}"
                ) from ex
        if self.use_jp_extra:
            if self.use_speaker_embedding:
                assert speaker_embedding is not None
                sample = (
                    phones,
                    spec,
                    wav,
                    sid,
                    tone,
                    language,
                    ja_bert,
                    style_vec,
                )
                if self.return_duration_token_types is True:
                    assert duration_token_types is not None
                    return (*sample, duration_token_types, speaker_embedding)
                return (*sample, speaker_embedding)
            sample = (phones, spec, wav, sid, tone, language, ja_bert, style_vec)
            if self.return_duration_token_types is True:
                assert duration_token_types is not None
                return (*sample, duration_token_types)
            return sample
        else:
            if self.use_speaker_embedding:
                assert speaker_embedding is not None
                sample = (
                    phones,
                    spec,
                    wav,
                    sid,
                    tone,
                    language,
                    bert,
                    ja_bert,
                    en_bert,
                    style_vec,
                )
                if self.return_duration_token_types is True:
                    assert duration_token_types is not None
                    return (*sample, duration_token_types, speaker_embedding)
                return (*sample, speaker_embedding)
            sample = (
                phones,
                spec,
                wav,
                sid,
                tone,
                language,
                bert,
                ja_bert,
                en_bert,
                style_vec,
            )
            if self.return_duration_token_types is True:
                assert duration_token_types is not None
                return (*sample, duration_token_types)
            return sample

    def get_audio(self, filename: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        音声ファイルを読み込み、スペクトログラムと正規化された波形を返す。

        Args:
            filename (str): 音声ファイルのパス

        Returns:
            tuple[torch.Tensor, torch.Tensor]: スペクトログラムと正規化された波形のタプル
        """

        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{filename} {sampling_rate} SR doesn't match target {self.sampling_rate} SR"
            )
        # soundfile (OGG/FLAC) は float32 [-1, 1] を返すため正規化不要
        # WAV は scipy.io.wavfile で int16 raw スケール (max ~32768) を返すため max_wav_value で割る
        audio_max = audio.abs().max()
        if audio_max <= 1.0:
            # 既に [-1, 1] に正規化されている (soundfile 経由)
            audio_norm = audio
        elif audio_max <= self.max_wav_value * 1.1:
            # int16 raw スケール (scipy 経由の WAV)
            audio_norm = audio / self.max_wav_value
        else:
            # 想定外の値域 — 24-bit や 32-bit の raw データが混入している可能性がある
            raise ValueError(
                f"Unexpected audio value range. max: {audio_max:.1f}, "
                f"expected: <= 1.0 (float32) or <= {self.max_wav_value} (int16). "
                f"file: {filename}"
            )
        audio_norm = audio_norm.unsqueeze(0)
        if self.use_mel_spec_posterior:
            spec_filename = str(Path(filename).with_suffix(".mel.pt"))
        else:
            spec_filename = str(Path(filename).with_suffix(".spec.pt"))
        try:
            spec = torch.load(spec_filename)
        except Exception:
            if self.use_mel_spec_posterior:
                assert self.hparams.mel_fmin is not None
                assert self.hparams.mel_fmax is not None
                spec = mel_spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.n_mel_channels,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    self.hparams.mel_fmin,
                    self.hparams.mel_fmax,
                    center=False,
                )
            else:
                spec = spectrogram_torch(
                    audio_norm,
                    self.filter_length,
                    self.sampling_rate,
                    self.hop_length,
                    self.win_length,
                    center=False,
                )
            spec = torch.squeeze(spec, 0)
            if self.spec_cache:
                torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(
        self,
        text: str,
        word2ph: list[int],
        phone: list[str],
        tone: list[int],
        language_str: str,
        wav_path: str,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        テキストデータを処理し、BERT 特徴量と音素シーケンスを返す。

        Args:
            text (str): テキスト
            word2ph (list[int]): 単語から音素へのマッピング
            phone (list[str]): 音素リスト
            tone (list[int]): トーンリスト
            language_str (str): 言語文字列
            wav_path (str): 音声ファイルのパス

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: BERT 特徴量と音素シーケンスのタプル
        """

        # 変更を加える前にコピーを作成しておく
        phone = phone.copy()
        tone = tone.copy()
        word2ph = word2ph.copy()
        # g2p 処理では対応しているが現行モデルでは対応していない特定音素を、対応する音素にフォールバックする
        # 変更は引数で与えられた phone / tone / word2ph に in-place で適用される
        language = Languages[language_str]
        convert_unsupported_phones_for_current_model(
            phone,
            tone,
            word2ph,
            language,
        )
        duration_token_type_seq: list[int] | None = None
        if self.return_duration_token_types is True:
            # DP/SDP だけに渡す補助情報なので、音素 ID 化前の記号列から休止・境界・絵文字制御の粗い種別を作る
            duration_token_type_seq = phone_symbols_to_duration_token_types(
                phone,
                add_blank=self.add_blank,
            )
        phone_seq, tone_seq, language_seq = cleaned_text_to_sequence(
            phone,
            tone,
            language,
            use_nanairo=self.use_nanairo,
        )
        if self.add_blank:
            phone_seq = commons.intersperse(phone_seq, 0)
            tone_seq = commons.intersperse(tone_seq, 0)
            language_seq = commons.intersperse(language_seq, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1
        if duration_token_type_seq is not None and len(duration_token_type_seq) != len(
            phone_seq
        ):
            mismatch_msg = (
                f"duration_token_type_seq length ({len(duration_token_type_seq)}) != "
                f"phone sequence length ({len(phone_seq)}); verify "
                f"phone_symbols_to_duration_token_types matches cleaned_text_to_sequence "
                f"with add_blank / commons.intersperse"
            )
            logger.error(mismatch_msg)
            raise ValueError(mismatch_msg)
        bert_path = str(Path(wav_path).with_suffix(".bert.pt"))
        try:
            # DataLoader 上で BERT 特徴量を CPU テンソルとして扱うことで、multiprocessing ワーカー上で CUDA の再初期化が試みられて
            # "Cannot re-initialize CUDA in forked subprocess" エラーが発生する問題を回避する
            bert_ori = torch.load(
                bert_path,
                map_location=torch.device("cpu"),
            ).to(dtype=torch.float32)
            if bert_ori.shape[-1] != len(phone_seq):
                raise ValueError(
                    f"BERT length mismatch: {bert_ori.shape[-1]} vs {len(phone_seq)}"
                )
            bert_ori = bert_ori.contiguous()
        except Exception as ex:
            logger.error("Bert load failed")
            logger.error(ex)
            raise RuntimeError(f"Failed to load BERT features from {bert_path}") from ex

        if language_str == "ZH":
            bert = bert_ori
            ja_bert = torch.zeros(1024, len(phone_seq))
            en_bert = torch.zeros(1024, len(phone_seq))
        elif language_str == "JP":
            bert = torch.zeros(1024, len(phone_seq))
            ja_bert = bert_ori
            en_bert = torch.zeros(1024, len(phone_seq))
        elif language_str == "EN":
            bert = torch.zeros(1024, len(phone_seq))
            ja_bert = torch.zeros(1024, len(phone_seq))
            en_bert = bert_ori
        else:
            raise ValueError(f"Unsupported language: {language_str}")
        phone_tensor = torch.LongTensor(phone_seq)
        tone_tensor = torch.LongTensor(tone_seq)
        language_tensor = torch.LongTensor(language_seq)
        duration_token_type_tensor = (
            torch.LongTensor(duration_token_type_seq)
            if duration_token_type_seq is not None
            else None
        )
        return (
            bert,
            ja_bert,
            en_bert,
            phone_tensor,
            tone_tensor,
            language_tensor,
            duration_token_type_tensor,
        )

    def get_sid(self, sid: str) -> torch.Tensor:
        """
        話者 ID をテンソルに変換する。

        Args:
            sid (str): 話者 ID 文字列

        Returns:
            torch.Tensor: 話者 ID テンソル
        """

        sid_tensor = torch.LongTensor([int(sid)])
        return sid_tensor

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        """
        インデックスに対応するデータを取得する。

        Args:
            index (int): データインデックス

        Returns:
            tuple[Any, ...]: 処理済みのデータタプル
        """

        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self) -> int:
        """
        データセットのサイズを返す。

        Returns:
            int: データセットのサイズ
        """

        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate:
    """
    Zero-pads model inputs and targets
    """

    def __init__(
        self,
        return_ids: bool = False,
        use_jp_extra: bool = False,
        use_speaker_embedding: bool = False,
        return_duration_token_types: bool = False,
    ):
        """
        TextAudioSpeakerCollate を初期化する。

        Args:
            return_ids (bool): ID を返すかどうか
            use_jp_extra (bool): JP-Extra モデルを使用するかどうか
            use_speaker_embedding (bool): 話者埋め込みを使用するかどうか
            return_duration_token_types (bool): Nanairo duration 用のトークン種別を返すかどうか
        """

        self.return_ids = return_ids
        self.use_jp_extra = use_jp_extra
        self.use_speaker_embedding = use_speaker_embedding
        self.return_duration_token_types = return_duration_token_types

    def __call__(self, batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
        """
        Collates training batches from normalized text, audio, and speaker identities.

        Args:
            batch (list[tuple[Any, ...]]): データサンプルのリスト

        Returns:
            tuple[Any, ...]: パディングされたバッチデータ
        """

        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]), dim=0, descending=True
        )

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        tone_padded = torch.LongTensor(len(batch), max_text_len)
        language_padded = torch.LongTensor(len(batch), max_text_len)
        duration_token_types_padded: torch.Tensor | None = None
        if self.return_duration_token_types is True:
            duration_token_types_padded = torch.LongTensor(len(batch), max_text_len)
        # This is ZH bert if not use_jp_extra, JA bert if use_jp_extra
        bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        ja_bert_padded: torch.Tensor | None = None
        en_bert_padded: torch.Tensor | None = None
        if not self.use_jp_extra:
            ja_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
            en_bert_padded = torch.FloatTensor(len(batch), 1024, max_text_len)
        style_vec = torch.FloatTensor(len(batch), 256)
        speaker_embedding: torch.Tensor | None = None
        if self.use_speaker_embedding:
            if self.use_jp_extra:
                speaker_embedding_index = (
                    9 if self.return_duration_token_types is True else 8
                )
            else:
                speaker_embedding_index = (
                    11 if self.return_duration_token_types is True else 10
                )
            speaker_embedding = torch.FloatTensor(
                len(batch), batch[0][speaker_embedding_index].numel()
            )

        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        tone_padded.zero_()
        language_padded.zero_()
        if duration_token_types_padded is not None:
            duration_token_types_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        bert_padded.zero_()
        if not self.use_jp_extra:
            assert ja_bert_padded is not None
            assert en_bert_padded is not None
            ja_bert_padded.zero_()
            en_bert_padded.zero_()
        style_vec.zero_()
        if self.use_speaker_embedding:
            assert speaker_embedding is not None
            speaker_embedding.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            tone = row[4]
            tone_padded[i, : tone.size(0)] = tone

            language = row[5]
            language_padded[i, : language.size(0)] = language

            bert = row[6]
            bert_padded[i, :, : bert.size(1)] = bert

            if self.use_jp_extra:
                style_vec[i, :] = row[7]
                if self.return_duration_token_types is True:
                    assert duration_token_types_padded is not None
                    duration_token_types = row[8]
                    duration_token_types_padded[i, : duration_token_types.size(0)] = (
                        duration_token_types
                    )
                if self.use_speaker_embedding:
                    assert speaker_embedding is not None
                    speaker_embedding_index = (
                        9 if self.return_duration_token_types is True else 8
                    )
                    speaker_embedding[i, :] = row[speaker_embedding_index].reshape(-1)
            else:
                ja_bert = row[7]
                assert ja_bert_padded is not None
                ja_bert_padded[i, :, : ja_bert.size(1)] = ja_bert

                en_bert = row[8]
                assert en_bert_padded is not None
                en_bert_padded[i, :, : en_bert.size(1)] = en_bert
                style_vec[i, :] = row[9]
                if self.return_duration_token_types is True:
                    assert duration_token_types_padded is not None
                    duration_token_types = row[10]
                    duration_token_types_padded[i, : duration_token_types.size(0)] = (
                        duration_token_types
                    )
                if self.use_speaker_embedding:
                    assert speaker_embedding is not None
                    speaker_embedding_index = (
                        11 if self.return_duration_token_types is True else 10
                    )
                    speaker_embedding[i, :] = row[speaker_embedding_index].reshape(-1)

        if self.use_jp_extra:
            if self.use_speaker_embedding:
                assert speaker_embedding is not None
                batch_items: tuple[Any, ...] = (
                    text_padded,
                    text_lengths,
                    spec_padded,
                    spec_lengths,
                    wav_padded,
                    wav_lengths,
                    sid,
                    tone_padded,
                    language_padded,
                    bert_padded,
                    style_vec,
                )
                if self.return_duration_token_types is True:
                    assert duration_token_types_padded is not None
                    return (
                        *batch_items,
                        duration_token_types_padded,
                        speaker_embedding,
                    )
                return (*batch_items, speaker_embedding)
            batch_items = (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                style_vec,
            )
            if self.return_duration_token_types is True:
                assert duration_token_types_padded is not None
                return (*batch_items, duration_token_types_padded)
            return batch_items
        else:
            if self.use_speaker_embedding:
                assert speaker_embedding is not None
                assert ja_bert_padded is not None
                assert en_bert_padded is not None
                batch_items = (
                    text_padded,
                    text_lengths,
                    spec_padded,
                    spec_lengths,
                    wav_padded,
                    wav_lengths,
                    sid,
                    tone_padded,
                    language_padded,
                    bert_padded,
                    ja_bert_padded,
                    en_bert_padded,
                    style_vec,
                )
                if self.return_duration_token_types is True:
                    assert duration_token_types_padded is not None
                    return (
                        *batch_items,
                        duration_token_types_padded,
                        speaker_embedding,
                    )
                return (*batch_items, speaker_embedding)
            assert ja_bert_padded is not None
            assert en_bert_padded is not None
            batch_items = (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                tone_padded,
                language_padded,
                bert_padded,
                ja_bert_padded,
                en_bert_padded,
                style_vec,
            )
            if self.return_duration_token_types is True:
                assert duration_token_types_padded is not None
                return (*batch_items, duration_token_types_padded)
            return batch_items


class DistributedBucketSampler(DistributedSampler[int]):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: TextAudioSpeakerLoader,
        batch_size: int,
        boundaries: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
    ):
        """
        DistributedBucketSampler を初期化する。

        Args:
            dataset (TextAudioSpeakerLoader): データセット
            batch_size (int): バッチサイズ
            boundaries (list[int]): バケット境界
            num_replicas (int | None): レプリカ数
            rank (int | None): ランク
            shuffle (bool): シャッフルするかどうか
        """

        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        logger.info(f"Bucket info: {self.num_samples_per_bucket}")
        # logger.info(
        #     f"Unused samples: {len(self.lengths) - sum(self.num_samples_per_bucket)}"
        # )
        # ↑マイナスになることあるし、別にこれは使われないサンプル数ではないようだ……
        # バケットの仕組みはよく分からない

        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self) -> tuple[list[list[int]], list[int]]:
        """
        バケットを作成する。

        Returns:
            tuple[list[list[int]], list[int]]: バケットと各バケットのサンプル数
        """

        buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) - 1)]
        excluded_too_short: list[tuple[int, int]] = []
        excluded_too_long: list[tuple[int, int]] = []

        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
            elif length <= self.boundaries[0]:
                excluded_too_short.append((i, length))
            else:
                excluded_too_long.append((i, length))

        # 除外されたサンプルについて警告を出す
        # boundaries のフレーム数を秒数に変換（hop_length=512, sampling_rate=44100 を仮定）
        ## 実際の値は config に依存するが、ここでは一般的な値を使用
        hop_length = 512
        sampling_rate = 44100
        min_frames = self.boundaries[0]
        max_frames = self.boundaries[-1]
        min_sec = min_frames * hop_length / sampling_rate
        max_sec = max_frames * hop_length / sampling_rate

        if excluded_too_short:
            logger.warning(
                f"{len(excluded_too_short)} samples were excluded from training "
                f"(too short: <= {min_frames} frames, ~{min_sec:.2f} sec). "
                f"Consider removing or extending these audio files."
            )
            # 最初の5件を例示
            for idx, length in excluded_too_short[:5]:
                sec = length * hop_length / sampling_rate
                logger.warning(
                    f"  - Sample index {idx}: {length} frames (~{sec:.2f} sec)"
                )
            if len(excluded_too_short) > 5:
                logger.warning(f"  ... and {len(excluded_too_short) - 5} more")

        if excluded_too_long:
            logger.warning(
                f"{len(excluded_too_long)} samples were excluded from training "
                f"(too long: > {max_frames} frames, ~{max_sec:.2f} sec). "
                f"Consider splitting these audio files or using --not_use_custom_batch_sampler option."
            )
            # 最初の5件を例示
            for idx, length in excluded_too_long[:5]:
                sec = length * hop_length / sampling_rate
                logger.warning(
                    f"  - Sample index {idx}: {length} frames (~{sec:.2f} sec)"
                )
            if len(excluded_too_long) > 5:
                logger.warning(f"  ... and {len(excluded_too_long) - 5} more")

        total_excluded = len(excluded_too_short) + len(excluded_too_long)
        total_samples = len(self.lengths)
        if total_excluded > 0:
            included = total_samples - total_excluded
            logger.info(
                f"Bucket sampler: {included}/{total_samples} samples will be used for training "
                f"({total_excluded} excluded, {100 * total_excluded / total_samples:.1f}%)"
            )

        try:
            for i in range(len(buckets) - 1, 0, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            logger.info("Bucket warning ", e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self) -> Iterator[list[int]]:  # type: ignore[override]
        """
        イテレータを返す。

        Yields:
            Iterator[list[int]]: バッチデータ
        """

        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            if len_bucket == 0:
                continue
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(
        self,
        x: int,
        lo: int = 0,
        hi: int | None = None,
    ) -> int:
        """
        バイナリサーチでバケットインデックスを探す。

        Args:
            x (int): 長さ
            lo (int): 下限
            hi (int | None): 上限

        Returns:
            int: バケットインデックス（見つからない場合は -1）
        """

        boundaries = self.boundaries
        if hi is None:
            hi = len(boundaries) - 1

        while hi > lo:
            mid = (lo + hi) // 2
            if boundaries[mid] < x and x <= boundaries[mid + 1]:
                return mid
            if x <= boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return -1

    def __len__(self) -> int:
        """
        バッチ数を返す。

        Returns:
            int: バッチ数
        """

        return self.num_samples // self.batch_size
