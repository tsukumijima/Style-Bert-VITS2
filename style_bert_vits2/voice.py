import numpy as np
import pyworld
from numpy.typing import NDArray


def adjust_voice(
    fs: int,
    wave: NDArray[np.float32],
    pitch_scale: float = 1.0,
    intonation_scale: float = 1.0,
) -> tuple[int, NDArray[np.float32]]:
    """
    音声のピッチと抑揚を調整する。
    変更すると若干音質が劣化するので、どちらも初期値のままならそのまま返す。

    Args:
        fs (int): 音声のサンプリング周波数
        wave (NDArray[np.float32]): 音声データ
        pitch_scale (float, optional): ピッチの高さ. Defaults to 1.0.
        intonation_scale (float, optional): 抑揚の平均からの変更比率. Defaults to 1.0.

    Returns:
        tuple[int, NDArray[np.float32]]: 調整後の音声データのサンプリング周波数と音声データ
    """

    if pitch_scale == 1.0 and intonation_scale == 1.0:
        # 初期値の場合は、音質劣化を避けるためにそのまま返す
        return fs, wave

    # pyworld で f0 を加工して合成
    # pyworld よりもよいのがあるかもしれないが……

    # pyworld での処理のために double に変換 (念のため)
    wave_double = wave.astype(np.double)

    # 質が高そうだしとりあえずharvestにしておく
    f0, t = pyworld.harvest(wave_double, fs)

    sp = pyworld.cheaptrick(wave_double, f0, t, fs)
    ap = pyworld.d4c(wave_double, f0, t, fs)

    non_zero_f0 = [f for f in f0 if f != 0]

    # 非ゼロの f0 が存在しない場合（無音に近い場合など）は元の音声をそのまま返す
    if len(non_zero_f0) == 0:
        return fs, wave  # 元の (float32 の) wave を返す

    f0_mean = sum(non_zero_f0) / len(non_zero_f0)

    for i, f_val in enumerate(f0):
        if f_val == 0:
            continue
        f0[i] = pitch_scale * f0_mean + intonation_scale * (f_val - f0_mean)

    synthesized_wave_double = pyworld.synthesize(f0, sp, ap, fs)

    # 最終的に float32 に変換してから返す
    return fs, synthesized_wave_double.astype(np.float32)
