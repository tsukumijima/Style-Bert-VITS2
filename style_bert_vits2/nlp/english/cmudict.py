import pickle
from pathlib import Path


CMU_DICT_PATH = Path(__file__).parent / "cmudict.rep"
CMU_CACHE_PATH = Path(__file__).parent / "cmudict_cache.pickle"
SHORT_FORM_DICT_PATH = Path(__file__).parent / "short_form_dict.rep"
SHORT_FORM_CACHE_PATH = Path(__file__).parent / "short_form_dict_cache.pickle"


def get_dict() -> dict[str, list[list[str]]]:
    if CMU_CACHE_PATH.exists():
        return _load_cached_dict(CMU_CACHE_PATH)
    g2p_dict = _read_dict(CMU_DICT_PATH, start_line=49)
    _cache_dict(g2p_dict, CMU_CACHE_PATH)
    return g2p_dict


def get_shortform_dict() -> dict[str, list[list[str]]]:
    if SHORT_FORM_CACHE_PATH.exists():
        return _load_cached_dict(SHORT_FORM_CACHE_PATH)
    g2p_dict = _read_dict(SHORT_FORM_DICT_PATH, start_line=1)
    _cache_dict(g2p_dict, SHORT_FORM_CACHE_PATH)
    return g2p_dict


def _read_dict(file_path: Path, start_line: int) -> dict[str, list[list[str]]]:
    g2p_dict = {}
    start_line = start_line
    with open(file_path, encoding="utf-8") as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def _cache_dict(g2p_dict: dict[str, list[list[str]]], file_path: Path) -> None:
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def _load_cached_dict(file_path: Path) -> dict[str, list[list[str]]]:
    with open(file_path, "rb") as pickle_file:
        return pickle.load(pickle_file)
