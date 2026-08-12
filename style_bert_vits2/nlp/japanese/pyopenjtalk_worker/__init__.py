"""
Run the pyopenjtalk worker in a separate process
to avoid user dictionary access error
"""

from __future__ import annotations

from typing import Any, Literal

from pyopenjtalk import NJDFeature, OpenJTalk

from style_bert_vits2.logging import logger
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_client import WorkerClient
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.worker_common import WORKER_PORT


WORKER_CLIENT: WorkerClient | None = None


# pyopenjtalk interface
# g2p(): not used


def run_frontend(
    text: str,
    *,
    run_marine: bool = False,
    use_vanilla: bool = False,
    use_tsqyomi: bool = False,
    use_sudachi_kanji_yomi: bool = True,
    predict_nani: bool = True,
    normalize_mode: Literal["None", "NFC", "NFKC"] = "None",
    use_read_as_pron: bool = False,
    revert_long_vowels: bool = False,
    revert_yotsugana: bool = False,
    jtalk: OpenJTalk | None = None,
) -> list[NJDFeature]:
    # tsqyomi は呼び出し元プロセス側でモデルをロードする前提のため、ワーカー経由では扱えない
    # jtalk も OpenJTalk インスタンスを JSON 経由で渡せないため、指定時はワーカーを迂回する
    if WORKER_CLIENT is not None and use_tsqyomi is False and jtalk is None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("run_frontend", text)
        assert isinstance(ret, list)
        return ret
    else:
        # without worker
        import pyopenjtalk

        return pyopenjtalk.run_frontend(
            text,
            run_marine=run_marine,
            use_vanilla=use_vanilla,
            use_tsqyomi=use_tsqyomi,
            use_sudachi_kanji_yomi=use_sudachi_kanji_yomi,
            predict_nani=predict_nani,
            normalize_mode=normalize_mode,
            use_read_as_pron=use_read_as_pron,
            revert_long_vowels=revert_long_vowels,
            revert_yotsugana=revert_yotsugana,
            jtalk=jtalk,
        )


def make_label(
    njd_features: list[NJDFeature],
    jtalk: OpenJTalk | None = None,
) -> list[str]:
    if WORKER_CLIENT is not None:
        ret = WORKER_CLIENT.dispatch_pyopenjtalk("make_label", njd_features)
        assert isinstance(ret, list)
        return ret
    else:
        # without worker
        import pyopenjtalk

        return pyopenjtalk.make_label(njd_features, jtalk)


def mecab_dict_index(path: str, out_path: str, dn_mecab: str | None = None) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("mecab_dict_index", path, out_path, dn_mecab)
    else:
        # without worker
        import pyopenjtalk

        pyopenjtalk.mecab_dict_index(path, out_path, dn_mecab)


def update_global_jtalk_with_user_dict(path: str) -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("update_global_jtalk_with_user_dict", path)
    else:
        # without worker
        import pyopenjtalk

        pyopenjtalk.update_global_jtalk_with_user_dict(path)


def unset_user_dict() -> None:
    if WORKER_CLIENT is not None:
        WORKER_CLIENT.dispatch_pyopenjtalk("unset_user_dict")
    else:
        # without worker
        import pyopenjtalk

        pyopenjtalk.unset_user_dict()


# initialize module when imported


def initialize_worker(port: int = WORKER_PORT) -> None:
    import atexit
    import signal
    import sys
    import time

    global WORKER_CLIENT
    if WORKER_CLIENT:
        return

    client = None
    try:
        client = WorkerClient(port)
    except (TimeoutError, OSError):
        logger.debug("try starting pyopenjtalk worker server")
        import os
        import subprocess

        worker_pkg_path = os.path.relpath(
            os.path.dirname(__file__), os.getcwd()
        ).replace(os.sep, ".")
        args = [sys.executable, "-m", worker_pkg_path, "--port", str(port)]
        # new session, new process group
        if sys.platform.startswith("win"):
            cf = subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore
            si = subprocess.STARTUPINFO()  # type: ignore
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore
            si.wShowWindow = subprocess.SW_HIDE  # type: ignore
            subprocess.Popen(args, creationflags=cf, startupinfo=si)
        else:
            # align with Windows behavior
            # start_new_session is same as specifying setsid in preexec_fn
            subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # wait until server listening
        count = 0
        while True:
            try:
                client = WorkerClient(port)
                break
            except OSError:
                time.sleep(0.5)
                count += 1
                # 20: max number of retries
                if count == 20:
                    raise TimeoutError("サーバーに接続できませんでした")

    logger.debug("pyopenjtalk worker server started")
    WORKER_CLIENT = client
    atexit.register(terminate_worker)

    # when the process is killed
    def signal_handler(signum: int, frame: Any):
        terminate_worker()

    try:
        signal.signal(signal.SIGTERM, signal_handler)
    except ValueError:
        # signal only works in main thread
        pass


# top-level declaration
def terminate_worker() -> None:
    global WORKER_CLIENT
    if not WORKER_CLIENT:
        return
    logger.debug("pyopenjtalk worker server terminated")

    # prepare for unexpected errors
    try:
        if WORKER_CLIENT.status() == 1:
            WORKER_CLIENT.quit_server()
    except Exception as e:
        logger.error(e)

    WORKER_CLIENT.close()
    WORKER_CLIENT = None
