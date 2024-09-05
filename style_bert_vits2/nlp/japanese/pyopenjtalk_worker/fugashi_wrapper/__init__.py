from pathlib import Path
import os

from fugashi import build_dictionary  # type: ignore
import unidic

from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.fugashi_wrapper.unidic_file import DICRC, RWRITE_DEF
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.fugashi_wrapper.left_id_def import LEFT_ID_DEF
from style_bert_vits2.nlp.japanese.pyopenjtalk_worker.fugashi_wrapper.right_id_def import RIGHT_ID_DEF
def fugashi_user_dict(compiled_dict_path: str, tmp_csv_path: str):
    # unidic-pyのunidicのpath
    unidic_path = unidic.DICDIR

    # windows環境だとパスが途中でエスケープされるバグがある(windows環境で引数指定する人がいないのかも)
    if os.name == "nt":
        # コンパイル後の辞書のpath
        compiled_dict_path = compiled_dict_path.replace("\\", "\\\\")
        # csvファイルのパス
        tmp_csv_path = tmp_csv_path.replace("\\", "\\\\")
        unidic_path = unidic_path.replace("\\", "\\\\")

    # 作業ディレクトリにリソースファイル等を作る必要がある

    current_dir = Path.cwd()
    dicrc_path = current_dir / "dicrc"
    rerwrite_def_path = current_dir / "rewrite.def"
    left_id_def_path = rerwrite_def_path = current_dir / "left_id.def"
    right_id_def_path = rerwrite_def_path = current_dir / "right_id..def"

    dicrc_path.write_text(DICRC, encoding="shift_jis")
    rerwrite_def_path.write_text(RWRITE_DEF, encoding="shift_jis")

    left_id_def_path.write_text(LEFT_ID_DEF, "utf-8")
    right_id_def_path.write_text(RIGHT_ID_DEF, "utf-8")

    # mecabは辞書のエンコード名が曖昧にでUTF8でもutf-8でも良いがシステム辞書とユーザー辞書で完全位置していないと動かない。unidic-pyのエンコードがutf8になっているのでビルド時の引数を変えてはいけない
    build_dictionary(f"-d{unidic_path} -u {compiled_dict_path} -f utf8 -t utf8 {tmp_csv_path}")

    # システム辞書が間違った情報でコンパイルされているのでユーザー辞書も間違った情報に合わせる
    # matrix.def に記載されている left_id.def と right_id.def のサイズが逆になっているため、
    # 実行時にファイルが壊れていると判断され、コスト推定の時点でエラーとなり、辞書ビルドに失敗する
    # matrix.defはmatrix.binのソースファイル？だが容量が4gbを超えるため読み書きや同梱ができない
    # コスト推定は飛ばしてコストはハードコードしたが、辞書の互換性チェック時left_id.defとright_id.defのサイズをチェックするのでそのままだとエラーが起きる
    # なので、ユーザー辞書のバイナリにunidic-pyのleft_id.defとright_id/defのサイズを直接書き込む
    # この現象について詳しく解説している記事を見つけたのでおいておく　https://zenn.dev/zagvym/articles/28056236903369
    compiled_dict = Path(compiled_dict_path).read_bytes()
    compiled_dict = compiled_dict[:16] + b"\x0a\x3d\00\x00\x1c\x3c\x00" + compiled_dict[23:]
    Path(compiled_dict_path).write_bytes(compiled_dict)
