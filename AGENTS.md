# AGENTS.md

## プロジェクト固有の注意事項

- **実装が完了したら、『必ず』`uv run task lint; uv run task format` を実行し、ruff と pyright によるコードチェック、コードフォーマットを完了させてください。また、エラーが発生していたら適宜修正してください。**
- **Pyright による型チェックを導入している。新規で実装した箇所には必ず Type Hint を付与すること。**  
- **NDArray の型は from numpy.typing import NDArray を使って定義すること。np.ndarray を使わない。**
- **一回しか使わない上に処理内容が薄い helper 関数やローカル関数は、認知負荷を増やすので原則として追加しないこと。** その場で十分読める処理は inline で記述し、どうしても切り出す必然性がある場合に限って関数化する。  
- **重要な処理行の直前には、その役割を端的に検索できる一行コメントを置くこと。** コメントは人間用の高速 grep index でもあるため、特に変換・分岐・例外扱いの行は「何をしている行か」が上の 1 行で分かる形にする。  
- 学習コードに関して、**モデル固有の処理ブロック以外は、コメントや空行も含めてできる限り各ファイル間で一致させる。**  
- 差異が見つかった場合は、型安全性やエラー処理の観点でより良い実装を採用し、他のファイルにも反映する。  
- 再現性や挙動差に関わる判断を行った場合は、その意図をコード近傍のコメントとして残す。
- **research/ 配下は別の Git リポジトリで、pyproject.toml / uv の依存関係も本体とは別に用意されているので、research/ 配下のコードを実行する際は research/ に移動してから実行すること。**
- **SBV2 / Nanairo / アノテーション / 前処理 / preference model に関連する設計判断や過去の知見を調べる際は、外部の GitHub や WebSearch より先に research/ 配下、特に research/AGENTS.md の索引を読み尽くすこと。** research/ には Aivis Project の研究知見・設計判断・実験ログが蓄積されており、大半のトピックは research/AGENTS.md の「トピック別の参照すべき資料」から辿れる。存在に気づかず外部リポジトリだけを調べてしまうミスが過去にあったので注意する。
- **style_bert_vits2/nlp/japanese/normalizer.py の normalize_text() は関数の実行順序に強く依存しているので、正規化ルールを追加・変更する際は順序を無闇に入れ替えないこと。** 度単位正規化 (__DEGREE_UNIT_PATTERN) のような記号の畳み込みは、__replace_symbols() 内で NFKC 正規化・英単語のカタカナ変換より前に行う必要がある。順序を誤ると、℃ が NFKC で °C に分解されて C が英単語としてカタカナ変換されてしまったり、単位記号を正しく畳み込めなくなったりする。
- **normalizer.py に新しい単位・記号の正規化ルールを追加する際は、代表的な1パターンだけでなく周辺のバリエーションまで含めた回帰テストを tests/test_normalizer.py に追加すること。** 度単位正規化のテストでは、摂氏だけでなく華氏・裸の度記号・符号付き表記・全角文字・for_irodori=True 経路までカバーしている。この網羅の仕方を新しい正規化ルールにも踏襲する。

## Python 環境セットアップ

このプロジェクトでは uv を使用した環境管理を行っています。

### 基本セットアップ（開発用）

```bash
# core グループがデフォルトでインストールされる
# core グループには PyTorch・ONNX Runtime・開発ツールが含まれる
uv sync
```

### ユースケース別セットアップ

```bash
# ONNX 変換ツールを使う場合
uv sync --group tools

# FastAPI サーバー(server_fastapi.py, server_editor.py)を使う場合
uv sync --group server

# Gradio WebUI (app.py) を使う場合
uv sync --group webui

# 学習・前処理を行う場合
uv sync --group train

# 全部入り
uv sync --group full

# 複数グループの組み合わせ（例: WebUI + サーバー + 学習）
uv sync --group webui --group server --group train
```

### 依存関係グループの構造

各グループは `core` を include しているため、どのグループを選んでも core は含まれます。

| グループ | 用途 | 追加される依存 |
|---------|------|---------------|
| core | 基盤（デフォルト） | torch, torchaudio, onnxruntime, ruff, pyright, pytest 等 |
| tools | ONNX 変換 | onnx, onnxsim-prebuilt, onnxconverter-common |
| server | FastAPI サーバー | GPUtil, psutil, requests |
| webui | Gradio WebUI | gradio, umap-learn |
| train | 学習・前処理 | pyannote.audio, tensorboard, librosa, pyloudnorm |
| full | 全部入り | 上記すべて |

### torch/onnxruntime のバリアント

`pyproject.toml` の `tool.uv.sources` で OS/アーキテクチャ別に自動選択されます。
- **Windows x64・Linux x64**: PyTorch CUDA 12.6 版
- **macOS・その他**: PyPI から CPU 版

## 開発コマンド

### 初期化・セットアップ

```bash
uv run python initialize.py  # 必要なモデルとデフォルト TTS モデルのダウンロード
```

### リンティング・フォーマット（コード変更時は必ず実行すること）

```bash
uv run task lint  # ruff + pyright でのコードチェック
uv run task format    # ruff でのフォーマット
uv run task typecheck # pyright での型チェックのみ
```

### テスト実行

```bash
# 正規化テスト
uv run task test-g2p
uv run task test-normalizer

# PyTorch 推論テスト
uv run task test       # CPU 推論テスト
uv run task test-cuda  # CUDA 推論テスト

# ONNX 推論テスト
uv run task test-onnx           # CPU 推論テスト（ONNX）
uv run task test-onnx-cuda      # CUDA 推論テスト（ONNX）
uv run task test-onnx-directml  # DirectML 推論テスト
uv run task test-onnx-tensorrt  # TensorRT 推論テスト
uv run task test-onnx-coreml    # CoreML 推論テスト
```

### 学習・推論

```bash
# 学習用スクリプト
uv run python train_ms.py           # 多言語版学習
uv run python train_ms_jp_extra.py  # JP-Extra 学習

# 推論
uv run python app.py             # WebUI 起動
uv run python server_editor.py   # エディター起動
uv run python server_fastapi.py  # API サーバー起動

# 前処理
uv run python preprocess_all.py  # 全前処理実行
uv run python bert_gen.py        # BERT 特徴生成
uv run python style_gen.py       # スタイルベクトル生成
```

## プロジェクト構造

### アーキテクチャ概要

Style-Bert-VITS2 は、Bert-VITS2 をベースとした日本語対応の音声合成システムです。感情や発話スタイルを制御可能な音声合成を実現しています。

**このフォークの特徴**:
- 推論コードのメンテナンスと性能改善が主な焦点
- 学習コードにも多少手を加えているが、互換性がなくなるレベルの大きな変更はしておらず互換性重視
- JP-Extra モデル（`models_jp_extra.py`）が現在のメイン
- 従来の多言語対応モデル（`models.py`）も並存
- このフォークは他者に使わせることを前提としていない（完全に自分用）ので、README や docs/ 以下のドキュメントは一切更新不要。実装詳細を残す場合は追加した実装の近くに残す。

### 主要コンポーネント

**style_bert_vits2/**: 推論専用ライブラリ（リファクタリング済み）
- `tts_model.py`: TTS モデルのエントリーポイント
- `models/`: 音声合成モデル
  - `models_jp_extra.py`: **現在のメインモデル**（JP-Extra 版）
  - `models.py`: 従来の多言語対応モデル
  - `infer.py`, `infer_onnx.py`: 推論エンジン
- `nlp/`: 言語処理（日本語、中国語、英語対応）
  - `nlp/japanese/user_dict/`: ユーザー辞書機能（**注意: server_editor.py 専用であり、ライブラリとして使える設計ではない**）
- `voice.py`: 音声調整機能

**学習・前処理スクリプト**（ルートディレクトリ）:
- `train_ms.py`, `train_ms_jp_extra.py`: 学習スクリプト
- `preprocess_*.py`: データ前処理
- `bert_gen.py`: BERT 特徴抽出
- `style_gen.py`: スタイルベクトル生成

**WebUI・API**:
- `app.py`: Gradio WebUI
- `server_editor.py`: エディター機能（FastAPI）
- `server_fastapi.py`: RESTful API（FastAPI）

### 依存関係の構造

**pyproject.toml** には2種類の依存関係定義があります。

**1. ライブラリ用（公開される）**:
- `[project.dependencies]`: 推論に必要な最小限のライブラリ（torch/onnxruntime は含まない）
- `[project.optional-dependencies]`: ライブラリ利用者向けのオプション
  - `torch`: PyTorch 推論に必要（バージョン/バリアントは利用者が指定）
  - `onnxruntime`: ONNX 推論に必要（OS 別にパッケージが異なる）

**2. 単体利用用（公開されない）**:
- `[dependency-groups]`: 開発・単体利用時の依存関係
  - バージョンやバリアントを固定
  - `tool.uv.sources` で torch の CUDA/CPU 版を OS 別に自動選択

**推論方式の選択**:
- ONNX 推論: torch 依存を除去可能、軽量（`infer_onnx.py`）
- PyTorch 推論: より高機能（`infer.py`）

### 設定ファイル

- `pyproject.toml`: Python 環境設定、依存関係、テスト設定、リンター設定
- `uv.lock`: 依存関係のロックファイル（uv が自動管理）
- `config.yml`: メイン設定ファイル（学習関連、このフォークでは積極的に維持されていない）
- `Data/{model_name}/config.json`: モデル固有設定

### モデル・データ構造

```
model_assets/
├── {model_name}/
│   ├── config.json
│   ├── {model}_file.safetensors
│   └── style_vectors.npy
Data/
├── {model_name}/
│   ├── raw/        # 元音声ファイル
│   ├── wavs/       # 前処理済み音声
│   ├── train.list  # 学習リスト
│   └── val.list    # 検証リスト
```
