# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python 環境セットアップ

このプロジェクトでは `.venv` 仮想環境を使用することを前提としています：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

以降のコマンドは全て仮想環境内で実行するか、 `.venv/bin/python` を使用してください。

## 開発コマンド

### 初期化・セットアップ
```bash
.venv/bin/python initialize.py  # 必要なモデルとデフォルト TTS モデルのダウンロード
```

### テスト実行
```bash
# PyTorchでの推論テスト
hatch run test:test  # CPU 推論テスト
hatch run test:test-cuda  # CUDA 推論テスト

# ONNX推論テスト  
hatch run test-onnx:test  # CPU 推論テスト（ONNX）
hatch run test-onnx:test-cuda  # CUDA 推論テスト（ONNX）
hatch run test-onnx:test-directml  # DirectML 推論テスト

# 正規化テスト
hatch run test-normalizer:test
```

### リンティング・フォーマット
```bash
hatch run style:check  # ruff でのコードチェック
hatch run style:fmt   # ruff でのフォーマット
```

### 学習・推論
```bash
# 学習用スクリプト（基本的にはメンテナンスしていない）
.venv/bin/python train_ms.py  # マルチスピーカー学習
.venv/bin/python train_ms_jp_extra.py  # JP-Extra 学習

# 推論
.venv/bin/python app.py  # WebUI 起動（旧版）
.venv/bin/python server_editor.py --inbrowser  # エディター起動
.venv/bin/python server_fastapi.py  # API サーバー起動

# 前処理
.venv/bin/python preprocess_all.py  # 全前処理実行
.venv/bin/python bert_gen.py  # BERT特徴生成
.venv/bin/python style_gen.py  # スタイルベクトル生成
```

## プロジェクト構造

### アーキテクチャ概要
Style-Bert-VITS2 は、Bert-VITS2 をベースとした日本語対応の音声合成システム。感情や発話スタイルを制御可能な音声合成を実現。

**このフォークの特徴**：
- 推論コードのメンテナンスと性能改善が主な焦点
- 学習コードは基本的に手を加えていない
- JP-Extra モデル（`models_jp_extra.py`）が現在のメイン
- 従来の多言語対応モデル（`models.py`）も並存

### 主要コンポーネント

**style_bert_vits2/**: 推論専用ライブラリ（リファクタリング済み）
- `tts_model.py`: TTS モデルのエントリーポイント
- `models/`: 音声合成モデル
  - `models_jp_extra.py`: **現在のメインモデル**（JP-Extra 版）
  - `models.py`: 従来の多言語対応モデル
  - `infer.py`, `infer_onnx.py`: 推論エンジン
- `nlp/`: 言語処理（日本語、中国語、英語対応）
- `voice.py`: 音声調整機能

**学習・前処理スクリプト**（ルートディレクトリ）:
- `train_ms.py`, `train_ms_jp_extra.py`: 学習スクリプト
- `preprocess_*.py`: データ前処理
- `bert_gen.py`: BERT 特徴抽出
- `style_gen.py`: スタイルベクトル生成

**WebUI・API**:
- `app.py`: 旧版 WebUI
- `server_editor.py`: エディター機能
- `server_fastapi.py`: RESTful API

### 依存関係の構造

**pyproject.toml** の依存関係は推論部分のみを定義：
- 基本依存関係：推論に必要な最小限のライブラリ
- `torch` オプション：PyTorch推論に必要
- `onnxruntime` オプション：ONNX推論に必要（Torch依存なし）

**推論方式の選択**：
- ONNX推論：Torch依存を除去可能、軽量
- PyTorch推論：別途Torchのインストールが必要（`torch` オプションを有効にする）

### 設定ファイル
- `config.yml`: メイン設定ファイル（学習、推論、サーバー設定）
- `pyproject.toml`: Python 環境設定、依存関係、テスト設定
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

## 開発時の注意点

### フォークの方針
- **推論コードの最適化がメイン**：メモリ効率、速度改善
- **学習コードは基本ノーメンテ**：必要最小限の修正のみ
- **JP-Extra が主軸**：日本語特化の改良版を重視

### 言語対応
日本語、中国語、英語の多言語対応。各言語ごとに異なる BERT モデルと前処理パイプラインを使用。

### モデル形式
- safetensors 形式を推奨（PyTorch の pth からの移行）
- ONNX 推論もサポート（`infer_onnx.py`）

### GPU・CPU対応
学習には CUDA 必須。推論は CPU でも可能（`--device cpu`）。

### 設定管理
`config.yml` で各コンポーネントの設定を一元管理。デバイス、ポート、パスなどはここで調整。
このフォークでは積極的に維持されておらず、またライブラリコンポーネントからも参照されない。

### テスト環境
hatch を使用した環境管理。PyTorch と ONNX、異なるデバイス（CPU/CUDA/DirectML）でのテストをサポート。

## 重要な注意事項（命令）
- 不明点がある場合は、作業開始前に必ず確認を取ってください。
- 重要な判断が必要な場合は、その都度報告し、承認を得てください。
- 予期せぬ問題が発生した場合は、即座に報告し、対応策を提案してください。
- **明示的に指示されていない変更は行わないでください。** 必要と思われる変更がある場合は、まず提案として報告し、承認を得てから実施してください。
- **技術スタックに記載のバージョン（APIやフレームワーク、ライブラリ等）を勝手に変更しないでください。** 変更が必要な場合は、その理由を明確にして承認を得るまでは変更を行わないでください。
- コード出力内のログ出力メッセージは「必ず」英語で記述してください。それ以外のレスポンスは常に日本語で回答してください。
- **コード中の既存のコメントは除去する必要がない場合は絶対に除去してはならず、可能な限り既存のコメントを保持します。** 新規にコードを書く箇所では詳細にコメントを記載して、後から読みやすいようにしてください。ただし、見ればすぐわかるような過度に冗長なコメントや前回からの差分のみを伝えるコメントは書きません。
- 基本的にコードの削除はコメントアウトではなくコード自体を削除して、ゴミコードが残らないようにすべきです。
- コーディングルール・コーディングスタイルは周辺のコードの雰囲気を読み取った上で適切に合わせる必要があります。
- HTML を除き基本シングルクオートを使っています。ダブルクオートは Python の docstring 以外使いません。ただし、周辺のコードがダブルクオートを使っている場合はダブルクオートを使います。
- 関数への引数・配列・辞書が複数行に跨る場合、末尾のカンマ (ケツカンマ) を必ず付与します。
- Python においては、なるべく関数の引数と戻り値に厳密な Type Hint を付与します。Python 3.10 以降を使っているので、from typing から Dict, List, Tuple などをインポートするのではなく、直接 list, dict, tuple（全て小文字）などのビルドイン型を Type Hint に指定します。typing.Dict/List/Tuple などのモジュールは絶対にインポートしません。
- Python においては、例外をログ出力する際に {e} と埋め込むとエラー詳細がわからなくなるため、logging モジュールなど exc_info を渡せるものであれば exc_info=ex (e ではなく ex という変数名を好みます) として渡し、print() 以外のロギング手段がないプロジェクトでは traceback.print_exc() を使うように心がけます。
- Python においては、ファイルシステム操作に関して、os.path など os 以下の関数はなるべく使用を避け、代わりによりモダンな pathlib を利用します。
- Python においては、クラス・メソッド・関数の docstring の下に1行空白行を空けます。
- コメントやログに出力する文字列に関して、可読性を高めるため、英単語と日本語の間には必ず半角スペースを一つ入れます。
  - 例: 'では Apple の iPhone はどうですか？'
- 英語では「This becomes:」のように文の末尾にコロンを使うシチュエーションでも、「これはこのようになります。」のように句点で文末を締めます。日本語では文末がコロンで終わる文はあまり一般的ではないためです。
- 問題が発生した時は、必ず Web で類似の問題や解決策がないかを積極的に検索し、最新の情報をもとに修正を行います。
- Python では、循環インポートの回避が必要な場合を除き、関数やメソッド内でのインポートを行いません。多くの場合、関数やメソッド内でのインポートは冗長なためです。
