# ROGII — 坑井地質予測：データ説明

- 原文: [Data | ROGII - Wellbore Geology Prediction | Kaggle](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/data)
- 原文確認日: 2026年7月19日
- ライセンス: コンペティション規則に従う

## データセットの説明

コンペティションデータは、地質予測に使用する水平坑井の軌跡と、垂直参照ログ（Typewell）で構成されています。目的は、各水平坑井の評価区間におけるTVT（True Vertical Thickness、真垂直層厚）を予測することです。

データは `train/` と `test/` ディレクトリに分けられています。各坑井は、固有の8文字のハッシュ（例: `015fe0d2`）によって識別されます。

## ファイルとフィールド

### `train/`

学習データが格納されています。各坑井には3つのファイルがあります。

#### `{WELLNAME}__horizontal_well.csv`

坑井軌跡、地質境界面、検層データが含まれます。

| フィールド | 説明 |
|---|---|
| `WELLNAME` | 坑井の固有識別子 |
| `MD` | Measured Depth（測定深度、ft）。地表から坑井に沿って測った坑井軌跡の全長 |
| `X` | Easting（東方向座標、ft）。水平面上の空間座標 |
| `Y` | Northing（北方向座標、ft）。水平面上の空間座標 |
| `Z` | True Vertical Depth（真垂直深度、ft）。海面下方向の垂直距離 |
| `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` | 各地層境界の予測深度。学習データにのみ存在 |
| `TVT` | True Vertical Thickness（真垂直層厚、ft）。水平坑井の水平区間について1 ftごとに人手で解釈された地質学的位置。目的変数であり、学習データにのみ存在 |
| `GR` | Gamma Ray（ガンマ線、API単位）。岩石の自然放射能を測定する検層値 |
| `TVT_input` | Input Target（入力目的値、ft）。特徴量として提供される `TVT` のコピー。評価区間では `NaN` になっている |

#### `{WELLNAME}__typewell.csv`

地質対比に使用する垂直参照ログです。

| フィールド | 説明 |
|---|---|
| `TVT` | Vertical Depth Index（垂直深度インデックス、ft）。垂直ログの主要な深度基準。対応する水平坑井の `TVT`（地質学的位置）に相当 |
| `GR` | Gamma Ray（ガンマ線、API単位）。対比に使用する垂直方向のガンマ線シグネチャ |
| `Geology` | Formation Label（地層ラベル）。地質単元を示すカテゴリラベル（例: `EGFDL`, `BUDA`） |

#### `{WELLNAME}.png`

坑井軌跡と地質断面の可視化画像です。

### `test/`

約200坑井の評価データが格納されます。各坑井には2つのファイルがあります。

#### `{WELLNAME}__horizontal_well.csv`

坑井軌跡と検層データが含まれます。評価区間の目的変数 `TVT` は非公開で、`NaN` に置き換えられています。

#### `{WELLNAME}__typewell.csv`

テスト坑井の垂直参照ログです。

> Kaggle上で確認できる `test/` フォルダには、提出コードの作成を支援するサンプルデータとして、学習セットから抽出した少数の例だけが含まれています。提出Notebookが非公開テストセット上で再実行される際、これらのファイルは実際のテストデータに置き換えられます。

### `sample_submission.csv`

正しい形式のサンプル提出ファイルです。

| フィールド | 説明 |
|---|---|
| `id` | 各予測点の固有識別子。`{WELLNAME}_{row_index}` 形式（例: `015fe0d2_1654`） |
| `tvt` | 予測したTrue Vertical Thickness（真垂直層厚、ft） |

## データセット概要

- ファイル数: 2,327
- 合計サイズ: 1.33 GB
- ファイル形式: CSV、PNG、PPTX
- 付属資料: `AI_wellbore_geology_prediction_task_en.pptx`（28.79 MB）
- ライセンス: コンペティション規則に従う

## データへのアクセス

データを閲覧・ダウンロードするには、コンペティション規則への同意が必要です。Kaggleへサインインまたは登録したうえで、規則を承諾してください。

---

この文書はKaggleのDataページを日本語訳したものです。正式なフィールド定義、利用条件、ファイル構成については、必ず原文の最新内容を確認してください。
