# Handoff Notes

このプロジェクトは、小さな文字レベル生成AIを遺伝的アルゴリズムで進化させる実験環境です。
モデルは `LSTM` または小型 `Transformer` で、遺伝子としてモデル構成・学習率・バッチサイズ・系列長などを探索します。

## 作業ディレクトリ

```bash
cd /Users/yaskodama/local-genai-chatgpt
```

## Python環境

仮想環境は作成済みです。

```bash
source .venv/bin/activate
python --version
```

依存関係:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
torch>=2.2
numpy>=1.26
```

## 主要ファイル

- `main.py`: 進化計算CLI
- `ga.py`: 遺伝的アルゴリズム本体、評価、保存
- `genome.py`: 遺伝子定義、ランダム生成、交叉、突然変異
- `model.py`: `CharLSTM` / `CharTransformer` / 文章生成
- `train.py`: 学習、validation loss、PPL/BPB計算
- `evaluate.py`: 進化スコア計算
- `generate.py`: 保存済みモデルから文章生成
- `data/train.txt`: 学習用テキスト
- `data/valid.txt`: 未学習評価用テキスト
- `README.md`: 基本説明
- `HANDOFF.md`: このファイル

## 現在のデータ量

```text
data/train.txt: 4519 bytes
data/valid.txt: 2852 bytes
total: 7371 bytes
```

以前よりデータを増やしてあり、図書館、物語、文章生成、PPL/BPB、学習/検証に関する日本語/英語テキストが入っています。

## 現在の進化状況

`results/log.jsonl`:

```text
rows: 674
last_generation: 68
```

### PPL最良

```text
generation: 47
rank: 0
valid_loss: 2.766759514058911
valid_ppl: 15.907003979020228
valid_bits_per_char: 3.991590230265149
valid_bits_per_byte: 2.7091590864522215
genome:
  model_type: lstm
  embed_dim: 48
  hidden_dim: 64
  num_layers: 1
  num_heads: 4
  dropout: 0.0
  learning_rate: 0.002
  batch_size: 16
  seq_length: 64
```

### BPB最良

現在の `results/best_valid_model.pt` はこの系統です。

```text
generation: 64
rank: 4
valid_loss: 2.9066006495402408
valid_ppl: 18.294503317050882
valid_bits_per_char: 4.193338342936346
valid_bits_per_byte: 2.561069909785795
genome:
  model_type: lstm
  embed_dim: 64
  hidden_dim: 64
  num_layers: 1
  num_heads: 1
  dropout: 0.2
  learning_rate: 0.002
  batch_size: 16
  seq_length: 64
```

## 再起動後にまず確認するコマンド

```bash
cd /Users/yaskodama/local-genai-chatgpt
source .venv/bin/activate
.venv/bin/python -m py_compile genome.py model.py train.py evaluate.py ga.py generate.py main.py
```

ログと現状確認:

```bash
wc -l results/log.jsonl
wc -c data/train.txt data/valid.txt
sed -n '1,120p' results/best_valid_genome.json
```

## 進化計算を続けるコマンド

通常の継続:

```bash
.venv/bin/python main.py --resume --generations 5 --population 10 --elite 4 --steps 260 --mutation-rate 0.15
```

より重めに進める場合:

```bash
.venv/bin/python main.py --resume --generations 6 --population 12 --elite 4 --steps 320 --mutation-rate 0.12
```

`--resume` は `results/best_valid_genome.json` を優先して初期個体に使います。

## 文章生成

BPB最良系モデルで生成:

```bash
.venv/bin/python generate.py \
  --checkpoint results/best_valid_model.pt \
  --prompt "春の朝、町の小さな図書館" \
  --length 220 \
  --temperature 0.7
```

総合スコア最良モデルで生成:

```bash
.venv/bin/python generate.py \
  --checkpoint results/best_model.pt \
  --prompt "春の朝、町の小さな図書館" \
  --length 220 \
  --temperature 0.7
```

## 重要な成果物

- `results/best_model.pt`: 総合スコア最良モデル
- `results/best_genome.json`: 総合スコア最良の遺伝子
- `results/best_sample.txt`: 総合スコア最良の生成サンプル
- `results/best_valid_model.pt`: validation基準の最良モデル
- `results/best_valid_genome.json`: validation基準の最良遺伝子
- `results/best_valid_sample.txt`: validation基準の生成サンプル
- `results/log.jsonl`: 全個体の評価ログ
- `results/evolution_visualization.html`: 遺伝子変化、交叉/突然変異推定、指標推移のHTML可視化
- `results/input_output_examples_20.txt`: 現在モデルの入力/出力例20件
- `results/sample_responses_20.txt`: サンプル入力に対する20パターン生成

## HTML可視化

ブラウザで以下を開くと、これまでの進化の流れを確認できます。

```text
results/evolution_visualization.html
```

注意: 現在のログには親IDや実際の突然変異箇所は保存されていません。
そのため、HTML内の交叉・突然変異は前世代上位個体との遺伝子差分からの推定です。

## 次に改善するなら

- `results/log.jsonl` に親ID、交叉元、突然変異した遺伝子名を保存する
- `best_valid_model.pt` をPPL最良とBPB最良で別ファイルに分ける
- 学習データと検証データをさらに増やす
- validation lossだけでなく、生成文の反復率や日本語/英語の崩れを別指標として可視化する
- 長時間実行する場合は `steps` と `population` を上げる

## 注意点

このプロジェクトは実験用の小型文字レベル生成AIです。
ChatGPTのような大規模LLMとは構造・データ量・計算量がまったく違います。
現在のPPL/BPBは、この小さなローカルデータセット上での比較指標として扱ってください。
