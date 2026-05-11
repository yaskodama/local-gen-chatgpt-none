# Tiny GA Text Generator

小さな文字レベル生成AIを短時間だけ学習し、遺伝的アルゴリズムでモデル設定を進化させる最小構成のPythonプロジェクトです。

## 構成

- `genome.py`: モデル設計図の遺伝子定義、交叉、突然変異
- `model.py`: 小型LSTM/Transformer文字生成モデル
- `train.py`: 短時間学習処理
- `evaluate.py`: 生成文と損失からスコアを計算
- `ga.py`: 選抜、交叉、突然変異、保存処理
- `main.py`: GA全体の実行
- `generate.py`: 保存済み最良モデルから文章生成
- `data/train.txt`: サンプル学習テキスト
- `data/valid.txt`: 未学習評価用テキスト
- `results/`: ログ、チェックポイント、最良個体の保存先

## セットアップ

Python 3.10以上を使います。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GAを実行

CPUで動く小さな設定です。

```bash
python main.py
```

さらに短く試す場合:

```bash
python main.py --generations 1 --population 3 --steps 5
```

前回の最良個体から進化を続ける場合:

```bash
python main.py --resume --generations 3 --population 6 --steps 35
```

`data/valid.txt` がある場合、進化計算は学習データではなく検証データのlossを優先して評価します。
最良個体には `valid_ppl`, `valid_bits_per_char`, `valid_bits_per_byte` も表示されます。

実行後、以下が保存されます。

- `results/log.jsonl`
- `results/best_model.pt`
- `results/best_genome.json`
- `results/best_sample.txt`
- `results/best_valid_model.pt`
- `results/best_valid_genome.json`

## 文章生成

```bash
python generate.py --prompt "春の朝" --length 200
```

英語プロンプトの例:

```bash
python generate.py --prompt "In a quiet" --length 200 --temperature 0.9
```

## 調整例

```bash
python main.py --generations 5 --population 8 --steps 50
```

`--steps`を増やすと各個体の学習時間が伸びます。CPUでは小さめの値から試してください。
`--mutation-rate`を上げると探索が広がり、下げると現在の最良個体の周辺を細かく探索します。
