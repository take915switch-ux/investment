# 市場データ取得テストアプリ

`yfinance` を使い、以下の価格データを取得できるか確認するStreamlitアプリです。

- SPY: S&P 500 ETF
- GLD: 金ETF
- AGG: 米国総合債券ETF
- ACWI: 全世界株式ETF
- JPY=X: USD/JPY

## 起動方法

Python 3.11 または 3.12 を推奨します。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShellでは仮想環境の有効化は次です。

```powershell
.venv\Scripts\Activate.ps1
```

起動後、ブラウザで通常は `http://localhost:8501` が開きます。

## 主な機能

- 日足・月足の選択
- 取得期間の指定
- 5系列の個別・一括テスト
- 成功、データなし、失敗の判定
- 取得行数、期間、最新値、欠損数の表示
- 価格推移グラフ
- 各系列のCSVダウンロード

## 注意

`yfinance` はYahoo Financeの非公式ライブラリです。個人用の試作・検証向けとして利用してください。
