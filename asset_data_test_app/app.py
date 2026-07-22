from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="市場データ取得テスト", layout="wide")

ASSETS = {
    "S&P 500 ETF (SPY)": "SPY",
    "金 ETF (GLD)": "GLD",
    "米国総合債券 ETF (AGG)": "AGG",
    "全世界株式 ETF (ACWI)": "ACWI",
    "USD/JPY": "JPY=X",
}


def fetch_one(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    # yfinance の end は指定日を含まないため、終了日の翌日を渡す。
    data = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if data.empty:
        return data

    # yfinance のバージョンにより列が MultiIndex になる場合に対応。
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


st.title("市場データ取得テスト")
st.caption("Yahoo Finance から各資産の価格データを取得できるか確認します。")

st.sidebar.header("取得条件")
selected_names = st.sidebar.multiselect(
    "取得対象",
    list(ASSETS.keys()),
    default=list(ASSETS.keys()),
)

frequency = st.sidebar.radio("頻度", ["日足", "月足"], horizontal=True)
interval = "1d" if frequency == "日足" else "1mo"

default_start = date.today() - timedelta(days=365 * 3)

# min_value を明示しない場合、Streamlit は初期値の10年前を下限にする。
# そのため初期値が3年前だと、選択可能な最古日は約13年前になってしまう。
start_date = st.sidebar.date_input(
    "開始日",
    value=default_start,
    min_value=date(1990, 1, 1),
    max_value=date.today(),
)
end_date = st.sidebar.date_input(
    "終了日",
    value=date.today(),
    min_value=date(1990, 1, 1),
    max_value=date.today(),
)

run = st.sidebar.button("データ取得", type="primary", use_container_width=True)

if run:
    if not selected_names:
        st.warning("取得対象を1つ以上選んでください。")
        st.stop()
    if start_date > end_date:
        st.error("開始日は終了日以前にしてください。")
        st.stop()

    results = []
    successful = {}

    with st.spinner("データを取得しています…"):
        for name in selected_names:
            ticker = ASSETS[name]
            try:
                df = fetch_one(ticker, start_date, end_date, interval)
                if df.empty or "Close" not in df.columns:
                    results.append(
                        {
                            "対象": name,
                            "ティッカー": ticker,
                            "結果": "データなし",
                            "件数": 0,
                            "開始": None,
                            "終了": None,
                            "最新値": None,
                            "欠損数": None,
                            "エラー": "",
                        }
                    )
                    continue

                close = df["Close"].dropna()
                if close.empty:
                    raise ValueError("終値データがありません。")

                successful[name] = close.rename(name)
                results.append(
                    {
                        "対象": name,
                        "ティッカー": ticker,
                        "結果": "成功",
                        "件数": len(close),
                        "開始": close.index.min(),
                        "終了": close.index.max(),
                        "最新値": float(close.iloc[-1]),
                        "欠損数": int(df["Close"].isna().sum()),
                        "エラー": "",
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "対象": name,
                        "ティッカー": ticker,
                        "結果": "失敗",
                        "件数": 0,
                        "開始": None,
                        "終了": None,
                        "最新値": None,
                        "欠損数": None,
                        "エラー": str(exc),
                    }
                )

    st.subheader("取得結果一覧")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    if successful:
        prices = pd.concat(successful.values(), axis=1).sort_index()
        chart_data = prices.reset_index().melt(
            id_vars=prices.index.name or "Date",
            var_name="対象",
            value_name="価格",
        )
        date_column = prices.index.name or "Date"
        fig = px.line(chart_data, x=date_column, y="価格", color="対象")
        st.plotly_chart(fig, use_container_width=True)

        csv = prices.to_csv().encode("utf-8-sig")
        st.download_button(
            "取得データをCSVでダウンロード",
            data=csv,
            file_name="market_prices.csv",
            mime="text/csv",
        )
else:
    st.info("左側で取得条件を選び、「データ取得」を押してください。")
