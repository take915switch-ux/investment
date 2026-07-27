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
}

FX_TICKER = "JPY=X"


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


def annualized_statistics(close: pd.Series, periods_per_year: int) -> tuple[float | None, float | None]:
    """価格系列から算術平均ベースの年平均リターンと年率リスクを返す。"""
    returns = close.pct_change(fill_method=None).dropna()
    if returns.empty:
        return None, None

    annual_return = float(returns.mean() * periods_per_year)
    annual_risk = float(returns.std() * (periods_per_year**0.5))
    return annual_return, annual_risk


def convert_to_jpy(usd_close: pd.Series, usd_jpy: pd.Series, name: str) -> pd.Series:
    """米ドル建て価格を、同日のUSD/JPYを用いて円換算する。"""
    aligned = pd.concat(
        [usd_close.rename("usd_price"), usd_jpy.rename("usd_jpy")],
        axis=1,
    ).sort_index()

    # 為替が一部の日に欠ける場合は直前の値で補い、資産価格がある日のみ残す。
    aligned["usd_jpy"] = aligned["usd_jpy"].ffill()
    aligned = aligned.loc[usd_close.index].dropna(subset=["usd_price", "usd_jpy"])

    return (aligned["usd_price"] * aligned["usd_jpy"]).rename(name)


st.title("市場データ取得テスト")
st.caption("Yahoo Finance から取得した各資産を、円建てまたはドル建てで比較します。")

st.sidebar.header("取得条件")
selected_names = st.sidebar.multiselect(
    "取得対象",
    list(ASSETS.keys()),
    default=list(ASSETS.keys()),
)

currency = st.sidebar.radio(
    "表示通貨",
    ["円建て", "ドル建て"],
    index=0,
    horizontal=True,
)

frequency = st.sidebar.radio("頻度", ["日足", "月足"], horizontal=True)
interval = "1d" if frequency == "日足" else "1mo"
periods_per_year = 252 if frequency == "日足" else 12

default_start = date(2008, 3, 28)

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

    successful = {}
    result_errors = {}
    original_missing_counts = {}
    usd_jpy = None

    spinner_text = (
        "データを取得して円換算しています…"
        if currency == "円建て"
        else "ドル建てデータを取得しています…"
    )

    with st.spinner(spinner_text):
        if currency == "円建て":
            try:
                fx_df = fetch_one(FX_TICKER, start_date, end_date, interval)
                if fx_df.empty or "Close" not in fx_df.columns:
                    raise ValueError("USD/JPYの為替データを取得できませんでした。")
                usd_jpy = fx_df["Close"].dropna()
                if usd_jpy.empty:
                    raise ValueError("USD/JPYの終値データがありません。")
            except Exception as exc:
                st.error(f"円換算に必要な為替データの取得に失敗しました: {exc}")
                st.stop()

        for name in selected_names:
            ticker = ASSETS[name]
            try:
                df = fetch_one(ticker, start_date, end_date, interval)
                if df.empty or "Close" not in df.columns:
                    result_errors[name] = ("データなし", "")
                    continue

                usd_close = df["Close"].dropna()
                if usd_close.empty:
                    raise ValueError("終値データがありません。")

                if currency == "円建て":
                    price_series = convert_to_jpy(usd_close, usd_jpy, name)
                    if price_series.empty:
                        raise ValueError("円換算後の価格データがありません。")
                else:
                    price_series = usd_close.rename(name)

                successful[name] = price_series
                original_missing_counts[name] = int(df["Close"].isna().sum())
            except Exception as exc:
                result_errors[name] = ("失敗", str(exc))

    currency_unit = "円" if currency == "円建て" else "ドル"
    latest_value_column = f"最新値（{currency_unit}）"

    results = []
    for name in selected_names:
        ticker = ASSETS[name]
        if name not in successful:
            result, error = result_errors.get(name, ("データなし", ""))
            results.append(
                {
                    "対象": name,
                    "ティッカー": ticker,
                    "結果": result,
                    "件数": 0,
                    "開始": None,
                    "終了": None,
                    latest_value_column: None,
                    "欠損数": None,
                    "年平均リターン": None,
                    "年率リスク": None,
                    "エラー": error,
                }
            )
            continue

        close = successful[name]
        annual_return, annual_risk = annualized_statistics(close, periods_per_year)
        results.append(
            {
                "対象": name,
                "ティッカー": ticker,
                "結果": "成功",
                "件数": len(close),
                "開始": close.index.min(),
                "終了": close.index.max(),
                latest_value_column: round(float(close.iloc[-1]), 2),
                "欠損数": original_missing_counts.get(name, 0),
                "年平均リターン": (
                    f"{annual_return:.2%}" if annual_return is not None else None
                ),
                "年率リスク": f"{annual_risk:.2%}" if annual_risk is not None else None,
                "エラー": "",
            }
        )

    st.subheader(f"取得結果一覧（{currency}）")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    if successful:
        prices = pd.concat(successful.values(), axis=1).sort_index()

        # 各アセットについて、設定期間内の最初の有効値を100として指数化する。
        indexed_prices = prices.apply(
            lambda series: series / series.dropna().iloc[0] * 100
            if not series.dropna().empty
            else series
        )
        indexed_prices.index.name = "Date"

        chart_data = indexed_prices.reset_index().melt(
            id_vars="Date",
            var_name="対象",
            value_name="指数",
        )
        fig = px.line(
            chart_data,
            x="Date",
            y="指数",
            color="対象",
            title=f"各アセットの{currency}推移（設定期間の初日＝100・{frequency}）",
            labels={"Date": "日付", "指数": "初日を100とした指数", "対象": "アセット"},
        )
        fig.add_hline(y=100, line_dash="dash")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        csv = prices.to_csv().encode("utf-8-sig")
        currency_code = "jpy" if currency == "円建て" else "usd"
        st.download_button(
            f"{currency}データをCSVでダウンロード",
            data=csv,
            file_name=f"market_prices_{currency_code}.csv",
            mime="text/csv",
        )
else:
    st.info("左側で取得条件を選び、「データ取得」を押してください。")