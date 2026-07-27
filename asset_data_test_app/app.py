from datetime import date, timedelta
from itertools import combinations

import numpy as np
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


def annualized_statistics(
    close: pd.Series, periods_per_year: int
) -> tuple[float | None, float | None]:
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


def minimum_variance_portfolio(
    returns: pd.DataFrame, periods_per_year: int
) -> tuple[pd.Series, float, float] | None:
    """空売りなし・合計100%の制約で最小分散ポートフォリオを求める。"""
    clean_returns = returns.dropna(how="any")
    if len(clean_returns) < 2 or clean_returns.shape[1] < 2:
        return None

    covariance = clean_returns.cov() * periods_per_year
    expected_returns = clean_returns.mean() * periods_per_year
    asset_names = list(covariance.columns)

    best_weights = None
    best_variance = np.inf
    tolerance = 1e-10

    # 最適解で比率が0になる資産にも対応するため、全ての資産部分集合を調べる。
    for subset_size in range(1, len(asset_names) + 1):
        for subset in combinations(range(len(asset_names)), subset_size):
            subset_covariance = covariance.iloc[list(subset), list(subset)].to_numpy()
            ones = np.ones(subset_size)
            inverse_covariance = np.linalg.pinv(subset_covariance)
            denominator = float(ones @ inverse_covariance @ ones)

            if denominator <= tolerance:
                continue

            subset_weights = inverse_covariance @ ones / denominator
            if np.any(subset_weights < -tolerance):
                continue

            subset_weights = np.clip(subset_weights, 0.0, None)
            subset_weights = subset_weights / subset_weights.sum()
            variance = float(subset_weights @ subset_covariance @ subset_weights)

            if variance < best_variance:
                full_weights = np.zeros(len(asset_names))
                full_weights[list(subset)] = subset_weights
                best_weights = full_weights
                best_variance = variance

    if best_weights is None:
        return None

    weights = pd.Series(best_weights, index=asset_names, name="配分比率")
    portfolio_return = float(weights @ expected_returns)
    portfolio_risk = float(np.sqrt(max(best_variance, 0.0)))
    return weights, portfolio_return, portfolio_risk


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

rolling_years = st.sidebar.selectbox(
    "相関係数の移動期間",
    [1, 3, 5],
    index=1,
    format_func=lambda years: f"{years}年",
)
rolling_window = periods_per_year * rolling_years

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

        if len(successful) >= 2:
            # 価格水準ではなく、各期間の騰落率同士の相関係数を計算する。
            returns = prices.pct_change(fill_method=None)
            correlation = returns.corr().round(2)

            st.subheader(f"アセット間の相関係数（{currency}・{frequency}リターン）")
            st.caption(
                "各セルは、設定期間内の各アセットの騰落率について計算したPearsonの相関係数です。"
            )
            st.dataframe(
                correlation.style.format("{:.2f}").background_gradient(
                    cmap="RdBu_r", vmin=-1, vmax=1
                ),
                use_container_width=True,
            )

            rolling_series = {}
            for asset_a, asset_b in combinations(successful.keys(), 2):
                pair_returns = returns[[asset_a, asset_b]].dropna()
                if len(pair_returns) < rolling_window:
                    continue
                rolling_series[f"{asset_a} × {asset_b}"] = (
                    pair_returns[asset_a]
                    .rolling(window=rolling_window, min_periods=rolling_window)
                    .corr(pair_returns[asset_b])
                )

            if rolling_series:
                rolling_corr = pd.concat(rolling_series, axis=1)
                rolling_corr.index.name = "Date"
                rolling_chart_data = rolling_corr.reset_index().melt(
                    id_vars="Date",
                    var_name="組み合わせ",
                    value_name="相関係数",
                ).dropna(subset=["相関係数"])

                rolling_fig = px.line(
                    rolling_chart_data,
                    x="Date",
                    y="相関係数",
                    color="組み合わせ",
                    title=f"相関係数の推移（{rolling_years}年移動相関・{currency}・{frequency}リターン）",
                    labels={
                        "Date": "日付",
                        "相関係数": "相関係数",
                        "組み合わせ": "アセットの組み合わせ",
                    },
                )
                rolling_fig.add_hline(y=0, line_dash="dash")
                rolling_fig.update_yaxes(range=[-1, 1])
                rolling_fig.update_layout(hovermode="x unified")

                st.subheader("相関係数の推移")
                st.caption(
                    f"直近{rolling_years}年分のリターンを使って、各アセットの組み合わせごとの相関係数を計算しています。"
                )
                st.plotly_chart(rolling_fig, use_container_width=True)
            else:
                st.info(
                    f"{rolling_years}年移動相関を計算するには、より長い取得期間が必要です。"
                )

            minimum_variance = minimum_variance_portfolio(returns, periods_per_year)
            st.subheader("最小分散ポートフォリオ")
            st.caption(
                "設定期間のリターンから推定した共分散行列を使い、空売りなし・配分合計100%の条件で年率リスクが最小になる配分を計算します。"
            )

            if minimum_variance is not None:
                weights, portfolio_return, portfolio_risk = minimum_variance
                allocation = (
                    weights.rename("配分比率")
                    .mul(100)
                    .reset_index()
                    .rename(columns={"index": "アセット"})
                )
                allocation["配分比率"] = allocation["配分比率"].round(2)

                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("推定年平均リターン", f"{portfolio_return:.2%}")
                metric_col2.metric("推定年率リスク", f"{portfolio_risk:.2%}")

                table_col, chart_col = st.columns([1, 1])
                with table_col:
                    st.dataframe(
                        allocation.style.format({"配分比率": "{:.2f}%"}),
                        use_container_width=True,
                        hide_index=True,
                    )
                with chart_col:
                    positive_allocation = allocation[allocation["配分比率"] > 0]
                    allocation_fig = px.pie(
                        positive_allocation,
                        names="アセット",
                        values="配分比率",
                        title="最小分散ポートフォリオの配分",
                        hole=0.35,
                    )
                    allocation_fig.update_traces(textposition="inside", textinfo="percent+label")
                    st.plotly_chart(allocation_fig, use_container_width=True)

                st.caption(
                    "推定値は選択した期間・頻度・表示通貨に依存します。将来のリターンやリスクを保証するものではありません。"
                )
            else:
                st.info("最小分散ポートフォリオの計算に必要な共通データが不足しています。")
        else:
            st.info("相関係数と最小分散ポートフォリオを表示するには、2つ以上のアセットを選択してください。")

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