from datetime import date, timedelta
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_one(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
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
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def annualized_statistics(
    close: pd.Series, periods_per_year: int
) -> tuple[float | None, float | None]:
    returns = close.pct_change(fill_method=None).dropna()
    if returns.empty:
        return None, None
    return (
        float(returns.mean() * periods_per_year),
        float(returns.std() * np.sqrt(periods_per_year)),
    )


def convert_to_jpy(usd_close: pd.Series, usd_jpy: pd.Series, name: str) -> pd.Series:
    aligned = pd.concat(
        [usd_close.rename("usd_price"), usd_jpy.rename("usd_jpy")], axis=1
    ).sort_index()
    aligned["usd_jpy"] = aligned["usd_jpy"].ffill()
    aligned = aligned.loc[usd_close.index].dropna(subset=["usd_price", "usd_jpy"])
    return (aligned["usd_price"] * aligned["usd_jpy"]).rename(name)


def portfolio_inputs(
    returns: pd.DataFrame, periods_per_year: int
) -> tuple[pd.DataFrame, pd.Series] | None:
    clean_returns = returns.dropna(how="any")
    if len(clean_returns) < 2 or clean_returns.shape[1] < 2:
        return None
    covariance = clean_returns.cov() * periods_per_year
    expected_returns = clean_returns.mean() * periods_per_year
    return covariance, expected_returns


def efficient_frontier(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    points: int = 160,
) -> pd.DataFrame:
    asset_names = list(covariance.columns)
    target_returns = np.linspace(
        float(expected_returns.min()), float(expected_returns.max()), points
    )
    rows = []
    tolerance = 1e-8

    for target in target_returns:
        best_weights = None
        best_variance = np.inf

        for subset_size in range(2, len(asset_names) + 1):
            for subset in combinations(range(len(asset_names)), subset_size):
                subset_covariance = covariance.iloc[list(subset), list(subset)].to_numpy()
                subset_returns = expected_returns.iloc[list(subset)].to_numpy()
                inverse_covariance = np.linalg.pinv(subset_covariance)
                ones = np.ones(subset_size)

                a = float(ones @ inverse_covariance @ ones)
                b = float(ones @ inverse_covariance @ subset_returns)
                c = float(subset_returns @ inverse_covariance @ subset_returns)
                system = np.array([[a, b], [b, c]])
                if abs(np.linalg.det(system)) < 1e-12:
                    continue

                multipliers = np.linalg.solve(system, np.array([1.0, target]))
                subset_weights = inverse_covariance @ (
                    multipliers[0] * ones + multipliers[1] * subset_returns
                )
                if np.any(subset_weights < -tolerance):
                    continue

                subset_weights = np.clip(subset_weights, 0.0, None)
                if subset_weights.sum() <= 0:
                    continue
                subset_weights = subset_weights / subset_weights.sum()

                actual_return = float(subset_weights @ subset_returns)
                if abs(actual_return - target) > 1e-5:
                    continue

                variance = float(subset_weights @ subset_covariance @ subset_weights)
                if variance < best_variance:
                    full_weights = np.zeros(len(asset_names))
                    full_weights[list(subset)] = subset_weights
                    best_weights = full_weights
                    best_variance = variance

        if best_weights is not None:
            rows.append(
                {
                    "年率リスク": float(np.sqrt(max(best_variance, 0.0))),
                    "年平均リターン": float(
                        best_weights @ expected_returns.to_numpy()
                    ),
                }
            )

    frontier = pd.DataFrame(rows)
    if frontier.empty:
        return frontier
    return (
        frontier.sort_values("年平均リターン")
        .drop_duplicates(subset=["年平均リターン"])
        .reset_index(drop=True)
    )


def maximum_sharpe_portfolio(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    risk_free_rate: float,
) -> tuple[pd.Series, float, float, float] | None:
    asset_names = list(covariance.columns)
    best = None
    tolerance = 1e-10

    for subset_size in range(1, len(asset_names) + 1):
        for subset in combinations(range(len(asset_names)), subset_size):
            subset_covariance = covariance.iloc[list(subset), list(subset)].to_numpy()
            subset_returns = expected_returns.iloc[list(subset)].to_numpy()

            if subset_size == 1:
                subset_weights = np.array([1.0])
            else:
                excess_returns = subset_returns - risk_free_rate
                inverse_covariance = np.linalg.pinv(subset_covariance)
                raw_weights = inverse_covariance @ excess_returns
                denominator = float(raw_weights.sum())
                if abs(denominator) <= tolerance:
                    continue
                subset_weights = raw_weights / denominator
                if np.any(subset_weights < -tolerance):
                    continue
                subset_weights = np.clip(subset_weights, 0.0, None)
                subset_weights = subset_weights / subset_weights.sum()

            portfolio_return = float(subset_weights @ subset_returns)
            variance = float(subset_weights @ subset_covariance @ subset_weights)
            portfolio_risk = float(np.sqrt(max(variance, 0.0)))
            if portfolio_risk <= tolerance:
                continue

            sharpe = (portfolio_return - risk_free_rate) / portfolio_risk
            if best is None or sharpe > best[3]:
                full_weights = np.zeros(len(asset_names))
                full_weights[list(subset)] = subset_weights
                best = (full_weights, portfolio_return, portfolio_risk, float(sharpe))

    if best is None:
        return None
    weights = pd.Series(best[0], index=asset_names, name="配分比率")
    return weights, best[1], best[2], best[3]


def allocation_table(weights: pd.Series) -> pd.DataFrame:
    allocation = (
        weights.rename("配分比率")
        .mul(100)
        .reset_index()
        .rename(columns={"index": "アセット"})
    )
    allocation["配分比率"] = allocation["配分比率"].round(2)
    return allocation


def rolling_correlations(
    returns: pd.DataFrame,
    years: int,
    frequency: str,
) -> pd.DataFrame:
    result = {}
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index)

    if frequency == "日足":
        window = f"{365 * years}D"
        min_periods = max(60, int(252 * years * 0.7))
    else:
        window = 12 * years
        min_periods = max(6, int(window * 0.7))

    for asset_a, asset_b in combinations(returns.columns, 2):
        pair = returns[[asset_a, asset_b]].dropna()
        if len(pair) < min_periods:
            continue
        result[f"{asset_a} × {asset_b}"] = (
            pair[asset_a]
            .rolling(window=window, min_periods=min_periods)
            .corr(pair[asset_b])
        )

    if not result:
        return pd.DataFrame()
    rolling = pd.DataFrame(result).dropna(how="all")
    rolling.index.name = "Date"
    return rolling


def monte_carlo_simulation(
    initial_amount: float,
    monthly_contribution: float,
    years: int,
    simulations: int,
    annual_return: float,
    annual_risk: float,
    seed: int = 42,
) -> pd.DataFrame:
    """幾何ブラウン運動を月次で近似し、積立込みの資産推移を返す。"""
    months = years * 12
    monthly_drift = (annual_return - 0.5 * annual_risk**2) / 12
    monthly_volatility = annual_risk / np.sqrt(12)

    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=(months, simulations))
    gross_returns = np.exp(monthly_drift + monthly_volatility * shocks)

    paths = np.empty((months + 1, simulations), dtype=float)
    paths[0] = initial_amount
    for month in range(1, months + 1):
        paths[month] = paths[month - 1] * gross_returns[month - 1]
        paths[month] += monthly_contribution

    index = np.arange(months + 1) / 12
    return pd.DataFrame(paths, index=index)


def monte_carlo_summary(paths: pd.DataFrame) -> pd.DataFrame:
    percentiles = paths.quantile([0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    percentiles.columns = ["10%", "25%", "中央値", "75%", "90%"]
    percentiles.index.name = "経過年数"
    return percentiles


st.title("市場データ取得テスト")
st.caption("Yahoo Finance から取得した各資産を、円建てまたはドル建てで比較します。")

st.sidebar.header("取得条件")
selected_names = st.sidebar.multiselect(
    "取得対象", list(ASSETS.keys()), default=list(ASSETS.keys())
)
currency = st.sidebar.radio(
    "表示通貨", ["円建て", "ドル建て"], index=0, horizontal=True
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

risk_free_rate_percent = st.sidebar.number_input(
    "無リスク金利（年率・%）",
    min_value=-5.0,
    max_value=20.0,
    value=0.0,
    step=0.1,
    help="シャープレシオ最大ポートフォリオの計算に使用します。",
)
risk_free_rate = risk_free_rate_percent / 100

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
run_clicked = st.sidebar.button(
    "データ取得", type="primary", use_container_width=True
)
if run_clicked:
    st.session_state["market_data_loaded"] = True
run = st.session_state.get("market_data_loaded", False)

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

                price_series = (
                    convert_to_jpy(usd_close, usd_jpy, name)
                    if currency == "円建て"
                    else usd_close.rename(name)
                )
                if price_series.empty:
                    raise ValueError("価格データがありません。")

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
                    "開始日": "",
                    "終了日": "",
                    latest_value_column: np.nan,
                    "欠損数": 0,
                    "エラー": error,
                }
            )
            continue

        series = successful[name]
        results.append(
            {
                "対象": name,
                "ティッカー": ticker,
                "結果": "成功",
                "件数": len(series),
                "開始日": series.index.min(),
                "終了日": series.index.max(),
                latest_value_column: float(series.iloc[-1]),
                "欠損数": original_missing_counts[name],
                "エラー": "",
            }
        )

    st.subheader("取得結果")
    st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

    if successful:
        prices = pd.concat(successful.values(), axis=1).sort_index()
        prices.index = pd.to_datetime(prices.index)
        prices.index.name = "Date"

        st.subheader(f"価格データ（{currency}）")
        st.dataframe(prices.tail(30), use_container_width=True)

        stats_rows = []
        for name in prices.columns:
            annual_return, annual_risk = annualized_statistics(
                prices[name].dropna(), periods_per_year
            )
            stats_rows.append(
                {
                    "アセット": name,
                    "年平均リターン": annual_return,
                    "年率リスク": annual_risk,
                }
            )

        st.subheader("リターンとリスク")
        st.dataframe(
            pd.DataFrame(stats_rows).style.format(
                {"年平均リターン": "{:.2%}", "年率リスク": "{:.2%}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        indexed_prices = prices.apply(
            lambda series: series / series.dropna().iloc[0] * 100
            if not series.dropna().empty
            else series
        )
        chart_data = indexed_prices.reset_index().melt(
            id_vars="Date", var_name="対象", value_name="指数"
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

        returns = prices.pct_change(fill_method=None)
        maximum_sharpe = None
        covariance = None
        expected_returns = None

        if len(successful) >= 2:
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

            rolling_corr = rolling_correlations(returns, rolling_years, frequency)
            st.subheader("相関係数の推移")
            if not rolling_corr.empty:
                pair_options = list(rolling_corr.columns)
                selected_pairs = st.multiselect(
                    "表示する組み合わせ",
                    pair_options,
                    default=pair_options,
                    key="rolling_pair_selector",
                )
                if selected_pairs:
                    rolling_chart_data = (
                        rolling_corr[selected_pairs]
                        .reset_index()
                        .melt(
                            id_vars="Date",
                            var_name="組み合わせ",
                            value_name="相関係数",
                        )
                        .dropna(subset=["相関係数"])
                    )
                    rolling_fig = px.line(
                        rolling_chart_data,
                        x="Date",
                        y="相関係数",
                        color="組み合わせ",
                        title=(
                            f"相関係数の推移（{rolling_years}年移動相関・"
                            f"{currency}・{frequency}リターン）"
                        ),
                        labels={
                            "Date": "日付",
                            "相関係数": "相関係数",
                            "組み合わせ": "アセットの組み合わせ",
                        },
                    )
                    rolling_fig.add_hline(y=0, line_dash="dash")
                    rolling_fig.update_yaxes(range=[-1, 1], dtick=0.2)
                    rolling_fig.update_layout(hovermode="x unified")
                    st.caption(
                        f"各時点からさかのぼった直近{rolling_years}年間のリターンで計算しています。"
                    )
                    st.plotly_chart(rolling_fig, use_container_width=True)
            else:
                st.info(
                    f"{rolling_years}年移動相関を計算できるだけの共通データがありません。"
                )

            inputs = portfolio_inputs(returns, periods_per_year)
            if inputs is not None:
                covariance, expected_returns = inputs
                maximum_sharpe = maximum_sharpe_portfolio(
                    covariance, expected_returns, risk_free_rate
                )
                frontier = efficient_frontier(covariance, expected_returns)

                st.subheader("効率的フロンティア")
                st.caption(
                    "空売りなし・配分合計100%の条件で、同じ期待リターンに対して年率リスクが最小となるポートフォリオを結んでいます。"
                )
                frontier_fig = go.Figure()
                if not frontier.empty:
                    frontier_fig.add_trace(
                        go.Scatter(
                            x=frontier["年率リスク"] * 100,
                            y=frontier["年平均リターン"] * 100,
                            mode="lines",
                            name="効率的フロンティア",
                        )
                    )

                asset_risks = np.sqrt(np.diag(covariance.to_numpy())) * 100
                asset_returns = expected_returns.to_numpy() * 100
                frontier_fig.add_trace(
                    go.Scatter(
                        x=asset_risks,
                        y=asset_returns,
                        mode="markers+text",
                        text=list(expected_returns.index),
                        textposition="top center",
                        name="各アセット",
                    )
                )

                if maximum_sharpe is not None:
                    sharpe_weights, sharpe_return, sharpe_risk, sharpe_ratio = maximum_sharpe
                    frontier_fig.add_trace(
                        go.Scatter(
                            x=[sharpe_risk * 100],
                            y=[sharpe_return * 100],
                            mode="markers",
                            marker={"size": 15, "symbol": "star"},
                            name="シャープレシオ最大",
                        )
                    )
                    max_x = max(
                        float(frontier["年率リスク"].max() * 100)
                        if not frontier.empty
                        else sharpe_risk * 100,
                        sharpe_risk * 100,
                    )
                    capital_market_x = np.linspace(0, max_x * 1.05, 100)
                    capital_market_y = (
                        risk_free_rate * 100 + sharpe_ratio * capital_market_x
                    )
                    frontier_fig.add_trace(
                        go.Scatter(
                            x=capital_market_x,
                            y=capital_market_y,
                            mode="lines",
                            line={"dash": "dash"},
                            name="資本市場線",
                            hoverinfo="skip",
                        )
                    )

                frontier_fig.update_layout(
                    xaxis_title="年率リスク（%）",
                    yaxis_title="推定年平均リターン（%）",
                    hovermode="closest",
                )
                st.plotly_chart(frontier_fig, use_container_width=True)

                if maximum_sharpe is not None:
                    sharpe_weights, sharpe_return, sharpe_risk, sharpe_ratio = maximum_sharpe
                    st.subheader("シャープレシオ最大ポートフォリオ")
                    st.caption(
                        f"無リスク金利を年率{risk_free_rate_percent:.1f}%として計算しています。"
                    )
                    sharpe_col1, sharpe_col2, sharpe_col3 = st.columns(3)
                    sharpe_col1.metric("推定年平均リターン", f"{sharpe_return:.2%}")
                    sharpe_col2.metric("推定年率リスク", f"{sharpe_risk:.2%}")
                    sharpe_col3.metric("シャープレシオ", f"{sharpe_ratio:.3f}")

                    sharpe_allocation = allocation_table(sharpe_weights)
                    sharpe_table_col, sharpe_chart_col = st.columns([1, 1])
                    with sharpe_table_col:
                        st.dataframe(
                            sharpe_allocation.style.format(
                                {"配分比率": "{:.2f}%"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with sharpe_chart_col:
                        sharpe_positive = sharpe_allocation[
                            sharpe_allocation["配分比率"] > 0
                        ]
                        sharpe_fig = px.pie(
                            sharpe_positive,
                            names="アセット",
                            values="配分比率",
                            title="シャープレシオ最大ポートフォリオの配分",
                            hole=0.35,
                        )
                        sharpe_fig.update_traces(
                            textposition="inside", textinfo="percent+label"
                        )
                        st.plotly_chart(sharpe_fig, use_container_width=True)

        st.subheader("モンテカルロシミュレーションによる将来資産推移")
        st.caption(
            "選択期間の過去リターンとリスクが将来も続くと仮定し、月次の値動きをランダム生成します。"
        )

        with st.expander("シミュレーション条件", expanded=True):
            mc_col1, mc_col2, mc_col3 = st.columns(3)
            with mc_col1:
                initial_default = 10_000_000 if currency == "円建て" else 100_000
                initial_amount = st.number_input(
                    f"初期資産（{currency_unit}）",
                    min_value=0.0,
                    value=float(initial_default),
                    step=float(initial_default / 10),
                )
                contribution_default = 100_000 if currency == "円建て" else 1_000
                monthly_contribution = st.number_input(
                    f"毎月積立額（{currency_unit}）",
                    min_value=0.0,
                    value=float(contribution_default),
                    step=float(contribution_default),
                )
            with mc_col2:
                simulation_years = st.slider("予測期間（年）", 1, 50, 30)
                simulation_count = st.select_slider(
                    "試行回数", options=[500, 1000, 3000, 5000, 10000], value=5000
                )
            with mc_col3:
                portfolio_options = ["均等配分"]
                if "S&P 500 ETF (SPY)" in returns.columns:
                    portfolio_options.append("S&P 500のみ")
                if "全世界株式 ETF (ACWI)" in returns.columns:
                    portfolio_options.append("ACWIのみ")
                if maximum_sharpe is not None:
                    portfolio_options.append("シャープレシオ最大")
                simulation_portfolio = st.radio(
                    "シミュレーション対象",
                    portfolio_options,
                    horizontal=False,
                )

        if simulation_portfolio == "S&P 500のみ":
            simulation_returns = returns[["S&P 500 ETF (SPY)"]].dropna()
        elif simulation_portfolio == "ACWIのみ":
            simulation_returns = returns[["全世界株式 ETF (ACWI)"]].dropna()
        else:
            simulation_returns = returns.dropna(how="any")

        if len(simulation_returns) >= 12:
            if simulation_portfolio == "シャープレシオ最大" and maximum_sharpe is not None:
                simulation_weights = maximum_sharpe[0].reindex(
                    simulation_returns.columns
                ).fillna(0)
            else:
                simulation_weights = pd.Series(
                    1 / simulation_returns.shape[1], index=simulation_returns.columns
                )

            portfolio_return_series = simulation_returns @ simulation_weights
            simulation_annual_return = float(
                portfolio_return_series.mean() * periods_per_year
            )
            simulation_annual_risk = float(
                portfolio_return_series.std() * np.sqrt(periods_per_year)
            )

            mc_paths = monte_carlo_simulation(
                initial_amount=initial_amount,
                monthly_contribution=monthly_contribution,
                years=simulation_years,
                simulations=simulation_count,
                annual_return=simulation_annual_return,
                annual_risk=simulation_annual_risk,
            )
            mc_summary = monte_carlo_summary(mc_paths)
            terminal_values = mc_paths.iloc[-1]
            total_contributed = initial_amount + monthly_contribution * simulation_years * 12
            loss_probability = float((terminal_values < total_contributed).mean())

            metric1, metric2, metric3, metric4 = st.columns(4)
            metric1.metric("前提リターン", f"{simulation_annual_return:.2%}")
            metric2.metric("前提リスク", f"{simulation_annual_risk:.2%}")
            metric3.metric(
                f"{simulation_years}年後の中央値",
                f"{terminal_values.median():,.0f} {currency_unit}",
            )
            metric4.metric("元本割れ確率", f"{loss_probability:.1%}")

            mc_fig = go.Figure()
            mc_fig.add_trace(
                go.Scatter(
                    x=mc_summary.index,
                    y=mc_summary["90%"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            mc_fig.add_trace(
                go.Scatter(
                    x=mc_summary.index,
                    y=mc_summary["10%"],
                    mode="lines",
                    fill="tonexty",
                    name="10〜90%範囲",
                    line={"width": 0},
                    hovertemplate="10%点: %{y:,.0f}<extra></extra>",
                )
            )
            mc_fig.add_trace(
                go.Scatter(
                    x=mc_summary.index,
                    y=mc_summary["75%"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            mc_fig.add_trace(
                go.Scatter(
                    x=mc_summary.index,
                    y=mc_summary["25%"],
                    mode="lines",
                    fill="tonexty",
                    name="25〜75%範囲",
                    line={"width": 0},
                    hovertemplate="25%点: %{y:,.0f}<extra></extra>",
                )
            )
            mc_fig.add_trace(
                go.Scatter(
                    x=mc_summary.index,
                    y=mc_summary["中央値"],
                    mode="lines",
                    name="中央値",
                    hovertemplate="%{x:.1f}年後<br>%{y:,.0f}<extra></extra>",
                )
            )
            contribution_line = initial_amount + monthly_contribution * 12 * mc_summary.index
            mc_fig.add_trace(
                go.Scatter(
                    x=mc_summary.index,
                    y=contribution_line,
                    mode="lines",
                    line={"dash": "dash"},
                    name="累計元本",
                    hovertemplate="累計元本: %{y:,.0f}<extra></extra>",
                )
            )
            mc_fig.update_layout(
                xaxis_title="経過年数",
                yaxis_title=f"資産額（{currency_unit}）",
                hovermode="x unified",
            )
            st.plotly_chart(mc_fig, use_container_width=True)

            terminal_summary = pd.DataFrame(
                {
                    "ケース": ["下位10%", "下位25%", "中央値", "上位25%", "上位10%"],
                    f"{simulation_years}年後の資産額（{currency_unit}）": [
                        terminal_values.quantile(0.10),
                        terminal_values.quantile(0.25),
                        terminal_values.quantile(0.50),
                        terminal_values.quantile(0.75),
                        terminal_values.quantile(0.90),
                    ],
                }
            )
            st.dataframe(
                terminal_summary.style.format(
                    {f"{simulation_years}年後の資産額（{currency_unit}）": "{:,.0f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "この結果は過去データに基づく確率的な試算です。税金・手数料・インフレ・相場構造の変化は考慮していません。"
            )
        else:
            st.info("シミュレーションには、対象となるリターンデータが12件以上必要です。")

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
