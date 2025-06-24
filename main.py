import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import time
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_fetcher import DataFetcher
from factor_calculator import FactorCalculator
from model_engine import ModelEngine

st.set_page_config(
    page_title="PE Ratio Fundamental Analysis Screener",
    page_icon="📊",
    layout="wide"
)

st.title("PE Ratio Fundamental Analysis Screener")
st.markdown("*A quantitative model to assess fair value P/E ratios based on fundamental strength*")

@st.cache_data(ttl=7*24*3600)  # Cache for 1 week
def load_data():
    """Load and cache data for 1 week"""
    data_fetcher = DataFetcher()
    return data_fetcher.get_all_stock_data()

def main():
    # Initialize session state
    if 'selected_factors' not in st.session_state:
        st.session_state.selected_factors = {
            'Risk Aversion': ['maxDrawdown', 'debtToEquity', 'volatility'],
            'Quality': ['returnOnEquity', 'returnOnAssets', 'operatingMargin'],
            'Momentum': ['priceChange52w', 'rsi', 'earningsGrowth']
        }
    
    if 'show_outliers' not in st.session_state:
        st.session_state.show_outliers = True
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Input Factors", "Sector Analysis", "Individual Stock Analysis"])
    
    with tab1:
        create_input_factors_tab()
    
    with tab2:
        create_sector_analysis_tab()
    
    with tab3:
        create_individual_stock_tab()

def create_input_factors_tab():
    st.header("Factor Selection")
    st.markdown("Select which factors to include in the fundamental analysis model:")
    
    # Define all available factors
    all_factors = {
        'Risk Aversion': {
            'maxDrawdown': 'Max Drawdown (12 months)',
            'debtToEquity': 'Debt-to-Equity',
            'volatility': 'Standard Deviation of Returns'
        },
        'Quality': {
            'returnOnEquity': 'Return on Equity',
            'returnOnAssets': 'Return on Assets', 
            'operatingMargin': 'Operating Margin'
        },
        'Momentum': {
            'priceChange52w': '52-week Price Change',
            'rsi': 'Relative Strength Index',
            'earningsGrowth': 'Earnings Growth'
        },
        'Size': {
            'marketCap': 'Market Cap',
            'totalAssets': 'Assets',
            'enterpriseValue': 'EV'
        },
        'Growth': {
            'revenueGrowth': 'Revenue Growth',
            'epsGrowth': 'EPS Growth',
            'cashFlowGrowth': 'Cash Flow Growth'
        },
        'Profitability': {
            'grossMargin': 'Gross Margin',
            'ebitdaMargin': 'EBITDA Margin',
            'netProfitMargin': 'Net Margin'
        },
        'Liquidity': {
            'currentRatio': 'Current Ratio',
            'quickRatio': 'Quick Ratio',
            'interestCoverage': 'Interest Coverage'
        }
    }
    
    # Create nested checkboxes
    for factor_group, metrics in all_factors.items():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            # Parent checkbox
            all_selected = all(
                metric_key in st.session_state.selected_factors.get(factor_group, [])
                for metric_key in metrics.keys()
            )
            
            parent_checked = st.checkbox(
                factor_group, 
                value=all_selected,
                key=f"parent_{factor_group}"
            )
            
            # Update all children when parent is clicked
            if parent_checked and not all_selected:
                st.session_state.selected_factors[factor_group] = list(metrics.keys())
            elif not parent_checked and all_selected:
                st.session_state.selected_factors[factor_group] = []
        
        with col2:
            # Child checkboxes
            if factor_group not in st.session_state.selected_factors:
                st.session_state.selected_factors[factor_group] = []
            
            for metric_key, metric_name in metrics.items():
                checked = st.checkbox(
                    metric_name,
                    value=metric_key in st.session_state.selected_factors[factor_group],
                    key=f"child_{factor_group}_{metric_key}"
                )
                
                if checked and metric_key not in st.session_state.selected_factors[factor_group]:
                    st.session_state.selected_factors[factor_group].append(metric_key)
                elif not checked and metric_key in st.session_state.selected_factors[factor_group]:
                    st.session_state.selected_factors[factor_group].remove(metric_key)
    
    # Display current selection summary
    st.subheader("Selected Factors Summary")
    total_selected = sum(len(factors) for factors in st.session_state.selected_factors.values())
    st.write(f"Total factors selected: {total_selected}")
    
    for factor_group, selected_metrics in st.session_state.selected_factors.items():
        if selected_metrics:
            st.write(f"**{factor_group}**: {len(selected_metrics)} factors")
    
    # Add outlier display control
    st.subheader("Display Options")
    st.session_state.show_outliers = st.checkbox(
        "Display outliers in analysis", 
        value=st.session_state.show_outliers,
        help="When unchecked, outliers will be hidden from the scatter plots and analysis"
    )

def create_sector_analysis_tab():
    st.header("Sector Analysis")
    
    # Load data
    with st.spinner("Loading market data..."):
        try:
            data = load_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return
    
    # Sector selection
    sectors = [
        'Technology', 'Financial Services', 'Consumer Cyclical', 
        'Communication Services', 'Healthcare', 'Industrials',
        'Consumer Defensive', 'Energy', 'Basic Materials', 
        'Real Estate', 'Utilities'
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_sector = st.selectbox("Select Sector", sectors)
    
    with col2:
        # Get top 50 stocks in sector by market cap
        sector_stocks = get_top_stocks_by_sector(data, selected_sector, 50)
        selected_stock = st.selectbox("Select Company", sector_stocks)
    
    if data is not None and not data.empty:
        # Run analysis
        model_engine = ModelEngine()
        results = model_engine.analyze_sector(data, selected_sector, st.session_state.selected_factors)
        
        if results is not None:
            # Create scatter plot
            fig = create_sector_scatter_plot(results, selected_stock, st.session_state.show_outliers)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display model statistics
            display_sector_stats(results, selected_stock)
        else:
            st.error("Unable to analyze sector with current factor selection")

def create_individual_stock_tab():
    st.header("Individual Stock Analysis")
    
    ticker_input = st.text_input("Enter Stock Ticker", placeholder="e.g., AAPL")
    
    if ticker_input and st.button("Analyze Stock"):
        with st.spinner(f"Analyzing {ticker_input.upper()}..."):
            try:
                # Get stock data
                data_fetcher = DataFetcher()
                stock_data = data_fetcher.get_single_stock_data(ticker_input.upper())
                
                if stock_data is not None:
                    # Load full market data for sector comparison
                    market_data = load_data()
                    
                    # Run analysis
                    model_engine = ModelEngine()
                    results = model_engine.analyze_individual_stock(
                        market_data, stock_data, st.session_state.selected_factors
                    )
                    
                    if results is not None:
                        # Display results
                        display_individual_stock_results(results, ticker_input.upper())
                    else:
                        st.error("Unable to analyze stock with current data")
                else:
                    st.error(f"Unable to fetch data for {ticker_input.upper()}")
                    
            except Exception as e:
                st.error(f"Error analyzing stock: {str(e)}")

@st.cache_data
def get_top_stocks_by_sector(data, sector, n=50):
    """Get top N stocks by market cap in a sector"""
    if data is None or data.empty:
        return []
    
    sector_data = data[data['sector'] == sector]
    top_stocks = sector_data.nlargest(n, 'marketCap')
    return top_stocks['symbol'].tolist()

def create_sector_scatter_plot(results, highlighted_stock=None, show_outliers=True):
    """Create scatter plot for sector analysis"""
    fig = go.Figure()
    
    # Separate regular stocks from outliers
    if 'is_outlier' in results.columns:
        regular_stocks = results[results['is_outlier'] != True]
        outlier_stocks = results[results['is_outlier'] == True]
        
        # Add regular stocks
        if not regular_stocks.empty:
            fig.add_trace(go.Scatter(
                x=regular_stocks['fundamental_zscore'],
                y=regular_stocks['pe_ratio'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    line=dict(width=1, color='darkblue')
                ),
                text=regular_stocks['symbol'],
                hovertemplate='<b>%{text}</b><br>Fundamental Z-Score: %{x:.2f}<br>P/E Ratio: %{y:.2f}<extra></extra>',
                name='Companies'
            ))
        
        # Add outliers with different styling only if show_outliers is True
        if show_outliers and not outlier_stocks.empty:
            fig.add_trace(go.Scatter(
                x=outlier_stocks['fundamental_zscore'],
                y=outlier_stocks['pe_ratio'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='orange',
                    line=dict(width=2, color='red'),
                    symbol='triangle-up'
                ),
                text=outlier_stocks['symbol'],
                hovertemplate='<b>%{text}</b> (Outlier)<br>Fundamental Z-Score: %{x:.2f}<br>P/E Ratio: %{y:.2f}<extra></extra>',
                name='Outliers'
            ))
    else:
        # Fallback if no outlier column
        fig.add_trace(go.Scatter(
            x=results['fundamental_zscore'],
            y=results['pe_ratio'],
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(width=1, color='darkblue')
            ),
            text=results['symbol'],
            hovertemplate='<b>%{text}</b><br>Fundamental Z-Score: %{x:.2f}<br>P/E Ratio: %{y:.2f}<extra></extra>',
            name='Companies'
        ))
    
    # Add regression line
    if 'predicted_pe' in results.columns:
        fig.add_trace(go.Scatter(
            x=results['fundamental_zscore'],
            y=results['predicted_pe'],
            mode='lines',
            line=dict(color='red', width=2),
            name='Fair Value Line'
        ))
    
    # Highlight selected stock
    if highlighted_stock and highlighted_stock in results['symbol'].values:
        highlight_data = results[results['symbol'] == highlighted_stock]
        fig.add_trace(go.Scatter(
            x=highlight_data['fundamental_zscore'],
            y=highlight_data['pe_ratio'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='star'
            ),
            name=f'{highlighted_stock} (Selected)',
            showlegend=True
        ))
    
    fig.update_layout(
        title='P/E Ratio vs Fundamental Strength',
        xaxis_title='Fundamental Z-Score',
        yaxis_title='P/E Ratio',
        hovermode='closest'
    )
    
    return fig

def display_sector_stats(results, selected_stock):
    """Display sector analysis statistics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Sector Statistics")
        st.metric("Number of Companies", len(results))
        st.metric("Average P/E Ratio", f"{results['pe_ratio'].mean():.2f}")
        st.metric("P/E Standard Deviation", f"{results['pe_ratio'].std():.2f}")
    
    with col2:
        if hasattr(results, 'attrs') and 'model_info' in results.attrs:
            model_info = results.attrs['model_info']
            st.subheader("Model Performance")
            st.metric("R\u00b2 Score", f"{model_info.get('r2_score', 0):.4f}")
            st.metric("Correlation", f"{model_info.get('correlation', 0):.4f}")
            st.metric("Slope", f"{model_info.get('slope', 0):.4f}")
            st.metric("Intercept", f"{model_info.get('intercept', 0):.4f}")
            
            # Display slope significance
            if 'slope_significance' in model_info:
                sig_info = model_info['slope_significance']
                is_significant = sig_info.get('is_significant', False)
                p_value = sig_info.get('p_value', np.nan)
                
                if not np.isnan(p_value):
                    if is_significant:
                        st.markdown(f"<div style='color: green; font-weight: bold;'>Slope is statistically significant (p={p_value:.4f} < 0.1)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color: red; font-weight: bold;'>Slope is NOT statistically significant (p={p_value:.4f} ≥ 0.1)</div>", unsafe_allow_html=True)
            
            # Display equation
            if 'equation' in model_info:
                st.write(f"**Equation:** {model_info['equation']}")
    
    with col3:
        if selected_stock in results['symbol'].values:
            stock_data = results[results['symbol'] == selected_stock].iloc[0]
            st.subheader(f"{selected_stock} Analysis")
            st.metric("Actual P/E", f"{stock_data['pe_ratio']:.2f}")
            if 'predicted_pe' in stock_data:
                predicted_pe = stock_data['predicted_pe']
                actual_pe = stock_data['pe_ratio']
                st.metric("Predicted P/E", f"{predicted_pe:.2f}")
                diff = actual_pe - predicted_pe
                
                # Color-coded over/undervalued display
                if diff > 0:
                    st.markdown(f"<div style='color: red; font-weight: bold;'>Overvalued by {diff:.2f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color: green; font-weight: bold;'>Undervalued by {abs(diff):.2f}</div>", unsafe_allow_html=True)
                
                # Fair price calculation and display
                if 'price' in stock_data or 'currentPrice' in stock_data:
                    current_price = stock_data.get('price', stock_data.get('currentPrice', 0))
                    if current_price > 0:
                        # Calculate fair price ratio
                        pe_ratio = predicted_pe / actual_pe if actual_pe != 0 else 1
                        fair_price = current_price * pe_ratio
                        upside_downside = ((fair_price - current_price) / current_price) * 100
                        
                        st.subheader("Fair Price Analysis")
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Fair Price", f"${fair_price:.2f}")
                        
                        # Color-coded upside/downside display
                        if upside_downside > 0:
                            st.markdown(f"<div style='color: green; font-weight: bold; font-size: 18px;'>Upside: +{upside_downside:.1f}%</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='color: red; font-weight: bold; font-size: 18px;'>Downside: {upside_downside:.1f}%</div>", unsafe_allow_html=True)
    
    # Display factor weights
    if hasattr(results, 'attrs') and 'model_info' in results.attrs:
        model_info = results.attrs['model_info']
        if 'factor_weights' in model_info:
            st.subheader("Factor Weights")
            weights_df = pd.DataFrame([
                {'Factor': factor.replace('_factor_score', ''), 'Weight': f"{weight:.4f}"}
                for factor, weight in model_info['factor_weights'].items()
            ])
            st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    # Display factor z-scores for selected stock
    if selected_stock in results['symbol'].values:
        display_factor_zscores(results, selected_stock)

def display_individual_stock_results(results, ticker):
    """Display individual stock analysis results"""
    st.subheader(f"Analysis Results for {ticker}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sector", results.get('sector', 'N/A'))
        st.metric("Actual P/E", f"{results.get('actual_pe', 0):.2f}")
    
    with col2:
        st.metric("Predicted P/E", f"{results.get('predicted_pe', 0):.2f}")
        st.metric("Fundamental Score", f"{results.get('fundamental_zscore', 0):.4f}")
    
    with col3:
        diff = results.get('actual_pe', 0) - results.get('predicted_pe', 0)
        # Color-coded over/undervalued display
        if diff > 0:
            st.markdown(f"<div style='color: red; font-weight: bold; font-size: 20px;'>Overvalued by {diff:.2f}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: green; font-weight: bold; font-size: 20px;'>Undervalued by {abs(diff):.2f}</div>", unsafe_allow_html=True)
    
    # Fair price calculation and display
    if 'current_price' in results and results['current_price'] > 0:
        actual_pe = results.get('actual_pe', 0)
        predicted_pe = results.get('predicted_pe', 0)
        current_price = results['current_price']
        
        if actual_pe != 0:
            # Calculate fair price ratio
            pe_ratio = predicted_pe / actual_pe
            fair_price = current_price * pe_ratio
            upside_downside = ((fair_price - current_price) / current_price) * 100
            
            st.subheader("Fair Price Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                st.metric("Fair Price", f"${fair_price:.2f}")
            
            with col3:
                # Color-coded upside/downside display
                if upside_downside > 0:
                    st.markdown(f"<div style='color: green; font-weight: bold; font-size: 20px;'>Upside: +{upside_downside:.1f}%</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color: red; font-weight: bold; font-size: 20px;'>Downside: {upside_downside:.1f}%</div>", unsafe_allow_html=True)
    
    # Display model performance for individual stock
    if 'model_performance' in results:
        model_perf = results['model_performance']
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Performance")
            st.metric("R\u00b2 Score", f"{model_perf.get('r2_score', 0):.4f}")
            st.metric("Correlation", f"{model_perf.get('correlation', 0):.4f}")
            
            # Display slope significance
            if 'slope_significance' in model_perf:
                sig_info = model_perf['slope_significance']
                is_significant = sig_info.get('is_significant', False)
                p_value = sig_info.get('p_value', np.nan)
                
                if not np.isnan(p_value):
                    if is_significant:
                        st.markdown(f"<div style='color: green; font-weight: bold;'>Slope is statistically significant (p={p_value:.4f} < 0.1)</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color: red; font-weight: bold;'>Slope is NOT statistically significant (p={p_value:.4f} ≥ 0.1)</div>", unsafe_allow_html=True)
            
            if 'equation' in model_perf:
                st.write(f"**Equation:** {model_perf['equation']}")
        
        with col2:
            if 'factor_weights' in model_perf:
                st.subheader("Factor Weights")
                weights_df = pd.DataFrame([
                    {'Factor': factor.replace('_factor_score', ''), 'Weight': f"{weight:.4f}"}
                    for factor, weight in model_perf['factor_weights'].items()
                ])
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
    
    # Display factor z-scores for individual stock
    if 'factor_zscores' in results:
        display_individual_factor_zscores(results['factor_zscores'], ticker)
    
    # Display individual metric z-scores for individual stock
    if 'metric_zscores' in results:
        display_individual_metric_zscores(results['metric_zscores'], ticker)
    
    # Show sector comparison plot with factor z-scores
    if 'sector_plot_data' in results:
        # Add the individual stock to the plot data if it's not already there
        plot_data = results['sector_plot_data'].copy()
        
        # Check if the stock is already in the plot data
        if ticker not in plot_data['symbol'].values:
            # Create a row for the individual stock
            stock_row = pd.DataFrame({
                'symbol': [ticker],
                'sector': [results['sector']],
                'pe_ratio': [results['actual_pe']],
                'fundamental_zscore': [results['fundamental_zscore']],
                'predicted_pe': [results['predicted_pe']],
                'is_outlier': [False]  # Mark as not an outlier for visualization
            })
            
            # Add any missing columns with default values
            for col in plot_data.columns:
                if col not in stock_row.columns:
                    stock_row[col] = 0
            
            # Ensure all columns match
            for col in stock_row.columns:
                if col not in plot_data.columns:
                    plot_data[col] = 0
                    
            # Append the stock to the plot data
            plot_data = pd.concat([plot_data, stock_row], ignore_index=True)
        
        fig = create_enhanced_sector_scatter_plot(plot_data, ticker, results.get('factor_zscores', {}), st.session_state.show_outliers)
        st.subheader(f"{ticker} vs {results['sector']} Sector Peers")
        st.plotly_chart(fig, use_container_width=True)

def display_factor_zscores(results, selected_stock):
    """Display factor z-scores for selected stock"""
    if selected_stock not in results['symbol'].values:
        return
    
    stock_data = results[results['symbol'] == selected_stock].iloc[0]
    
    st.subheader(f"Factor Z-Scores for {selected_stock}")
    
    # Get all factor score columns
    factor_cols = [col for col in stock_data.index if col.endswith('_factor_score')]
    
    if factor_cols:
        zscore_data = []
        for factor_col in factor_cols:
            factor_name = factor_col.replace('_factor_score', '')
            zscore_value = stock_data[factor_col]
            zscore_data.append({
                'Factor': factor_name,
                'Z-Score': f"{zscore_value:.3f}",
                'Interpretation': get_zscore_interpretation(zscore_value)
            })
        
        zscore_df = pd.DataFrame(zscore_data)
        st.dataframe(zscore_df, use_container_width=True, hide_index=True)
    
    # Also show individual metric z-scores
    st.subheader(f"Individual Metric Z-Scores for {selected_stock}")
    metric_zscores = []
    
    for col in stock_data.index:
        if col.endswith('_zscore') and not col.endswith('_factor_score'):
            metric_name = col.replace('_zscore', '')
            zscore_value = stock_data[col]
            metric_zscores.append({
                'Metric': metric_name,
                'Z-Score': f"{zscore_value:.3f}",
                'Interpretation': get_zscore_interpretation(zscore_value)
            })
    
    if metric_zscores:
        metric_df = pd.DataFrame(metric_zscores)
        st.dataframe(metric_df, use_container_width=True, hide_index=True)

def display_individual_factor_zscores(factor_zscores, ticker):
    """Display factor z-scores for individual stock analysis"""
    st.subheader(f"Factor Z-Scores for {ticker}")
    
    zscore_data = []
    for factor_name, zscore_value in factor_zscores.items():
        zscore_data.append({
            'Factor': factor_name,
            'Z-Score': f"{zscore_value:.3f}",
            'Interpretation': get_zscore_interpretation(zscore_value)
        })
    
    if zscore_data:
        zscore_df = pd.DataFrame(zscore_data)
        st.dataframe(zscore_df, use_container_width=True, hide_index=True)

def display_individual_metric_zscores(metric_zscores, ticker):
    """Display individual metric z-scores for individual stock analysis"""
    st.subheader(f"Individual Metric Z-Scores for {ticker}")
    
    zscore_data = []
    for metric_name, zscore_value in metric_zscores.items():
        zscore_data.append({
            'Metric': metric_name,
            'Z-Score': f"{zscore_value:.3f}",
            'Interpretation': get_zscore_interpretation(zscore_value)
        })
    
    if zscore_data:
        zscore_df = pd.DataFrame(zscore_data)
        st.dataframe(zscore_df, use_container_width=True, hide_index=True)

def get_zscore_interpretation(zscore):
    """Get interpretation of z-score value"""
    if zscore > 2:
        return "Very High (+2σ)"
    elif zscore > 1:
        return "High (+1σ)"
    elif zscore > 0.5:
        return "Above Average"
    elif zscore > -0.5:
        return "Average"
    elif zscore > -1:
        return "Below Average"
    elif zscore > -2:
        return "Low (-1σ)"
    else:
        return "Very Low (-2σ)"

def create_enhanced_sector_scatter_plot(results, highlighted_stock=None, factor_zscores=None, show_outliers=True):
    """Create enhanced scatter plot with factor z-scores overlay"""
    fig = go.Figure()
    
    # Separate regular stocks from outliers
    if 'is_outlier' in results.columns:
        regular_stocks = results[results['is_outlier'] != True]
        outlier_stocks = results[results['is_outlier'] == True]
        
        # Add regular stocks
        if not regular_stocks.empty:
            fig.add_trace(go.Scatter(
                x=regular_stocks['fundamental_zscore'],
                y=regular_stocks['pe_ratio'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightblue',
                    line=dict(width=1, color='darkblue')
                ),
                text=regular_stocks['symbol'],
                hovertemplate='<b>%{text}</b><br>Fundamental Z-Score: %{x:.2f}<br>P/E Ratio: %{y:.2f}<extra></extra>',
                name='Companies'
            ))
        
        # Add outliers with different styling only if show_outliers is True
        if show_outliers and not outlier_stocks.empty:
            fig.add_trace(go.Scatter(
                x=outlier_stocks['fundamental_zscore'],
                y=outlier_stocks['pe_ratio'],
                mode='markers',
                marker=dict(
                    size=10,
                    color='orange',
                    line=dict(width=2, color='red'),
                    symbol='triangle-up'
                ),
                text=outlier_stocks['symbol'],
                hovertemplate='<b>%{text}</b> (Outlier)<br>Fundamental Z-Score: %{x:.2f}<br>P/E Ratio: %{y:.2f}<extra></extra>',
                name='Outliers'
            ))
    else:
        # Fallback if no outlier column
        fig.add_trace(go.Scatter(
            x=results['fundamental_zscore'],
            y=results['pe_ratio'],
            mode='markers',
            marker=dict(
                size=8,
                color='lightblue',
                line=dict(width=1, color='darkblue')
            ),
            text=results['symbol'],
            hovertemplate='<b>%{text}</b><br>Fundamental Z-Score: %{x:.2f}<br>P/E Ratio: %{y:.2f}<extra></extra>',
            name='Companies'
        ))
    
    # Add regression line
    if 'predicted_pe' in results.columns:
        fig.add_trace(go.Scatter(
            x=results['fundamental_zscore'],
            y=results['predicted_pe'],
            mode='lines',
            line=dict(color='red', width=2),
            name='Fair Value Line'
        ))
    
    # Highlight selected stock
    if highlighted_stock and highlighted_stock in results['symbol'].values:
        highlight_data = results[results['symbol'] == highlighted_stock]
        fig.add_trace(go.Scatter(
            x=highlight_data['fundamental_zscore'],
            y=highlight_data['pe_ratio'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='star'
            ),
            name=f'{highlighted_stock} (Selected)',
            showlegend=True
        ))
        
    
    fig.update_layout(
        title='P/E Ratio vs Fundamental Strength (with Factor Z-Scores)',
        xaxis_title='Fundamental Z-Score',
        yaxis_title='P/E Ratio',
        hovermode='closest'
    )
    
    return fig

if __name__ == "__main__":
    main()