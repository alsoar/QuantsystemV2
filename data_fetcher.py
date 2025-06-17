import yfinance as yf
import pandas as pd
import numpy as np
import time
import pickle
import os
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class DataFetcher:
    def __init__(self, cache_dir='data_cache'):
        self.cache_dir = cache_dir
        self.cache_duration = 7 * 24 * 3600  # 1 week in seconds
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Define sector mappings
        self.sector_mappings = {
            'Technology': 'Technology',
            'Financial Services': 'Financial Services', 
            'Consumer Cyclical': 'Consumer Cyclical',
            'Communication Services': 'Communication Services',
            'Healthcare': 'Healthcare',
            'Industrials': 'Industrials',
            'Consumer Defensive': 'Consumer Defensive',
            'Energy': 'Energy',
            'Basic Materials': 'Basic Materials',
            'Real Estate': 'Real Estate',
            'Utilities': 'Utilities'
        }
        
        # Popular tickers by sector for data collection (expanded to 60 per sector to ensure 50 after outlier removal)
        self.sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'CRM', 'ADBE', 'INTC', 'CSCO', 'ORCL', 'AMD', 'PYPL', 'UBER', 'SPOT', 'SHOP', 'SQ', 'TWTR', 'AVGO', 'TXN', 'QCOM', 'NOW', 'INTU', 'MU', 'AMAT', 'LRCX', 'ADI', 'KLAC', 'MRVL', 'SNPS', 'CDNS', 'FTNT', 'TEAM', 'WDAY', 'ZM', 'DOCU', 'OKTA', 'CRWD', 'NET', 'DDOG', 'SNOW', 'PLTR', 'RBLX', 'U', 'FSLY', 'TWLO', 'ZS', 'ESTC', 'SPLK', 'VEEV', 'RNG', 'COUP', 'BILL', 'PAYC', 'PCTY', 'SMAR', 'APPN', 'WORK'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB', 'PNC', 'TFC', 'COF', 'BLK', 'SCHW', 'CB', 'ICE', 'CME', 'SPGI', 'MCO', 'AON', 'MMC', 'V', 'MA', 'BRK-B', 'BX', 'KKR', 'APO', 'CG', 'MKTX', 'NDAQ', 'MSCI', 'FIS', 'FISV', 'GPN', 'AJG', 'BRO', 'PGR', 'ALL', 'TRV', 'AFL', 'MET', 'PRU', 'AIG', 'L', 'GL', 'RGA', 'RE', 'ACGL', 'CINF', 'Y', 'WTW', 'TROW', 'IVZ', 'NTRS', 'STT', 'BK', 'RF', 'CFG', 'KEY', 'HBAN', 'FITB'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'GM', 'F', 'ABNB', 'EBAY', 'MAR', 'HLT', 'RCL', 'CCL', 'NCLH', 'LVS', 'MGM', 'ORLY', 'AZO', 'ROST', 'YUM', 'CMG', 'DPZ', 'QSR', 'DRI', 'EAT', 'DIN', 'BJRI', 'CAKE', 'BLMN', 'TXRH', 'SHAK', 'WING', 'PZZA', 'RUTH', 'JACK', 'SONC', 'DAVE', 'WEN', 'MCD', 'YUM', 'SBUX', 'CMG', 'DPZ', 'QSR', 'DRI', 'EAT', 'DIN', 'BJRI', 'CAKE', 'BLMN', 'TXRH', 'SHAK', 'WING', 'PZZA', 'RUTH'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'ATVI', 'EA', 'TTWO', 'ROKU', 'PINS', 'SNAP', 'TWTR', 'DISH', 'SIRI', 'IPG', 'OMC', 'GOOG', 'WBD', 'PARA', 'FOX', 'FOXA', 'LUMN', 'FYBR', 'CABO', 'FUBO', 'GSAT', 'IRDM', 'VIAC', 'DISCA', 'DISCK', 'LILAK', 'LILA', 'BATRK', 'BATRA', 'FWONK', 'FWONA', 'LSXMK', 'LSXMA', 'LSXMB', 'TRIP', 'MTCH', 'BMBL', 'MSGS', 'MSGN', 'NWSA', 'NWS', 'WMG', 'LYV', 'EDR', 'REZI', 'CARS', 'ANGI', 'IAC', 'QNST', 'ZNGA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'DHR', 'BMY', 'CVS', 'MDT', 'GILD', 'AMGN', 'ISRG', 'VRTX', 'REGN', 'CI', 'HUM', 'ANTM', 'CVS', 'UHS', 'LLY', 'MRK', 'BIIB', 'ILMN', 'BDX', 'BSX', 'EW', 'SYK', 'ZBH', 'BAX', 'PKI', 'A', 'WST', 'RMD', 'HOLX', 'TFX', 'DXCM', 'ZTS', 'IDXX', 'IQV', 'CRL', 'LH', 'DGX', 'MTD', 'TECH', 'MOH', 'CNC', 'VEEV', 'TDOC', 'DOCU', 'AMWL', 'OCDX', 'HCAT', 'MRNA', 'BNTX', 'NVAX', 'VCEL', 'BLUE', 'BEAM'],
            'Industrials': ['BA', 'HON', 'UPS', 'RTX', 'LMT', 'CAT', 'DE', 'GE', 'MMM', 'FDX', 'NOC', 'EMR', 'ETN', 'ITW', 'CSX', 'NSC', 'UNP', 'LUV', 'DAL', 'AAL', 'GD', 'TDG', 'CTAS', 'CMI', 'PH', 'ROK', 'DOV', 'PCAR', 'IEX', 'FTV', 'AME', 'ROP', 'LDOS', 'TXT', 'ALLE', 'OTIS', 'CARR', 'PWR', 'JCI', 'IR', 'SWK', 'FAST', 'ODFL', 'XPO', 'CHRW', 'JBHT', 'KNX', 'EXPD', 'LSTR', 'HUBG', 'ARCB', 'MATX', 'GWR', 'AL', 'FWRD', 'SNDR', 'RXO', 'SAIA', 'YELL', 'WERN'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'K', 'HSY', 'MKC', 'CPB', 'SJM', 'CAG', 'HRL', 'TSN', 'KHC', 'CHD', 'CLX', 'MNST', 'KDP', 'STZ', 'TAP', 'BF-B', 'DEO', 'SAM', 'CCEP', 'FMX', 'ABEV', 'BUD', 'BREW', 'COKE', 'KOF', 'FIZZ', 'CELH', 'REED', 'ZVIA', 'PRMW', 'WTER', 'HINT', 'BWT', 'WPRT', 'MNST', 'BANG', 'RMHB', 'KONA', 'NAPA', 'STKL', 'GPRE', 'MGPI', 'LWAY', 'EAST', 'WEST', 'UNFI', 'SFM', 'VLGEA', 'CALM', 'SAFM'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'MPC', 'VLO', 'PSX', 'OXY', 'BKR', 'HAL', 'DVN', 'FANG', 'APA', 'MRO', 'HES', 'NOV', 'HP', 'RRC', 'KMI', 'OKE', 'WMB', 'ENB', 'TRP', 'ET', 'EPD', 'MPLX', 'PAA', 'WES', 'AM', 'TRGP', 'EQT', 'CTRA', 'CNX', 'AR', 'SM', 'RIG', 'VAL', 'NE', 'MTDR', 'PR', 'MUR', 'CDEV', 'CPG', 'MEG', 'CVE', 'SU', 'CNQ', 'TOU', 'PBA', 'WCP', 'BTE', 'CJ', 'ERF', 'VET', 'ATH', 'CPE', 'CRC'],
            'Basic Materials': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'DOW', 'DD', 'ALB', 'CE', 'VMC', 'MLM', 'NUE', 'STLD', 'X', 'CLF', 'AA', 'CENX', 'MP', 'LAC', 'PPG', 'RPM', 'FMC', 'LYB', 'CF', 'MOS', 'IFF', 'EMN', 'WLK', 'AXTA', 'OLN', 'CC', 'ASH', 'KWR', 'HUN', 'NEU', 'PRM', 'BCPC', 'ESI', 'GRA', 'KRA', 'OMG', 'PKG', 'SEE', 'SON', 'BCC', 'SLGN', 'TROX', 'FUL', 'HWKN', 'SMG', 'SXT', 'TTEK', 'UEC', 'URG', 'UUUU', 'DNN', 'LTBR', 'LEU', 'PALAF'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'O', 'PSA', 'WELL', 'DLR', 'SBAC', 'EXR', 'AVB', 'EQR', 'VTR', 'ESS', 'MAA', 'UDR', 'CPT', 'AIV', 'BXP', 'VICI', 'HST', 'ARE', 'PEAK', 'SLG', 'REG', 'FRT', 'KIM', 'TCO', 'ROIC', 'WPC', 'NNN', 'STAG', 'KRC', 'HIW', 'DEI', 'CUZ', 'BDN', 'PGRE', 'OFC', 'JBGS', 'SHO', 'RHP', 'ELS', 'UMH', 'SUI', 'MSA', 'CSR', 'AHH', 'ALEX', 'BRX', 'COR', 'CUBE', 'DRH', 'RLJ', 'PEB', 'APLE', 'INN', 'XHR', 'CHCT'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'PEG', 'XEL', 'WEC', 'ED', 'ETR', 'AWK', 'ES', 'FE', 'AEE', 'DTE', 'PPL', 'CMS', 'NI', 'PCG', 'EIX', 'ATO', 'EVRG', 'CNP', 'VST', 'LNT', 'AES', 'NRG', 'CEG', 'PNW', 'UGI', 'OGE', 'MDU', 'BKH', 'POR', 'SR', 'NWE', 'AVA', 'AGR', 'CPK', 'HE', 'NEP', 'ALE', 'IDA', 'PNM', 'NJR', 'SWX', 'OTTR', 'UTL', 'MGEE', 'AWR', 'CWT', 'YORW', 'ARTNA', 'CTWS', 'SJW', 'MSEX', 'CWCO']
        }
    
    def _get_cache_path(self, filename):
        """Get cache file path"""
        return os.path.join(self.cache_dir, filename)
    
    def _is_cache_valid(self, filepath):
        """Check if cache file is still valid"""
        if not os.path.exists(filepath):
            return False
        
        file_age = time.time() - os.path.getmtime(filepath)
        return file_age < self.cache_duration
    
    def _save_to_cache(self, data, filename):
        """Save data to cache"""
        filepath = self._get_cache_path(filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_from_cache(self, filename):
        """Load data from cache"""
        filepath = self._get_cache_path(filename)
        if self._is_cache_valid(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_stock_info(self, symbol):
        """Get basic stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            time.sleep(0.1)  # Rate limiting
            return info
        except Exception as e:
            print(f"Error fetching info for {symbol}: {str(e)}")
            return None
    
    def get_stock_financials(self, symbol):
        """Get financial data for a stock"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data for price calculations
            hist = ticker.history(period="1y")
            
            # Get financial statements
            income_stmt = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            time.sleep(0.2)  # Rate limiting
            
            return {
                'history': hist,
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            print(f"Error fetching financials for {symbol}: {str(e)}")
            return None
    
    def calculate_financial_metrics(self, symbol, info, financials):
        """Calculate all required financial metrics for a stock"""
        try:
            metrics = {'symbol': symbol}
            
            # Basic info
            metrics['sector'] = info.get('sector', 'Unknown')
            metrics['marketCap'] = info.get('marketCap', np.nan)
            metrics['enterpriseValue'] = info.get('enterpriseValue', np.nan)
            metrics['totalAssets'] = info.get('totalAssets', np.nan)
            
            # P/E Ratio (main target variable)
            metrics['pe_ratio'] = info.get('trailingPE', np.nan)
            
            # Skip if no P/E ratio or negative P/E
            if pd.isna(metrics['pe_ratio']) or metrics['pe_ratio'] <= 0:
                return None
            
            # Risk Aversion factors
            if financials and 'history' in financials:
                hist = financials['history']
                if not hist.empty:
                    # Max Drawdown (12 months)
                    rolling_max = hist['Close'].expanding().max()
                    drawdown = (hist['Close'] - rolling_max) / rolling_max
                    metrics['maxDrawdown'] = drawdown.min()
                    
                    # Volatility (Standard Deviation of Returns)
                    returns = hist['Close'].pct_change().dropna()
                    metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
                    
                    # 52-week Price Change
                    if len(hist) >= 252:
                        metrics['priceChange52w'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[-252] - 1)
                    else:
                        metrics['priceChange52w'] = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
            
            # Debt-to-Equity
            metrics['debtToEquity'] = info.get('debtToEquity', np.nan)
            
            # Quality factors
            metrics['returnOnEquity'] = info.get('returnOnEquity', np.nan)
            metrics['returnOnAssets'] = info.get('returnOnAssets', np.nan) 
            metrics['operatingMargin'] = info.get('operatingMargins', np.nan)
            metrics['grossMargin'] = info.get('grossMargins', np.nan)
            metrics['netProfitMargin'] = info.get('profitMargins', np.nan)
            
            # Growth factors
            metrics['revenueGrowth'] = info.get('revenueGrowth', np.nan)
            metrics['earningsGrowth'] = info.get('earningsGrowth', np.nan)
            
            # Try to calculate EPS growth from financials
            if financials and 'income_statement' in financials:
                income = financials['income_statement']
                if not income.empty and len(income.columns) >= 2:
                    try:
                        current_eps = income.loc['Diluted EPS'].iloc[0] if 'Diluted EPS' in income.index else np.nan
                        prev_eps = income.loc['Diluted EPS'].iloc[1] if 'Diluted EPS' in income.index and len(income.columns) > 1 else np.nan
                        if not pd.isna(current_eps) and not pd.isna(prev_eps) and prev_eps != 0:
                            metrics['epsGrowth'] = (current_eps - prev_eps) / abs(prev_eps)
                    except:
                        metrics['epsGrowth'] = np.nan
            
            if 'epsGrowth' not in metrics:
                metrics['epsGrowth'] = np.nan
            
            # Cash flow growth - approximate from info
            metrics['cashFlowGrowth'] = info.get('operatingCashflow', np.nan)
            
            # EBITDA Margin
            metrics['ebitdaMargin'] = info.get('ebitdaMargins', np.nan)
            
            # Liquidity factors
            metrics['currentRatio'] = info.get('currentRatio', np.nan)
            metrics['quickRatio'] = info.get('quickRatio', np.nan)
            metrics['interestCoverage'] = info.get('interestCoverage', np.nan)
            
            # RSI - approximate calculation
            if financials and 'history' in financials:
                hist = financials['history']
                if not hist.empty and len(hist) >= 14:
                    metrics['rsi'] = self._calculate_rsi(hist['Close'])
                else:
                    metrics['rsi'] = np.nan
            else:
                metrics['rsi'] = np.nan
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics for {symbol}: {str(e)}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return np.nan
    
    def get_sector_stocks(self, sector, max_stocks=60):
        """Get list of stocks for a specific sector"""
        return self.sector_tickers.get(sector, [])[:max_stocks]
    
    def get_all_stock_data(self):
        """Get data for all stocks across all sectors"""
        cache_filename = 'all_stocks_data.pkl'
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_filename)
        if cached_data is not None:
            print("Loading data from cache...")
            return cached_data
        
        print("Fetching fresh data from Yahoo Finance...")
        all_data = []
        
        for sector, tickers in self.sector_tickers.items():
            print(f"Processing {sector} sector...")
            
            for i, symbol in enumerate(tickers):
                try:
                    print(f"  Fetching {symbol} ({i+1}/{len(tickers)})...")
                    
                    # Get basic info
                    info = self.get_stock_info(symbol)
                    if not info:
                        continue
                    
                    # Get financial data
                    financials = self.get_stock_financials(symbol)
                    
                    # Calculate metrics
                    metrics = self.calculate_financial_metrics(symbol, info, financials)
                    if metrics:
                        all_data.append(metrics)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"    Error processing {symbol}: {str(e)}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Remove rows with no P/E ratio or negative P/E
        df = df[df['pe_ratio'].notna()]
        df = df[df['pe_ratio'] > 0]
        
        print(f"Successfully fetched data for {len(df)} stocks")
        
        # Save to cache
        self._save_to_cache(df, cache_filename)
        
        return df
    
    def get_single_stock_data(self, symbol):
        """Get data for a single stock"""
        try:
            print(f"Fetching data for {symbol}...")
            
            # Get basic info
            info = self.get_stock_info(symbol)
            if not info:
                return None
            
            # Get financial data
            financials = self.get_stock_financials(symbol)
            
            # Calculate metrics
            metrics = self.calculate_financial_metrics(symbol, info, financials)
            
            return metrics
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None