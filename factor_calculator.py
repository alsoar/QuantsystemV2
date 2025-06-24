import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional

class FactorCalculator:
    def __init__(self):
        # Define factor mappings
        self.factor_mappings = {
            'Risk Aversion': {
                'maxDrawdown': 'maxDrawdown',
                'debtToEquity': 'debtToEquity', 
                'volatility': 'volatility'
            },
            'Quality': {
                'returnOnEquity': 'returnOnEquity',
                'returnOnAssets': 'returnOnAssets',
                'operatingMargin': 'operatingMargin'
            },
            'Momentum': {
                'priceChange52w': 'priceChange52w',
                'rsi': 'rsi',
                'earningsGrowth': 'earningsGrowth'
            },
            'Size': {
                'marketCap': 'marketCap',
                'totalAssets': 'totalAssets',
                'enterpriseValue': 'enterpriseValue'
            },
            'Growth': {
                'revenueGrowth': 'revenueGrowth',
                'epsGrowth': 'epsGrowth',
                'cashFlowGrowth': 'cashFlowGrowth'
            },
            'Profitability': {
                'grossMargin': 'grossMargin',
                'ebitdaMargin': 'ebitdaMargin',
                'netProfitMargin': 'netProfitMargin'
            },
            'Liquidity': {
                'currentRatio': 'currentRatio',
                'quickRatio': 'quickRatio',
                'interestCoverage': 'interestCoverage'
            }
        }
    
    def calculate_z_scores_by_sector(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate z-scores for each metric within each sector/industry with outlier handling
        """
        df = data.copy()
        
        # Get all metric columns (exclude non-metric columns)
        non_metric_cols = ['symbol', 'sector', 'pe_ratio']
        metric_cols = [col for col in df.columns if col not in non_metric_cols]
        
        # Calculate z-scores within each sector
        for sector in df['sector'].unique():
            sector_mask = df['sector'] == sector
            sector_data = df[sector_mask]
            
            if len(sector_data) < 3:  # Need at least 3 companies for meaningful z-scores
                continue
                
            for col in metric_cols:
                if col in df.columns:
                    sector_values = sector_data[col].dropna()
                    
                    if len(sector_values) >= 3 and sector_values.std() > 0:
                        # Use robust statistics to reduce impact of outliers
                        # Use median and IQR-based scaling instead of mean/std for some metrics
                        q25, q50, q75 = sector_values.quantile([0.25, 0.5, 0.75])
                        iqr = q75 - q25
                        
                        if iqr > 0:
                            # Robust z-score using median and IQR
                            robust_zscore = (df.loc[sector_mask, col] - q50) / (iqr * 1.35)  # 1.35 approximates std for normal dist
                            df.loc[sector_mask, f'{col}_zscore'] = robust_zscore
                        else:
                            # Fallback to standard z-score
                            mean_val = sector_values.mean()
                            std_val = sector_values.std()
                            standard_zscore = (df.loc[sector_mask, col] - mean_val) / std_val
                            df.loc[sector_mask, f'{col}_zscore'] = standard_zscore
                    else:
                        # If not enough data or no variance, set z-score to 0
                        df.loc[sector_mask, f'{col}_zscore'] = 0
        
        return df
    
    def calculate_factor_scores(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Calculate factor scores by averaging z-scores of component metrics
        """
        df = data.copy()
        
        # Calculate factor scores for each factor group
        for factor_name, selected_metrics in selected_factors.items():
            # Always create the factor score column, even if no metrics selected
            if not selected_metrics:
                # If no metrics selected for this factor, set factor score to 0
                df[f'{factor_name}_factor_score'] = 0
                continue
                
            # Get the z-score columns for selected metrics
            zscore_cols = []
            for metric in selected_metrics:
                zscore_col = f'{metric}_zscore'
                if zscore_col in df.columns:
                    zscore_cols.append(zscore_col)
            
            if zscore_cols:
                # Calculate mean z-score across selected metrics
                df[f'{factor_name}_factor_score'] = df[zscore_cols].mean(axis=1, skipna=True)
                
                # For Risk Aversion, multiply by -1 (higher risk should be negative)
                if factor_name == 'Risk Aversion':
                    df[f'{factor_name}_factor_score'] *= -1
            else:
                # If no valid metrics, set factor score to 0
                df[f'{factor_name}_factor_score'] = 0
        
        return df
    
    def get_factor_score_columns(self, selected_factors: Dict[str, List[str]]) -> List[str]:
        """
        Get list of factor score column names based on selected factors
        """
        factor_cols = []
        for factor_name, selected_metrics in selected_factors.items():
            # Only include factors that have at least one metric selected
            if selected_metrics and len(selected_metrics) > 0:
                factor_cols.append(f'{factor_name}_factor_score')
        return factor_cols
    
    def validate_data_for_analysis(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> bool:
        """
        Validate that we have sufficient data for analysis
        """
        if data.empty:
            return False
        
        # Check if we have P/E ratios
        if 'pe_ratio' not in data.columns or data['pe_ratio'].isna().all():
            return False
        
        # Check if we have at least one selected factor with valid data
        has_valid_factor = False
        for factor_name, selected_metrics in selected_factors.items():
            if selected_metrics:
                # Check if any of the selected metrics have valid data
                for metric in selected_metrics:
                    if metric in data.columns and not data[metric].isna().all():
                        has_valid_factor = True
                        break
                if has_valid_factor:
                    break
        
        return has_valid_factor
    
    def filter_outliers(self, data: pd.DataFrame, column: str = 'pe_ratio', iqr_threshold: float = 4.0) -> pd.DataFrame:
        """
        Mark outliers using IQR method within each sector - more conservative threshold to reduce false positives
        """
        df = data.copy()
        df['is_outlier'] = False  # Initialize all as non-outliers
        
        for sector in df['sector'].unique():
            sector_mask = df['sector'] == sector
            sector_data = df[sector_mask]
            
            if len(sector_data) < 8:  # Need more data for reliable IQR method
                continue
            
            values = sector_data[column].dropna()
            if len(values) < 8:
                continue
                
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            
            if iqr > 0:
                # Use more conservative IQR threshold to reduce false positives
                lower_bound = q1 - iqr_threshold * iqr
                upper_bound = q3 + iqr_threshold * iqr
                
                # Mark only extreme outliers
                outlier_mask = (sector_data[column] < lower_bound) | (sector_data[column] > upper_bound)
                
                # Additional check: only mark as outlier if it's really extreme (top/bottom 5%)
                extreme_outliers = []
                for idx in sector_data[outlier_mask].index:
                    value = sector_data.loc[idx, column]
                    percentile = (values <= value).mean() * 100
                    if percentile <= 2.5 or percentile >= 97.5:  # Only top/bottom 2.5%
                        extreme_outliers.append(idx)
                
                df.loc[extreme_outliers, 'is_outlier'] = True
        
        return df
    
    def filter_outliers_iterative_25std(self, data: pd.DataFrame, column: str = 'pe_ratio', 
                                      max_iterations: int = 10) -> pd.DataFrame:
        """
        Detect outliers using iterative 2.5 standard deviation method.
        Any company with a P/E ratio greater than 2.5 std devs away from mean 
        (calculated iteratively by industry) is marked as an outlier.
        
        Args:
            data: DataFrame with the data
            column: Column to check for outliers (P/E ratio)
            max_iterations: Maximum number of iterations to prevent infinite loops
            
        Returns:
            DataFrame with 'is_outlier' column added
        """
        df = data.copy()
        df['is_outlier'] = False
        
        for sector in df['sector'].unique():
            sector_mask = df['sector'] == sector
            sector_data = df[sector_mask].copy()
            
            if len(sector_data) < 5:  # Need minimum data for meaningful statistics
                continue
            
            values = sector_data[column].dropna()
            if len(values) < 5:
                continue
            
            # Iterative outlier detection with 2.5 std dev threshold
            outlier_indices = set()
            current_values = values.copy()
            
            for iteration in range(max_iterations):
                if len(current_values) < 3:  # Stop if too few values remain
                    break
                
                # Calculate mean and std for current iteration
                mean_val = current_values.mean()
                std_val = current_values.std()
                
                if std_val == 0:  # No variation, no outliers
                    break
                
                # Find outliers: values > 2.5 std devs from mean
                z_scores = np.abs((current_values - mean_val) / std_val)
                new_outliers = current_values[z_scores > 2.5].index.tolist()
                
                if not new_outliers:  # No new outliers found
                    break
                
                # Add new outliers to the set
                outlier_indices.update(new_outliers)
                
                # Remove outliers from current values for next iteration
                current_values = current_values.drop(new_outliers)
            
            # Mark outliers in the main dataframe
            for idx in outlier_indices:
                df.loc[idx, 'is_outlier'] = True
        
        return df

    def filter_outliers_iterative_zscore(self, data: pd.DataFrame, column: str = 'pe_ratio', 
                                       z_threshold: float = 3.5, max_iterations: int = 5) -> pd.DataFrame:
        """
        Detect outliers using modified z-score method with more conservative thresholds
        to reduce false positives. Only marks extreme outliers that are clearly anomalous.
        
        Args:
            data: DataFrame with the data
            column: Column to check for outliers
            z_threshold: Z-score threshold for outlier detection (default 3.5 for conservative detection)
            max_iterations: Maximum number of iterations to prevent infinite loops
            
        Returns:
            DataFrame with 'is_outlier' column added
        """
        df = data.copy()
        df['is_outlier'] = False
        
        for sector in df['sector'].unique():
            sector_mask = df['sector'] == sector
            sector_data = df[sector_mask]
            
            if len(sector_data) < 8:  # Need at least 8 points for reliable statistics
                continue
            
            values = sector_data[column].dropna()
            if len(values) < 8:
                continue
            
            # Use a single pass approach - for each point, calculate z-score using all other points
            sector_indices = values.index.tolist()
            outliers_found = []
            
            for idx in sector_indices:
                # Get all values except the current one
                other_values = values.drop(idx)
                
                # Calculate mean and std excluding current point
                mean_without_point = other_values.mean()
                std_without_point = other_values.std()
                
                if std_without_point > 0:
                    # Calculate z-score for current point
                    z_score = abs((values[idx] - mean_without_point) / std_without_point)
                    
                    # Mark as outlier if z-score exceeds threshold
                    if z_score > z_threshold:
                        outliers_found.append((idx, z_score, values[idx]))
            
            # Sort by z-score (most extreme first) and only mark the most extreme outliers
            if outliers_found:
                outliers_found.sort(key=lambda x: x[1], reverse=True)
                
                # More conservative: only mark top 10% of potential outliers and max 2 per sector
                max_outliers = min(2, max(1, len(values) // 10))  # At most 10% or 2 stocks
                confirmed_outliers = []
                
                for i, (idx, z_score, value) in enumerate(outliers_found):
                    if i >= max_outliers:
                        break
                    
                    # Only mark if z-score is really extreme (>4.0) or if it's the most extreme
                    if z_score > 4.0 or (i == 0 and z_score > z_threshold):
                        confirmed_outliers.append((idx, z_score, value))
                
                # Mark confirmed outliers
                for idx, z_score, value in confirmed_outliers:
                    df.loc[idx, 'is_outlier'] = True
        
        return df
    
    def prepare_data_for_modeling(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
        """
        Complete data preparation pipeline for modeling
        """
        # Validate input data
        if not self.validate_data_for_analysis(data, selected_factors):
            return None
        
        # Step 1: Calculate z-scores by sector
        df = self.calculate_z_scores_by_sector(data)
        
        # Step 2: Calculate factor scores
        df = self.calculate_factor_scores(df, selected_factors)
        
        # Step 3: Identify outliers using 3 standard deviations as specified
        # Calculate mean and std dev for P/E ratios
        pe_mean = df['pe_ratio'].mean()
        pe_std = df['pe_ratio'].std()
        
        # Mark outliers as companies 3 standard deviations above or below the mean
        df['is_outlier'] = (
            (df['pe_ratio'] < pe_mean - 3 * pe_std) |
            (df['pe_ratio'] > pe_mean + 3 * pe_std)
        )
        
        # Step 4: Remove rows with missing P/E ratios or negative P/E ratios
        df = df[df['pe_ratio'].notna()]
        df = df[df['pe_ratio'] > 0]
        
        # Step 5: Get factor score columns
        factor_cols = self.get_factor_score_columns(selected_factors)
        
        # Step 6: Remove rows where all factor scores are NaN
        if factor_cols:
            # Only check for columns that actually exist in the DataFrame
            existing_factor_cols = [col for col in factor_cols if col in df.columns]
            if existing_factor_cols:
                df = df[df[existing_factor_cols].notna().any(axis=1)]
            else:
                # If no factor score columns exist, return empty DataFrame
                return None
        
        # Return ALL data (including outliers) with outlier flags
        # The model training will decide which data to use for regression
        return df
    
    def prepare_data_for_individual_analysis(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
        """
        Prepare data for individual stock analysis - includes outliers for comprehensive analysis
        """
        # Validate input data
        if not self.validate_data_for_analysis(data, selected_factors):
            return None
        
        df = data.copy()
        
        # Step 1: Calculate factor scores for all stocks
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector]
            
            if len(sector_data) < 3:
                continue
            
            # Calculate factor scores for this sector
            for factor_name, metrics in selected_factors.items():
                if factor_name in self.factor_mappings:
                    factor_score = self.calculate_factor_score(sector_data, factor_name, metrics)
                    df.loc[df['sector'] == sector, f'{factor_name}_score'] = factor_score
        
        # Step 2: Mark outliers but don't exclude them - use 2.5 std dev method
        df = self.filter_outliers_iterative_25std(df, column='pe_ratio')
        
        # Step 3: Remove rows with missing P/E ratios or negative P/E ratios
        df = df[df['pe_ratio'].notna()]
        df = df[df['pe_ratio'] > 0]
        
        # Step 4: Get factor score columns
        factor_cols = self.get_factor_score_columns(selected_factors)
        
        # Step 5: Remove rows where all factor scores are NaN
        if factor_cols:
            # Only check for columns that actually exist in the DataFrame
            existing_factor_cols = [col for col in factor_cols if col in df.columns]
            if existing_factor_cols:
                df = df[df[existing_factor_cols].notna().any(axis=1)]
            else:
                # If no factor score columns exist, return empty DataFrame
                return None
        
        return df

    def calculate_individual_stock_factor_scores(self, stock_data: Dict, sector_data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> Optional[Dict]:
        """
        Calculate factor scores for an individual stock by comparing it to sector peers
        """
        try:
            if not stock_data or 'sector' not in stock_data:
                return None
            
            sector = stock_data['sector']
            
            # Filter sector data to the same sector
            sector_peers = sector_data[sector_data['sector'] == sector].copy()
            
            if len(sector_peers) < 3:
                return None
            
            # Calculate z-scores for sector data first
            sector_with_zscores = self.calculate_z_scores_by_sector(sector_peers)
            
            # Calculate factor scores for the individual stock
            factor_scores = {}
            
            for factor_name, metrics in selected_factors.items():
                if factor_name in self.factor_mappings and metrics:
                    # Calculate factor scores for sector peers using z-scores
                    sector_factor_scores = self.calculate_factor_score(sector_with_zscores, factor_name, metrics)
                    
                    if sector_factor_scores is not None and len(sector_factor_scores) > 0:
                        # Calculate the individual stock's factor score using same method
                        individual_factor_scores = []
                        reverse_factors = self.factor_mappings.get(factor_name, {}).get('reverse', [])
                        
                        for metric in metrics:
                            if metric in stock_data and stock_data[metric] is not None:
                                stock_value = float(stock_data[metric])
                                
                                # Calculate z-score relative to sector
                                if metric in sector_peers.columns:
                                    sector_values = sector_peers[metric].dropna()
                                    if len(sector_values) > 0:
                                        mean_val = sector_values.mean()
                                        std_val = sector_values.std()
                                        
                                        if std_val > 0:
                                            z_score = (stock_value - mean_val) / std_val
                                            
                                            # Apply reverse scoring if needed
                                            if metric in reverse_factors:
                                                z_score = -z_score
                                            
                                            individual_factor_scores.append(z_score)
                        
                        if individual_factor_scores:
                            # Average the z-scores for this factor
                            factor_score = np.mean(individual_factor_scores)
                            
                            # Apply Risk Aversion negation
                            if factor_name == 'Risk Aversion':
                                factor_score *= -1
                            
                            factor_scores[f'{factor_name}_factor_score'] = factor_score
            
            return factor_scores if factor_scores else None
            
        except Exception as e:
            print(f"Error calculating individual factor scores: {str(e)}")
            return None

    
    def get_available_metrics(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get list of available metrics in the data for each factor category
        """
        available_metrics = {}
        
        for factor_name, metrics in self.factor_mappings.items():
            available_for_factor = []
            for metric_key, metric_col in metrics.items():
                if metric_col in data.columns and not data[metric_col].isna().all():
                    available_for_factor.append(metric_key)
            available_metrics[factor_name] = available_for_factor
        
        return available_metrics
    
    def calculate_correlation_matrix(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Calculate correlation matrix between factor scores and P/E ratios
        """
        factor_cols = self.get_factor_score_columns(selected_factors)
        
        if not factor_cols:
            return pd.DataFrame()
        
        # Include P/E ratio in correlation analysis
        correlation_cols = factor_cols + ['pe_ratio']
        available_cols = [col for col in correlation_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return pd.DataFrame()
        
        return data[available_cols].corr()
    
    def get_factor_summary_stats(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> Dict:
        """
        Get summary statistics for each factor
        """
        factor_cols = self.get_factor_score_columns(selected_factors)
        summary_stats = {}
        
        for factor_col in factor_cols:
            if factor_col in data.columns:
                factor_data = data[factor_col].dropna()
                if len(factor_data) > 0:
                    summary_stats[factor_col] = {
                        'mean': factor_data.mean(),
                        'std': factor_data.std(),
                        'min': factor_data.min(),
                        'max': factor_data.max(),
                        'count': len(factor_data)
                    }
        
        return summary_stats
    
    def calculate_factor_score(self, data: pd.DataFrame, factor_name: str, selected_metrics: List[str]) -> pd.Series:
        """
        Calculate factor score for a single factor using z-scores of component metrics
        """
        if not selected_metrics:
            return pd.Series(0, index=data.index)
        
        # Get the z-score columns for selected metrics
        zscore_cols = []
        for metric in selected_metrics:
            zscore_col = f'{metric}_zscore'
            if zscore_col in data.columns:
                zscore_cols.append(zscore_col)
        
        if zscore_cols:
            # Calculate mean z-score across selected metrics
            factor_score = data[zscore_cols].mean(axis=1, skipna=True)
            
            # For Risk Aversion, multiply by -1 (higher risk should be negative)
            if factor_name == 'Risk Aversion':
                factor_score *= -1
        else:
            # If no valid metrics, set factor score to 0
            factor_score = pd.Series(0, index=data.index)
        
        return factor_score