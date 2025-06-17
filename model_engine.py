import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
from scipy import stats
import warnings
from typing import Dict, List, Optional, Tuple
from factor_calculator import FactorCalculator

warnings.filterwarnings('ignore')

class ModelEngine:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Ridge regression regularization parameter
        self.factor_calculator = FactorCalculator()
        self.models = {}  # Store models for each sector
        self.scalers = {}  # Store scalers for each sector
        self.sector_stats = {}  # Store sector statistics
    
    def calculate_slope_significance(self, X: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> Dict:
        """
        Calculate statistical significance of slope coefficient at given alpha level (default p=0.1)
        Returns p-value, t-statistic, and significance flag
        """
        try:
            n = len(X)
            if n < 3:
                return {'p_value': np.nan, 't_statistic': np.nan, 'is_significant': False}
            
            # Calculate slope and residuals
            X_mean = np.mean(X)
            y_mean = np.mean(y)
            
            # Calculate slope using least squares
            numerator = np.sum((X - X_mean) * (y - y_mean))
            denominator = np.sum((X - X_mean) ** 2)
            
            if denominator == 0:
                return {'p_value': np.nan, 't_statistic': np.nan, 'is_significant': False}
            
            slope = numerator / denominator
            intercept = y_mean - slope * X_mean
            
            # Calculate predicted values and residuals
            y_pred = slope * X + intercept
            residuals = y - y_pred
            
            # Calculate standard error of slope
            mse = np.sum(residuals ** 2) / (n - 2)  # degrees of freedom = n - 2
            se_slope = np.sqrt(mse / np.sum((X - X_mean) ** 2))
            
            # Calculate t-statistic (testing H0: slope = 0)
            t_statistic = slope / se_slope
            
            # Calculate p-value (two-tailed test)
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-2))
            
            # Check if significant at given alpha level
            is_significant = p_value < alpha
            
            return {
                'p_value': p_value,
                't_statistic': t_statistic,
                'is_significant': is_significant,
                'slope': slope,
                'se_slope': se_slope,
                'degrees_freedom': n - 2
            }
            
        except Exception as e:
            print(f"Error calculating slope significance: {str(e)}")
            return {'p_value': np.nan, 't_statistic': np.nan, 'is_significant': False}

    def optimize_factor_weights_for_positive_correlation(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Optimize factor weights to ensure positive correlation between fundamental score and P/E ratio
        """
        n_factors = X.shape[1]
        
        def objective(weights):
            # Ensure weights sum to 1 and are non-negative
            weights = np.abs(weights)
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_factors) / n_factors
            
            # Calculate weighted fundamental score
            fundamental_score = np.sum(X.values * weights, axis=1)
            
            # Calculate correlation with P/E ratio
            correlation = np.corrcoef(fundamental_score, y)[0, 1]
            
            # We want to maximize positive correlation, so minimize negative correlation
            # Add penalty for negative correlation
            if correlation < 0:
                return -correlation + 10  # Heavy penalty for negative correlation
            else:
                return -correlation  # Maximize positive correlation
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_factors) / n_factors
        
        # Constraints: weights must be non-negative and sum to 1
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1},  # Sum to 1
        ]
        
        bounds = [(0, 1) for _ in range(n_factors)]  # Each weight between 0 and 1
        
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = np.abs(result.x)
                optimized_weights = optimized_weights / np.sum(optimized_weights)
                return optimized_weights
            else:
                # Fallback to Ridge regression if optimization fails
                ridge_model = Ridge(alpha=self.alpha, random_state=42)
                ridge_model.fit(X, y)
                raw_weights = np.abs(ridge_model.coef_)
                return raw_weights / np.sum(raw_weights) if np.sum(raw_weights) > 0 else initial_weights
        except:
            # Fallback to equal weights if optimization fails
            return initial_weights
    
    def train_sector_model(self, data: pd.DataFrame, sector: str, selected_factors: Dict[str, List[str]]) -> Optional[Dict]:
        """
        Train a model with optimized factor weights for positive correlation with P/E ratio
        """
        # Filter data for the sector
        sector_data = data[data['sector'] == sector].copy()
        
        if len(sector_data) < 5:  # Need minimum data for training
            return None
        
        # Prepare data using factor calculator
        prepared_data = self.factor_calculator.prepare_data_for_modeling(sector_data, selected_factors)
        
        if prepared_data is None or len(prepared_data) < 5:
            return None
        
        # Get factor score columns
        factor_cols = self.factor_calculator.get_factor_score_columns(selected_factors)
        
        if not factor_cols:
            return None
        
        # Prepare features and target
        # Only use columns that actually exist in the DataFrame
        existing_factor_cols = [col for col in factor_cols if col in prepared_data.columns]
        if not existing_factor_cols:
            return None
        
        X = prepared_data[existing_factor_cols].fillna(0)  # Fill NaN with 0 for missing factor scores
        y = prepared_data['pe_ratio']
        
        # Remove any remaining NaN values
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 5:  # Need minimum data after cleaning
            return None
        
        try:
            # Step 1: Optimize factor weights for positive correlation
            optimized_weights = self.optimize_factor_weights_for_positive_correlation(X, y)
            
            # Step 2: Create weighted fundamental score
            fundamental_scores = np.sum(X.values * optimized_weights, axis=1)
            
            # Step 3: Run linear regression between weighted fundamental score and P/E ratios
            final_model = LinearRegression()
            fundamental_scores_reshaped = fundamental_scores.reshape(-1, 1)
            final_model.fit(fundamental_scores_reshaped, y)
            
            # Make predictions using the linear model
            y_pred = final_model.predict(fundamental_scores_reshaped)
            
            # Calculate performance metrics
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            
            # Calculate correlation to verify it's positive
            correlation = np.corrcoef(fundamental_scores, y)[0, 1]
            
            # Calculate significance test for slope at p = 0.1
            significance_test = self.calculate_slope_significance(fundamental_scores, y.values, alpha=0.1)
            
            # Calculate residuals for fair value line
            residuals = y - y_pred
            residual_std = residuals.std()
            
            # Get the linear equation parameters
            slope = final_model.coef_[0]
            intercept = final_model.intercept_
            
            # Store model components
            model_info = {
                'final_model': final_model,
                'factor_columns': existing_factor_cols,
                'factor_weights': dict(zip(existing_factor_cols, optimized_weights)),
                'r2_score': r2,
                'mse': mse,
                'residual_std': residual_std,
                'n_samples': len(X),
                'slope': slope,
                'intercept': intercept,
                'correlation': correlation,
                'equation': f'P/E = {slope:.4f} × Fundamental_Score + {intercept:.4f}',
                'pe_mean': y.mean(),
                'pe_std': y.std(),
                'slope_significance': significance_test
            }
            
            # Add predictions to prepared data
            prepared_data = prepared_data[valid_mask].copy()
            prepared_data['predicted_pe'] = y_pred
            prepared_data['residuals'] = residuals
            prepared_data['fundamental_zscore'] = fundamental_scores
            
            model_info['data'] = prepared_data
            
            return model_info
            
        except Exception as e:
            print(f"Error training model for sector {sector}: {str(e)}")
            return None
    
    def predict_pe_ratio(self, factor_scores: Dict[str, float], sector: str) -> Optional[float]:
        """
        Predict P/E ratio for given factor scores in a specific sector
        """
        if sector not in self.models:
            return None
        
        model_info = self.models[sector]
        final_model = model_info['final_model']
        factor_cols = model_info['factor_columns']
        weights = model_info['factor_weights']
        
        try:
            # Calculate weighted fundamental score
            fundamental_score = 0
            for i, factor_col in enumerate(factor_cols):
                factor_value = factor_scores.get(factor_col.replace('_factor_score', ''), 0)
                weight = weights[factor_col]
                fundamental_score += factor_value * weight
            
            # Predict using linear model
            prediction = final_model.predict([[fundamental_score]])[0]
            return prediction
        except Exception as e:
            print(f"Error predicting for sector {sector}: {str(e)}")
            return None
    
    def analyze_sector(self, data: pd.DataFrame, sector: str, selected_factors: Dict[str, List[str]]) -> Optional[pd.DataFrame]:
        """
        Perform complete sector analysis including model training and predictions
        """
        try:
            # Train model for this sector
            model_info = self.train_sector_model(data, sector, selected_factors)
            
            if model_info is None:
                return None
            
            # Store model for future use
            self.models[sector] = model_info
            
            # Return the prepared data with predictions
            result_data = model_info['data'].copy()
            
            # For plotting, also prepare data that includes outliers for visualization
            plot_data = self.factor_calculator.prepare_data_for_individual_analysis(data[data['sector'] == sector], selected_factors)
            if plot_data is not None:
                # Add predictions for all data points (including outliers)
                factor_cols = self.factor_calculator.get_factor_score_columns(selected_factors)
                if factor_cols:
                    existing_factor_cols = [col for col in factor_cols if col in plot_data.columns]
                    if existing_factor_cols:
                        X_plot = plot_data[existing_factor_cols].fillna(0)
                        plot_data['predicted_pe'] = model_info['final_model'].predict(X_plot)
                        plot_data['residuals'] = plot_data['pe_ratio'] - plot_data['predicted_pe']
                        plot_data['fundamental_zscore'] = self.calculate_fundamental_zscore(plot_data, selected_factors)
                
                result_data = plot_data  # Use the full dataset for plotting
            
            # Add model performance info
            result_data.attrs['model_info'] = {
                'r2_score': model_info['r2_score'],
                'mse': model_info['mse'],  
                'residual_std': model_info['residual_std'],
                'n_samples': model_info['n_samples'],
                'factor_weights': model_info['factor_weights'],
                'equation': model_info['equation'],
                'slope': model_info['slope'],
                'intercept': model_info['intercept'],
                'correlation': model_info.get('correlation', 0),
                'slope_significance': model_info.get('slope_significance', {})
            }
            
            return result_data
            
        except Exception as e:
            print(f"Error analyzing sector {sector}: {str(e)}")
            return None
    
    def analyze_individual_stock(self, market_data: pd.DataFrame, stock_data: Dict, selected_factors: Dict[str, List[str]]) -> Optional[Dict]:
        """
        Analyze an individual stock by comparing it to its sector
        """
        try:
            if not stock_data or 'sector' not in stock_data:
                return None
            
            sector = stock_data['sector']
            symbol = stock_data['symbol']
            
            # Get sector data and train model
            sector_results = self.analyze_sector(market_data, sector, selected_factors)
            
            if sector_results is None:
                return None
            
            # Calculate factor scores for the individual stock by comparing to sector peers
            sector_data = market_data[market_data['sector'] == sector]
            individual_factor_scores = self.factor_calculator.calculate_individual_stock_factor_scores(
                stock_data, sector_data, selected_factors
            )
            
            if individual_factor_scores is None:
                return None
            
            # Get factor score columns
            factor_cols = self.factor_calculator.get_factor_score_columns(selected_factors)
            
            if not factor_cols:
                return None
            
            # Calculate predictions for the stock
            model_info = self.models[sector]
            final_model = model_info['final_model']
            weights = model_info['factor_weights']
            
            # Prepare features using the calculated factor scores
            existing_factor_cols = [col for col in factor_cols if col in individual_factor_scores]
            if not existing_factor_cols:
                return None
            
            # Calculate weighted fundamental score
            fundamental_zscore = 0
            factor_zscores = {}
            for factor_col in existing_factor_cols:
                factor_value = individual_factor_scores[factor_col]
                weight = weights.get(factor_col, 0)  # Use get() to handle missing weights
                fundamental_zscore += factor_value * weight
                # Store individual factor z-scores for display
                factor_name = factor_col.replace('_factor_score', '')
                factor_zscores[factor_name] = factor_value
            
            # Predict using linear model
            predicted_pe = final_model.predict([[fundamental_zscore]])[0]
            
            # Prepare results
            results = {
                'symbol': symbol,
                'sector': sector,
                'actual_pe': stock_data.get('pe_ratio', np.nan),
                'predicted_pe': predicted_pe,
                'fundamental_zscore': fundamental_zscore,
                'difference': stock_data.get('pe_ratio', np.nan) - predicted_pe,
                'factor_zscores': factor_zscores,
                'sector_plot_data': sector_results,
                'model_performance': {
                    'r2_score': model_info['r2_score'],
                    'mse': model_info['mse'],
                    'n_samples': model_info['n_samples'],
                    'correlation': model_info.get('correlation', 0),
                    'equation': model_info['equation'],
                    'factor_weights': model_info['factor_weights'],
                    'slope_significance': model_info.get('slope_significance', {})
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error analyzing individual stock: {str(e)}")
            return None
    
    def get_sector_model_performance(self, sector: str) -> Optional[Dict]:
        """
        Get performance metrics for a sector model
        """
        if sector in self.models:
            model_info = self.models[sector]
            return {
                'r2_score': model_info['r2_score'],
                'mse': model_info['mse'],
                'residual_std': model_info['residual_std'],
                'n_samples': model_info['n_samples'],
                'factor_weights': model_info['factor_weights'],
                'equation': model_info['equation'],
                'slope': model_info['slope'],
                'intercept': model_info['intercept'],
                'correlation': model_info.get('correlation', 0),
                'slope_significance': model_info.get('slope_significance', {})
            }
        return None
    
    def get_all_sector_models(self) -> Dict:
        """
        Get information about all trained sector models
        """
        return {sector: self.get_sector_model_performance(sector) for sector in self.models.keys()}
    
    def retrain_all_models(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> Dict[str, bool]:
        """
        Retrain models for all sectors with new factor selection
        """
        # Clear existing models
        self.models.clear()
        self.scalers.clear()
        
        results = {}
        sectors = data['sector'].unique()
        
        for sector in sectors:
            try:
                model_info = self.train_sector_model(data, sector, selected_factors)
                if model_info is not None:
                    self.models[sector] = model_info
                    results[sector] = True
                else:
                    results[sector] = False
            except Exception as e:
                print(f"Error retraining model for sector {sector}: {str(e)}")
                results[sector] = False
        
        return results
    
    def calculate_fair_value_statistics(self, sector_data: pd.DataFrame) -> Dict:
        """
        Calculate fair value line statistics for a sector
        """
        if 'predicted_pe' not in sector_data.columns or 'pe_ratio' not in sector_data.columns:
            return {}
        
        residuals = sector_data['pe_ratio'] - sector_data['predicted_pe']
        
        return {
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'mean_absolute_error': np.abs(residuals).mean(),
            'r_squared': np.corrcoef(sector_data['pe_ratio'], sector_data['predicted_pe'])[0, 1] ** 2 if len(sector_data) > 1 else 0
        }
    
    def calculate_fundamental_zscore(self, data: pd.DataFrame, selected_factors: Dict[str, List[str]]) -> np.ndarray:
        """
        Calculate fundamental z-score using optimized factor weights
        """
        # Get factor score columns
        factor_cols = self.factor_calculator.get_factor_score_columns(selected_factors)
        
        if not factor_cols:
            return np.zeros(len(data))
        
        # Prepare features
        existing_factor_cols = [col for col in factor_cols if col in data.columns]
        if not existing_factor_cols:
            return np.zeros(len(data))
        X = data[existing_factor_cols].fillna(0)
        
        if len(X) == 0:
            return np.zeros(len(data))
        
        # Use equal weights if no optimization has been done
        weights = np.ones(len(existing_factor_cols)) / len(existing_factor_cols)
        
        # Calculate weighted fundamental score
        fundamental_scores = np.sum(X.values * weights, axis=1)
        
        return fundamental_scores