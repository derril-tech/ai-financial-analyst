"""Risk analytics and factor models."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from app.core.observability import trace_function


@dataclass
class RiskMetrics:
    """Risk metrics for a security or portfolio."""
    symbol: str
    period_start: date
    period_end: date
    
    # Return metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    
    # Risk metrics
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    max_drawdown: float
    
    # Factor exposures
    beta: Optional[float] = None
    alpha: Optional[float] = None
    r_squared: Optional[float] = None
    
    # Factor loadings (if factor model applied)
    factor_loadings: Optional[Dict[str, float]] = None


@dataclass
class FactorModelResult:
    """Factor model regression result."""
    symbol: str
    model_type: str  # CAPM, FF3, FF5, etc.
    
    # Regression statistics
    alpha: float
    beta: float
    r_squared: float
    adjusted_r_squared: float
    
    # Factor loadings
    factor_loadings: Dict[str, float]
    
    # Statistical significance
    p_values: Dict[str, float]
    t_stats: Dict[str, float]
    
    # Residual analysis
    residual_std: float
    jarque_bera_stat: float
    jarque_bera_p_value: float


class RiskAnalytics:
    """Risk analytics and factor model implementation."""
    
    def __init__(self) -> None:
        """Initialize risk analytics."""
        self.trading_days_per_year = 252
        self.risk_free_rate = 0.02  # Default 2% risk-free rate
    
    @trace_function("risk_analytics.calculate_risk_metrics")
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: Optional[float] = None,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** self.trading_days_per_year - 1
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Sharpe ratio
        excess_returns = returns - risk_free_rate / self.trading_days_per_year
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Maximum Drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Beta and alpha (if benchmark provided)
        beta = None
        alpha = None
        r_squared = None
        
        if benchmark_returns is not None:
            # Align returns
            aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
            if len(aligned_returns) > 10:
                asset_returns = aligned_returns.iloc[:, 0]
                market_returns = aligned_returns.iloc[:, 1]
                
                # Calculate beta
                covariance = np.cov(asset_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / market_variance
                
                # Calculate alpha
                alpha = asset_returns.mean() - beta * market_returns.mean()
                
                # R-squared
                correlation = np.corrcoef(asset_returns, market_returns)[0, 1]
                r_squared = correlation ** 2
        
        return RiskMetrics(
            symbol=returns.name or "Unknown",
            period_start=returns.index[0].date() if len(returns) > 0 else date.today(),
            period_end=returns.index[-1].date() if len(returns) > 0 else date.today(),
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
        )
    
    @trace_function("risk_analytics.capm_regression")
    def capm_regression(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        risk_free_rate: Optional[float] = None,
    ) -> FactorModelResult:
        """Perform CAPM regression analysis."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate / self.trading_days_per_year
        
        # Align returns and calculate excess returns
        aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
        asset_excess = aligned_returns.iloc[:, 0] - risk_free_rate
        market_excess = aligned_returns.iloc[:, 1] - risk_free_rate
        
        # Regression
        X = market_excess.values.reshape(-1, 1)
        y = asset_excess.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Statistics
        alpha = model.intercept_
        beta = model.coef_[0]
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Adjusted R-squared
        n = len(y)
        p = 1  # number of predictors
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        # T-statistics and p-values
        mse = ss_res / (n - p - 1)
        var_beta = mse / np.sum((X.flatten() - np.mean(X)) ** 2)
        se_beta = np.sqrt(var_beta)
        t_stat_beta = beta / se_beta
        p_value_beta = 2 * (1 - stats.t.cdf(np.abs(t_stat_beta), n - p - 1))
        
        # Alpha statistics (approximate)
        se_alpha = np.sqrt(mse * (1/n + np.mean(X)**2 / np.sum((X.flatten() - np.mean(X))**2)))
        t_stat_alpha = alpha / se_alpha
        p_value_alpha = 2 * (1 - stats.t.cdf(np.abs(t_stat_alpha), n - p - 1))
        
        # Jarque-Bera test for normality of residuals
        jb_stat, jb_p_value = stats.jarque_bera(residuals)
        
        return FactorModelResult(
            symbol=asset_returns.name or "Unknown",
            model_type="CAPM",
            alpha=alpha * self.trading_days_per_year,  # Annualize
            beta=beta,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            factor_loadings={"market": beta},
            p_values={"alpha": p_value_alpha, "market": p_value_beta},
            t_stats={"alpha": t_stat_alpha, "market": t_stat_beta},
            residual_std=np.std(residuals),
            jarque_bera_stat=jb_stat,
            jarque_bera_p_value=jb_p_value,
        )
    
    @trace_function("risk_analytics.fama_french_3_factor")
    def fama_french_3_factor(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        smb_returns: pd.Series,  # Small Minus Big
        hml_returns: pd.Series,  # High Minus Low
        risk_free_rate: Optional[float] = None,
    ) -> FactorModelResult:
        """Perform Fama-French 3-factor model regression."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate / self.trading_days_per_year
        
        # Align all returns
        all_returns = pd.concat([
            asset_returns, market_returns, smb_returns, hml_returns
        ], axis=1).dropna()
        
        asset_excess = all_returns.iloc[:, 0] - risk_free_rate
        market_excess = all_returns.iloc[:, 1] - risk_free_rate
        smb = all_returns.iloc[:, 2]
        hml = all_returns.iloc[:, 3]
        
        # Regression: R_i - R_f = α + β(R_m - R_f) + s*SMB + h*HML + ε
        X = np.column_stack([market_excess, smb, hml])
        y = asset_excess.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Extract coefficients
        alpha = model.intercept_
        beta_market, beta_smb, beta_hml = model.coef_
        
        # Statistics
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        n = len(y)
        p = 3  # number of predictors
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        # Standard errors and t-statistics (simplified)
        mse = ss_res / (n - p - 1)
        
        # Jarque-Bera test
        jb_stat, jb_p_value = stats.jarque_bera(residuals)
        
        return FactorModelResult(
            symbol=asset_returns.name or "Unknown",
            model_type="Fama-French 3-Factor",
            alpha=alpha * self.trading_days_per_year,
            beta=beta_market,
            r_squared=r_squared,
            adjusted_r_squared=adjusted_r_squared,
            factor_loadings={
                "market": beta_market,
                "smb": beta_smb,
                "hml": beta_hml,
            },
            p_values={},  # Would need more complex calculation
            t_stats={},   # Would need more complex calculation
            residual_std=np.std(residuals),
            jarque_bera_stat=jb_stat,
            jarque_bera_p_value=jb_p_value,
        )
    
    @trace_function("risk_analytics.rolling_beta")
    def rolling_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series,
        window: int = 60,  # 60 trading days (3 months)
    ) -> pd.Series:
        """Calculate rolling beta over time."""
        # Align returns
        aligned_returns = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        def calculate_beta(window_data):
            if len(window_data) < 10:  # Minimum observations
                return np.nan
            
            asset_ret = window_data.iloc[:, 0]
            market_ret = window_data.iloc[:, 1]
            
            covariance = np.cov(asset_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            if market_variance == 0:
                return np.nan
            
            return covariance / market_variance
        
        rolling_betas = aligned_returns.rolling(window=window).apply(
            lambda x: calculate_beta(x), raw=False
        )
        
        return rolling_betas.iloc[:, 0]  # Return only the beta series
    
    @trace_function("risk_analytics.calculate_var_cvar")
    def calculate_var_cvar(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99],
        method: str = "historical",  # historical, parametric, monte_carlo
    ) -> Dict[str, Dict[str, float]]:
        """Calculate Value at Risk and Conditional VaR."""
        results = {}
        
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            
            if method == "historical":
                # Historical simulation
                var = np.percentile(returns, alpha * 100)
                cvar = returns[returns <= var].mean()
            
            elif method == "parametric":
                # Parametric (normal distribution assumption)
                mean_return = returns.mean()
                std_return = returns.std()
                var = stats.norm.ppf(alpha, mean_return, std_return)
                
                # CVaR for normal distribution
                phi = stats.norm.pdf(stats.norm.ppf(alpha))
                cvar = mean_return - std_return * phi / alpha
            
            elif method == "monte_carlo":
                # Monte Carlo simulation
                mean_return = returns.mean()
                std_return = returns.std()
                
                # Generate random returns
                np.random.seed(42)  # For reproducibility
                simulated_returns = np.random.normal(mean_return, std_return, 10000)
                
                var = np.percentile(simulated_returns, alpha * 100)
                cvar = simulated_returns[simulated_returns <= var].mean()
            
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            results[f"{confidence_level:.0%}"] = {
                "var": var,
                "cvar": cvar,
            }
        
        return results
    
    @trace_function("risk_analytics.regime_analysis")
    def regime_analysis(
        self,
        returns: pd.Series,
        market_returns: pd.Series,
        volatility_threshold: float = 0.02,  # 2% daily volatility threshold
    ) -> Dict[str, Any]:
        """Analyze performance across different market regimes."""
        # Calculate rolling volatility
        rolling_vol = market_returns.rolling(window=20).std()
        
        # Define regimes
        high_vol_periods = rolling_vol > volatility_threshold
        low_vol_periods = rolling_vol <= volatility_threshold
        
        # Calculate statistics for each regime
        high_vol_returns = returns[high_vol_periods]
        low_vol_returns = returns[low_vol_periods]
        
        results = {
            "high_volatility_regime": {
                "mean_return": high_vol_returns.mean(),
                "volatility": high_vol_returns.std(),
                "sharpe_ratio": high_vol_returns.mean() / high_vol_returns.std() if high_vol_returns.std() > 0 else 0,
                "observations": len(high_vol_returns),
            },
            "low_volatility_regime": {
                "mean_return": low_vol_returns.mean(),
                "volatility": low_vol_returns.std(),
                "sharpe_ratio": low_vol_returns.mean() / low_vol_returns.std() if low_vol_returns.std() > 0 else 0,
                "observations": len(low_vol_returns),
            },
        }
        
        # Calculate regime-specific betas
        if len(high_vol_returns) > 10:
            high_vol_market = market_returns[high_vol_periods]
            aligned_high = pd.concat([high_vol_returns, high_vol_market], axis=1).dropna()
            if len(aligned_high) > 10:
                beta_high = np.cov(aligned_high.iloc[:, 0], aligned_high.iloc[:, 1])[0, 1] / np.var(aligned_high.iloc[:, 1])
                results["high_volatility_regime"]["beta"] = beta_high
        
        if len(low_vol_returns) > 10:
            low_vol_market = market_returns[low_vol_periods]
            aligned_low = pd.concat([low_vol_returns, low_vol_market], axis=1).dropna()
            if len(aligned_low) > 10:
                beta_low = np.cov(aligned_low.iloc[:, 0], aligned_low.iloc[:, 1])[0, 1] / np.var(aligned_low.iloc[:, 1])
                results["low_volatility_regime"]["beta"] = beta_low
        
        return results
