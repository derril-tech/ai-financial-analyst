"""Market data adapters with rate limiting and multiple providers."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, date, timedelta
from dataclasses import dataclass

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.observability import trace_function


@dataclass
class MarketDataPoint:
    """Market data point."""
    symbol: str
    date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None


@dataclass
class FundamentalData:
    """Fundamental data point."""
    symbol: str
    period: str
    metric: str
    value: Optional[float]
    unit: Optional[str] = None
    date: Optional[date] = None


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def get_price_data(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> List[MarketDataPoint]:
        """Get historical price data."""
        pass
    
    @abstractmethod
    async def get_fundamentals(
        self, 
        symbol: str, 
        metrics: List[str]
    ) -> List[FundamentalData]:
        """Get fundamental data."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        pass


class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage market data provider."""
    
    def __init__(self, api_key: str) -> None:
        """Initialize Alpha Vantage provider."""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = 5  # requests per minute
        self.last_request_time = 0
        
    async def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """Make rate-limited API request."""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 60 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        params["apikey"] = self.api_key
        
        async with httpx.AsyncClient() as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_price_data(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> List[MarketDataPoint]:
        """Get historical price data from Alpha Vantage."""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
        }
        
        try:
            data = await self._make_request(params)
            time_series = data.get("Time Series (Daily)", {})
            
            price_data = []
            for date_str, values in time_series.items():
                price_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                
                if start_date <= price_date <= end_date:
                    point = MarketDataPoint(
                        symbol=symbol,
                        date=price_date,
                        open=float(values["1. open"]),
                        high=float(values["2. high"]),
                        low=float(values["3. low"]),
                        close=float(values["4. close"]),
                        adjusted_close=float(values["5. adjusted close"]),
                        volume=int(values["6. volume"]),
                    )
                    price_data.append(point)
            
            # Sort by date
            price_data.sort(key=lambda x: x.date)
            return price_data
            
        except Exception as e:
            print(f"Alpha Vantage price data request failed: {e}")
            return []
    
    async def get_fundamentals(
        self, 
        symbol: str, 
        metrics: List[str]
    ) -> List[FundamentalData]:
        """Get fundamental data from Alpha Vantage."""
        # Alpha Vantage provides fundamental data through different endpoints
        fundamental_data = []
        
        # Get income statement
        if any(metric in ["revenue", "net_income", "eps"] for metric in metrics):
            income_data = await self._get_income_statement(symbol)
            fundamental_data.extend(income_data)
        
        # Get balance sheet
        if any(metric in ["total_assets", "total_debt", "cash"] for metric in metrics):
            balance_data = await self._get_balance_sheet(symbol)
            fundamental_data.extend(balance_data)
        
        return fundamental_data
    
    async def _get_income_statement(self, symbol: str) -> List[FundamentalData]:
        """Get income statement data."""
        params = {
            "function": "INCOME_STATEMENT",
            "symbol": symbol,
        }
        
        try:
            data = await self._make_request(params)
            annual_reports = data.get("annualReports", [])
            
            fundamental_data = []
            for report in annual_reports[:5]:  # Last 5 years
                fiscal_date = datetime.strptime(report["fiscalDateEnding"], "%Y-%m-%d").date()
                
                # Map Alpha Vantage fields to our metrics
                metrics_map = {
                    "totalRevenue": "revenue",
                    "netIncome": "net_income",
                    "operatingIncome": "operating_income",
                }
                
                for av_field, our_metric in metrics_map.items():
                    if av_field in report and report[av_field] != "None":
                        try:
                            value = float(report[av_field])
                            fundamental_data.append(FundamentalData(
                                symbol=symbol,
                                period="annual",
                                metric=our_metric,
                                value=value,
                                unit="USD",
                                date=fiscal_date,
                            ))
                        except (ValueError, TypeError):
                            continue
            
            return fundamental_data
            
        except Exception as e:
            print(f"Alpha Vantage income statement request failed: {e}")
            return []
    
    async def _get_balance_sheet(self, symbol: str) -> List[FundamentalData]:
        """Get balance sheet data."""
        params = {
            "function": "BALANCE_SHEET",
            "symbol": symbol,
        }
        
        try:
            data = await self._make_request(params)
            annual_reports = data.get("annualReports", [])
            
            fundamental_data = []
            for report in annual_reports[:5]:  # Last 5 years
                fiscal_date = datetime.strptime(report["fiscalDateEnding"], "%Y-%m-%d").date()
                
                # Map Alpha Vantage fields to our metrics
                metrics_map = {
                    "totalAssets": "total_assets",
                    "totalLiabilities": "total_liabilities",
                    "totalShareholderEquity": "total_equity",
                    "cashAndCashEquivalentsAtCarryingValue": "cash",
                }
                
                for av_field, our_metric in metrics_map.items():
                    if av_field in report and report[av_field] != "None":
                        try:
                            value = float(report[av_field])
                            fundamental_data.append(FundamentalData(
                                symbol=symbol,
                                period="annual",
                                metric=our_metric,
                                value=value,
                                unit="USD",
                                date=fiscal_date,
                            ))
                        except (ValueError, TypeError):
                            continue
            
            return fundamental_data
            
        except Exception as e:
            print(f"Alpha Vantage balance sheet request failed: {e}")
            return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Alpha Vantage."""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        }
        
        try:
            data = await self._make_request(params)
            quote = data.get("Global Quote", {})
            
            if "05. price" in quote:
                return float(quote["05. price"])
            
            return None
            
        except Exception as e:
            print(f"Alpha Vantage current price request failed: {e}")
            return None


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance provider (free alternative)."""
    
    def __init__(self) -> None:
        """Initialize Yahoo Finance provider."""
        self.base_url = "https://query1.finance.yahoo.com"
        self.rate_limit = 100  # requests per minute (more generous)
        self.last_request_time = 0
    
    async def get_price_data(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> List[MarketDataPoint]:
        """Get historical price data from Yahoo Finance."""
        # Convert dates to timestamps
        start_ts = int(start_date.strftime("%s"))
        end_ts = int(end_date.strftime("%s"))
        
        url = f"{self.base_url}/v8/finance/chart/{symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            
            price_data = []
            for i, ts in enumerate(timestamps):
                price_date = datetime.fromtimestamp(ts).date()
                
                point = MarketDataPoint(
                    symbol=symbol,
                    date=price_date,
                    open=quotes["open"][i],
                    high=quotes["high"][i],
                    low=quotes["low"][i],
                    close=quotes["close"][i],
                    volume=quotes["volume"][i],
                )
                price_data.append(point)
            
            return price_data
            
        except Exception as e:
            print(f"Yahoo Finance price data request failed: {e}")
            return []
    
    async def get_fundamentals(
        self, 
        symbol: str, 
        metrics: List[str]
    ) -> List[FundamentalData]:
        """Get fundamental data from Yahoo Finance."""
        # Yahoo Finance fundamentals require scraping or different endpoints
        # For now, return empty list (would implement with yfinance library)
        return []
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Yahoo Finance."""
        url = f"{self.base_url}/v8/finance/chart/{symbol}"
        params = {"interval": "1d", "range": "1d"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            result = data["chart"]["result"][0]
            current_price = result["meta"]["regularMarketPrice"]
            
            return float(current_price)
            
        except Exception as e:
            print(f"Yahoo Finance current price request failed: {e}")
            return None


class MarketDataService:
    """Market data service with multiple providers and failover."""
    
    def __init__(self) -> None:
        """Initialize market data service."""
        self.providers = []
        
        # Add Alpha Vantage if API key is available
        if hasattr(settings, 'ALPHA_VANTAGE_API_KEY') and settings.ALPHA_VANTAGE_API_KEY:
            self.providers.append(AlphaVantageProvider(settings.ALPHA_VANTAGE_API_KEY))
        
        # Add Yahoo Finance as fallback
        self.providers.append(YahooFinanceProvider())
        
        if not self.providers:
            raise ValueError("No market data providers configured")
    
    @trace_function("market_data_service.get_price_data")
    async def get_price_data(
        self, 
        symbol: str, 
        start_date: date, 
        end_date: date
    ) -> List[MarketDataPoint]:
        """Get price data with provider failover."""
        for provider in self.providers:
            try:
                data = await provider.get_price_data(symbol, start_date, end_date)
                if data:
                    return data
            except Exception as e:
                print(f"Provider {type(provider).__name__} failed: {e}")
                continue
        
        return []
    
    @trace_function("market_data_service.get_fundamentals")
    async def get_fundamentals(
        self, 
        symbol: str, 
        metrics: List[str]
    ) -> List[FundamentalData]:
        """Get fundamental data with provider failover."""
        for provider in self.providers:
            try:
                data = await provider.get_fundamentals(symbol, metrics)
                if data:
                    return data
            except Exception as e:
                print(f"Provider {type(provider).__name__} failed: {e}")
                continue
        
        return []
    
    @trace_function("market_data_service.get_current_price")
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with provider failover."""
        for provider in self.providers:
            try:
                price = await provider.get_current_price(symbol)
                if price is not None:
                    return price
            except Exception as e:
                print(f"Provider {type(provider).__name__} failed: {e}")
                continue
        
        return None
    
    @trace_function("market_data_service.get_multiple_prices")
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols."""
        tasks = [self.get_current_price(symbol) for symbol in symbols]
        prices = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = {}
        for symbol, price in zip(symbols, prices):
            if isinstance(price, Exception):
                print(f"Failed to get price for {symbol}: {price}")
                result[symbol] = None
            else:
                result[symbol] = price
        
        return result
