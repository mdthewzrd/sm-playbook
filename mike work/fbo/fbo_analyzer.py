"""
FBO Analyzer - Detailed parameter analysis for FBO setups
Shows exact parameter values for debugging and validation
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
API_KEY = "Fm7brz4s23eSocDErnL68cE7wspz2K1I"
BASE_URL = "https://api.polygon.io"

class FBOAnalyzer:
    """Detailed analyzer for FBO (Follow Back Over) setups"""
    
    def __init__(self, polygon_api_key=None):
        self.api_key = polygon_api_key or API_KEY
        self.base_url = BASE_URL
        
        # FBO analysis thresholds (same as scanner)
        self.scan_thresholds = {
            'min_trend_atr': 4.0,           
            'min_gap_atr': 0.5,             
            'min_extension_atr': 1.5,       
            'min_range_close_pct': 70.0,    
            'min_volume_multiple': 0.7,     
            'min_change_atr': 0.5,         
            'max_downtrend_slope': -0.50,    
            'min_ema_extension_pct': 10.0,  
            'min_fade_atr': 2.0,            
            'min_days_since_high': 1.0,       
            'max_days_since_high': 30.0,      
            'min_price': 10.0,              
            'max_price': 1000.0,            
            'min_volume': 10_000_000,
            'min_dollar_volume_20d': 20_000_000,
            'require_dev_band_upper': True,   
            'require_d_minus_1_green': True,  
            
            # FBO Pattern Requirements
            'min_red_days_consecutive': 3,    
            'min_red_days_atr': 0.5,         
            'outlier_volume_multiple': 2,   
            'outlier_range_atr': 2,         
            'outlier_fade_atr': 2.0,          
            'fade_lookback_days': 10,         
            
            # Trend Duration Validation
            'min_trend_duration_days': 3,     
            'max_trend_duration_days': 60,    
            
            # FBO Range Filter - D0 open must be >80% in 30-day range
            'min_range_position_pct': 80.0
        }
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'fbo-analyzer'})

    def fetch_daily_data(self, symbol, start_date, end_date):
        """Fetch daily OHLCV data from Polygon"""
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {'adjusted': 'true', 'sort': 'asc', 'apikey': self.api_key}
        
        try:
            time.sleep(0.012)
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['date'] = pd.to_datetime(df['t'], unit='ms').dt.date
                    df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
                    return df[['date', 'open', 'high', 'low', 'close', 'volume']]
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_all_indicators(self, df):
        """Calculate all indicators needed for FBO analysis"""
        if df.empty or len(df) < 10:
            return df
            
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # Basic calculations
        df['pdc'] = df['close'].shift(1)
        df['true_range'] = np.maximum.reduce([
            df['high'] - df['low'],
            np.abs(df['high'] - df['pdc']),
            np.abs(df['low'] - df['pdc'])
        ])
        
        # Adaptive ATR
        if len(df) >= 200:
            df['atr'] = df['true_range'].rolling(200, min_periods=50).mean()
        elif len(df) >= 50:
            df['atr'] = df['true_range'].rolling(50, min_periods=14).mean()
        else:
            df['atr'] = df['true_range'].rolling(14, min_periods=5).mean()
        
        # EMAs
        df['ema_9'] = df['close'].ewm(span=9, min_periods=5).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=10).mean()
        df['ema_89'] = df['close'].ewm(span=89, min_periods=20).mean()
        
        # Gap, range, and volume calculations
        df['gap_dollars'] = df['open'] - df['pdc']
        df['gap_atr'] = df['gap_dollars'] / df['atr']
        df['range_dollars'] = df['high'] - df['low']
        df['range_atr'] = df['range_dollars'] / df['atr']
        df['close_range'] = np.where(df['range_dollars'] > 0, 
                                   (df['close'] - df['low']) / df['range_dollars'], 0)
        df['price_change'] = df['close'] - df['pdc']
        df['price_change_atr'] = df['price_change'] / df['atr']
        df['open_to_prev_low'] = df['open'] - df['low'].shift(1)
        df['extension_atr'] = df['open_to_prev_low'] / df['atr']
        df['avg_volume_20'] = df['volume'].rolling(20, min_periods=5).mean()
        df['volume_multiple'] = df['volume'] / df['avg_volume_20']
        
        # Dollar volume
        df['dollar_volume'] = df['close'] * df['volume']
        df['avg_dollar_volume_20d'] = df['dollar_volume'].rolling(20, min_periods=5).mean()
        
        # Red/Green classification
        df['is_red_day'] = df['close'] < df['open']
        df['is_green_day'] = df['close'] > df['open']
        
        # Deviation bands
        df['ATR_9'] = df['true_range'].rolling(9, min_periods=3).mean()
        df['dev_band_upper_1'] = df['ema_9'] + 1.0 * df['ATR_9']
        df['dev_band_upper_2'] = df['ema_9'] + 0.5 * df['ATR_9']
        df['dev_band_lower_1'] = df['ema_20'] - 2.0 * df['ATR_9']
        df['dev_band_lower_2'] = df['ema_20'] - 2.4 * df['ATR_9']
        
        df['open_above_dev_upper'] = df['open'] > df['dev_band_upper_2']
        df['high_above_dev_upper'] = df['high'] > df['dev_band_upper_2']
        
        # FBO Range calculations
        df['rolling_high_30'] = df['high'].rolling(30, min_periods=10).max()
        df['rolling_low_30'] = df['low'].rolling(30, min_periods=10).min()
        df['range_30_days'] = df['rolling_high_30'] - df['rolling_low_30']
        df['open_position_in_range_pct'] = np.where(
            df['range_30_days'] > 0,
            ((df['open'] - df['rolling_low_30']) / df['range_30_days'] * 100),
            50.0
        )
        df['open_high_in_range'] = df['open_position_in_range_pct'] > self.scan_thresholds['min_range_position_pct']
        
        return df

    def find_trend_and_euphoric_high(self, df, setup_idx):
        """Find trend start and euphoric high with validation"""
        if setup_idx < 20:
            return None, None, None, None
            
        pre_setup = df.iloc[:setup_idx].copy()
        ema_9_above_20 = (pre_setup['ema_9'] > pre_setup['ema_20']).astype(int)
        cross_signal = ema_9_above_20.diff()
        
        # Find last bullish cross
        cross_indices = pre_setup[cross_signal == 1].index
        if len(cross_indices) == 0:
            return None, None, None, None
            
        trend_start_idx = cross_indices[-1]
        trend_start_date = pre_setup.loc[trend_start_idx, 'date']
        trend_start_price = pre_setup.loc[trend_start_idx, 'close']
        
        # Find euphoric high
        trend_data = pre_setup.iloc[trend_start_idx:]
        if trend_data.empty:
            return None, None, None, None
            
        euphoric_idx = trend_data['high'].idxmax()
        euphoric_date = trend_data.loc[euphoric_idx, 'date']
        euphoric_high = trend_data.loc[euphoric_idx, 'high']
        
        # Validate trend duration
        trend_duration = (euphoric_date - trend_start_date).days
        if not (self.scan_thresholds['min_trend_duration_days'] <= trend_duration <= self.scan_thresholds['max_trend_duration_days']):
            return None, None, None, None
        
        return trend_start_date, trend_start_price, euphoric_date, euphoric_high

    def analyze_fade_pattern_detailed(self, df, setup_idx, euphoric_high):
        """Detailed fade pattern analysis with all calculations shown"""
        if setup_idx < self.scan_thresholds['fade_lookback_days']:
            return False, {}, None
        
        # Find euphoric high index
        euphoric_idx = None
        for i in range(setup_idx):
            if abs(df.iloc[i]['high'] - euphoric_high) < 0.01:
                euphoric_idx = i
                break
        
        if euphoric_idx is None or euphoric_idx >= setup_idx - 1:
            return False, {}, None
        
        # Analyze fade data
        fade_data = df.iloc[euphoric_idx:setup_idx].copy()
        if len(fade_data) < 2:
            return False, {}, None
        
        # Consecutive red days analysis
        red_days_analysis = []
        consecutive_reds = 0
        max_consecutive_reds = 0
        quality_consecutive_reds = 0
        max_quality_consecutive_reds = 0
        
        for i, row in fade_data.iterrows():
            red_day_atr = abs(row['price_change_atr']) if not pd.isna(row['price_change_atr']) else 0
            is_quality_red = red_day_atr >= self.scan_thresholds['min_red_days_atr']
            
            red_days_analysis.append({
                'date': row['date'],
                'is_red_day': row['is_red_day'],
                'price_change_atr': red_day_atr,
                'is_quality_red': is_quality_red,
                'volume_multiple': row['volume_multiple'] if not pd.isna(row['volume_multiple']) else 0,
                'range_atr': row['range_atr'] if not pd.isna(row['range_atr']) else 0
            })
            
            if row['is_red_day']:
                consecutive_reds += 1
                max_consecutive_reds = max(max_consecutive_reds, consecutive_reds)
                
                if is_quality_red:
                    quality_consecutive_reds += 1
                    max_quality_consecutive_reds = max(max_quality_consecutive_reds, quality_consecutive_reds)
                else:
                    quality_consecutive_reds = 0
            else:
                consecutive_reds = 0
                quality_consecutive_reds = 0
        
        # Outlier fade day analysis
        outlier_analysis = []
        has_outlier_fade_day = False
        best_outlier = None
        
        for i, row in fade_data.iterrows():
            if pd.isna(row['volume_multiple']) or pd.isna(row['range_atr']) or pd.isna(row['atr']):
                continue
                
            fade_from_high = (euphoric_high - row['close']) / row['atr']
            is_high_volume = row['volume_multiple'] >= self.scan_thresholds['outlier_volume_multiple']
            is_wide_range = row['range_atr'] >= self.scan_thresholds['outlier_range_atr']
            is_big_fade = fade_from_high >= self.scan_thresholds['outlier_fade_atr']
            
            outlier_conditions = sum([is_high_volume, is_wide_range, is_big_fade])
            
            outlier_analysis.append({
                'date': row['date'],
                'volume_multiple': row['volume_multiple'],
                'range_atr': row['range_atr'],
                'fade_atr': fade_from_high,
                'is_high_volume': is_high_volume,
                'is_wide_range': is_wide_range,
                'is_big_fade': is_big_fade,
                'conditions_met': outlier_conditions,
                'qualifies': outlier_conditions >= 2
            })
            
            if outlier_conditions >= 2 and not has_outlier_fade_day:
                has_outlier_fade_day = True
                best_outlier = outlier_analysis[-1]
        
        has_multiple_red_days = max_quality_consecutive_reds >= self.scan_thresholds['min_red_days_consecutive']
        fade_pattern_valid = has_multiple_red_days or has_outlier_fade_day
        
        fade_info = {
            'multiple_red_days': has_multiple_red_days,
            'max_consecutive_reds': max_consecutive_reds,
            'max_quality_consecutive_reds': max_quality_consecutive_reds,
            'red_days_analysis': red_days_analysis,
            'outlier_fade_day': has_outlier_fade_day,
            'outlier_analysis': outlier_analysis,
            'best_outlier': best_outlier
        }
        
        fade_type = "multiple_red" if has_multiple_red_days else "outlier_fade"
        return fade_pattern_valid, fade_info, fade_type

    def analyze_setup(self, symbol, date_str):
        """Analyze a specific FBO setup in detail"""
        setup_date = pd.to_datetime(date_str).date()
        
        # Fetch extended data
        start_date = (pd.to_datetime(date_str) - timedelta(days=400)).strftime('%Y-%m-%d')
        end_date = (pd.to_datetime(date_str) + timedelta(days=5)).strftime('%Y-%m-%d')
        
        df = self.fetch_daily_data(symbol, start_date, end_date)
        if df.empty:
            print(f"❌ No data available for {symbol}")
            return
        
        # Calculate all indicators
        df = self.calculate_all_indicators(df)
        
        # Find setup day index
        setup_idx = None
        for i, row in df.iterrows():
            if row['date'] == setup_date:
                setup_idx = i
                break
        
        if setup_idx is None:
            print(f"❌ Setup date {date_str} not found for {symbol}")
            return
        
        if setup_idx == 0:
            print(f"❌ Cannot analyze setup on first day for {symbol}")
            return
        
        d_minus_1 = df.iloc[setup_idx - 1]
        d_0 = df.iloc[setup_idx]
        
        print(f"\n🔍 FBO ANALYSIS: {symbol} on {date_str}")
        print("=" * 70)
        
        # Basic setup info
        print(f"📊 BASIC SETUP INFO:")
        print(f"   D-1 Date: {d_minus_1['date']}")
        print(f"   D-1 OHLCV: O:{d_minus_1['open']:.2f} H:{d_minus_1['high']:.2f} L:{d_minus_1['low']:.2f} C:{d_minus_1['close']:.2f} V:{d_minus_1['volume']:,.0f}")
        print(f"   D0 Date: {d_0['date']}")
        print(f"   D0 OHLCV: O:{d_0['open']:.2f} H:{d_0['high']:.2f} L:{d_0['low']:.2f} C:{d_0['close']:.2f} V:{d_0['volume']:,.0f}")
        
        # Basic filters
        print(f"\n✅ BASIC FILTERS:")
        print(f"   Price Range: {d_0['close']:.2f} (req: ${self.scan_thresholds['min_price']:.0f}-${self.scan_thresholds['max_price']:.0f}) ✅" if self.scan_thresholds['min_price'] <= d_0['close'] <= self.scan_thresholds['max_price'] else f"   Price Range: {d_0['close']:.2f} ❌")
        print(f"   Volume: {d_minus_1['volume']:,.0f} (req: ≥{self.scan_thresholds['min_volume']:,.0f}) ✅" if d_minus_1['volume'] >= self.scan_thresholds['min_volume'] else f"   Volume: {d_minus_1['volume']:,.0f} ❌")
        print(f"   ATR: {d_minus_1['atr']:.3f} ✅" if not pd.isna(d_minus_1['atr']) and d_minus_1['atr'] > 0 else f"   ATR: Invalid ❌")
        print(f"   D-1 Green: {d_minus_1['close']:.2f} > {d_minus_1['open']:.2f} ✅" if d_minus_1['close'] > d_minus_1['open'] else f"   D-1 Green: ❌")
        
        # Dollar volume
        d_minus_1_dollar_vol = d_minus_1['avg_dollar_volume_20d'] if not pd.isna(d_minus_1['avg_dollar_volume_20d']) else 0
        print(f"   Dollar Volume: ${d_minus_1_dollar_vol/1_000_000:.1f}M (req: ≥${self.scan_thresholds['min_dollar_volume_20d']/1_000_000:.0f}M) ✅" if d_minus_1_dollar_vol >= self.scan_thresholds['min_dollar_volume_20d'] else f"   Dollar Volume: ${d_minus_1_dollar_vol/1_000_000:.1f}M ❌")
        
        # Deviation band check
        d0_open_above_dev = d_0['open_above_dev_upper'] if not pd.isna(d_0['open_above_dev_upper']) else False
        d0_high_above_dev = d_0['high_above_dev_upper'] if not pd.isna(d_0['high_above_dev_upper']) else False
        dev_band_upper = d_0['dev_band_upper_2'] if not pd.isna(d_0['dev_band_upper_2']) else 0
        print(f"   Dev Band: D0 open {d_0['open']:.2f} vs band {dev_band_upper:.2f} ✅" if d0_open_above_dev or d0_high_above_dev else f"   Dev Band: ❌")
        
        # FBO Range Filter
        open_range_pct = d_0['open_position_in_range_pct'] if not pd.isna(d_0['open_position_in_range_pct']) else 0
        range_high_30 = d_0['rolling_high_30'] if not pd.isna(d_0['rolling_high_30']) else 0
        range_low_30 = d_0['rolling_low_30'] if not pd.isna(d_0['rolling_low_30']) else 0
        print(f"   FBO Range: {open_range_pct:.1f}% in 30-day range (req: >{self.scan_thresholds['min_range_position_pct']:.0f}%) ✅" if open_range_pct > self.scan_thresholds['min_range_position_pct'] else f"   FBO Range: {open_range_pct:.1f}% ❌")
        print(f"   30-day Range: ${range_low_30:.2f} - ${range_high_30:.2f} (D0 open: ${d_0['open']:.2f})")
        
        # Trend analysis
        trend_start_date, trend_start_price, euphoric_date, euphoric_high = self.find_trend_and_euphoric_high(df, setup_idx)
        
        if all([trend_start_date, trend_start_price, euphoric_date, euphoric_high]):
            trend_duration = (euphoric_date - trend_start_date).days
            print(f"\n📈 TREND ANALYSIS:")
            print(f"   Trend Start: {trend_start_date} at ${trend_start_price:.2f}")
            print(f"   Euphoric High: {euphoric_date} at ${euphoric_high:.2f}")
            print(f"   Trend Duration: {trend_duration} days (req: {self.scan_thresholds['min_trend_duration_days']}-{self.scan_thresholds['max_trend_duration_days']}) ✅" if self.scan_thresholds['min_trend_duration_days'] <= trend_duration <= self.scan_thresholds['max_trend_duration_days'] else f"   Trend Duration: {trend_duration} days ❌")
            
            # Fade pattern analysis
            fade_valid, fade_info, fade_type = self.analyze_fade_pattern_detailed(df, setup_idx, euphoric_high)
            
            print(f"\n🔴 FADE PATTERN ANALYSIS:")
            print(f"   Fade Type: {fade_type if fade_valid else 'INVALID'}")
            print(f"   Multiple Red Days: {fade_info['multiple_red_days']} (req: {self.scan_thresholds['min_red_days_consecutive']}+ quality consecutive)")
            print(f"   Max Consecutive Reds: {fade_info['max_consecutive_reds']}")
            print(f"   Max Quality Consecutive Reds: {fade_info['max_quality_consecutive_reds']}")
            
            # Show red days breakdown
            if fade_info['red_days_analysis']:
                print(f"\n   📋 RED DAYS BREAKDOWN:")
                for day in fade_info['red_days_analysis']:
                    status = "✅" if day['is_quality_red'] and day['is_red_day'] else "❌" if day['is_red_day'] else "⚪"
                    print(f"      {day['date']}: Red:{day['is_red_day']} ATR:{day['price_change_atr']:.2f} Vol:{day['volume_multiple']:.1f}x {status}")
            
            # Show outlier analysis
            print(f"\n   🎯 OUTLIER FADE ANALYSIS:")
            print(f"   Has Outlier Fade: {fade_info['outlier_fade_day']}")
            if fade_info['best_outlier']:
                best = fade_info['best_outlier']
                print(f"   Best Outlier: {best['date']} - Vol:{best['volume_multiple']:.1f}x Range:{best['range_atr']:.1f} Fade:{best['fade_atr']:.1f} ({best['conditions_met']}/3)")
            
            if fade_info['outlier_analysis']:
                print(f"   📋 OUTLIER DAYS BREAKDOWN:")
                for day in fade_info['outlier_analysis']:
                    status = "✅" if day['qualifies'] else "❌"
                    print(f"      {day['date']}: Vol:{day['volume_multiple']:.1f}x({day['is_high_volume']}) Range:{day['range_atr']:.1f}({day['is_wide_range']}) Fade:{day['fade_atr']:.1f}({day['is_big_fade']}) = {day['conditions_met']}/3 {status}")
            
            # Calculate final metrics
            if fade_valid:
                days_since_high = (d_minus_1['date'] - euphoric_date).days
                trend_atr_multiples = (euphoric_high - trend_start_price) / d_minus_1['atr']
                fade_atr = (euphoric_high - d_minus_1['close']) / d_minus_1['atr']
                downtrend_slope = (d_minus_1['close'] - euphoric_high) / max(days_since_high, 1)
                
                gap_atr = d_0['gap_atr'] if not pd.isna(d_0['gap_atr']) else 0
                extension_atr = d_0['extension_atr'] if not pd.isna(d_0['extension_atr']) else 0
                range_close_pct = d_minus_1['close_range'] * 100
                volume_multiple = d_minus_1['volume_multiple'] if not pd.isna(d_minus_1['volume_multiple']) else 0
                change_atr = abs(d_minus_1['price_change_atr']) if not pd.isna(d_minus_1['price_change_atr']) else 0
                ema_extension_pct = ((d_0['open'] - d_0['ema_89']) / d_0['ema_89']) * 100 if d_0['ema_89'] > 0 else 0
                
                print(f"\n📏 FINAL METRICS:")
                print(f"   Trend ATR: {trend_atr_multiples:.2f} (req: ≥{self.scan_thresholds['min_trend_atr']:.1f}) ✅" if trend_atr_multiples >= self.scan_thresholds['min_trend_atr'] else f"   Trend ATR: {trend_atr_multiples:.2f} ❌")
                print(f"   Gap ATR: {gap_atr:.2f} (req: ≥{self.scan_thresholds['min_gap_atr']:.1f}) ✅" if gap_atr >= self.scan_thresholds['min_gap_atr'] else f"   Gap ATR: {gap_atr:.2f} ❌")
                print(f"   Extension ATR: {extension_atr:.2f} (req: ≥{self.scan_thresholds['min_extension_atr']:.1f}) ✅" if extension_atr >= self.scan_thresholds['min_extension_atr'] else f"   Extension ATR: {extension_atr:.2f} ❌")
                print(f"   Range Close %: {range_close_pct:.1f}% (req: ≥{self.scan_thresholds['min_range_close_pct']:.1f}%) ✅" if range_close_pct >= self.scan_thresholds['min_range_close_pct'] else f"   Range Close %: {range_close_pct:.1f}% ❌")
                print(f"   Volume Multiple: {volume_multiple:.2f} (req: ≥{self.scan_thresholds['min_volume_multiple']:.1f}) ✅" if volume_multiple >= self.scan_thresholds['min_volume_multiple'] else f"   Volume Multiple: {volume_multiple:.2f} ❌")
                print(f"   Change ATR: {change_atr:.2f} (req: ≥{self.scan_thresholds['min_change_atr']:.1f}) ✅" if change_atr >= self.scan_thresholds['min_change_atr'] else f"   Change ATR: {change_atr:.2f} ❌")
                print(f"   Fade ATR: {fade_atr:.2f} (req: ≥{self.scan_thresholds['min_fade_atr']:.1f}) ✅" if fade_atr >= self.scan_thresholds['min_fade_atr'] else f"   Fade ATR: {fade_atr:.2f} ❌")
                print(f"   Days Since High: {days_since_high} (req: {self.scan_thresholds['min_days_since_high']:.0f}-{self.scan_thresholds['max_days_since_high']:.0f}) ✅" if self.scan_thresholds['min_days_since_high'] <= days_since_high <= self.scan_thresholds['max_days_since_high'] else f"   Days Since High: {days_since_high} ❌")
                print(f"   Downtrend Slope: {downtrend_slope:.3f} (req: ≤{self.scan_thresholds['max_downtrend_slope']:.2f}) ✅" if downtrend_slope <= self.scan_thresholds['max_downtrend_slope'] else f"   Downtrend Slope: {downtrend_slope:.3f} ❌")
                print(f"   EMA Extension %: {ema_extension_pct:.1f}% (req: ≥{self.scan_thresholds['min_ema_extension_pct']:.1f}%) ✅" if ema_extension_pct >= self.scan_thresholds['min_ema_extension_pct'] else f"   EMA Extension %: {ema_extension_pct:.1f}% ❌")
                
                # Final qualification
                scan_criteria = all([
                    trend_atr_multiples >= self.scan_thresholds['min_trend_atr'],
                    gap_atr >= self.scan_thresholds['min_gap_atr'],
                    extension_atr >= self.scan_thresholds['min_extension_atr'],
                    range_close_pct >= self.scan_thresholds['min_range_close_pct'],
                    volume_multiple >= self.scan_thresholds['min_volume_multiple'],
                    change_atr >= self.scan_thresholds['min_change_atr'],
                    downtrend_slope <= self.scan_thresholds['max_downtrend_slope'],
                    ema_extension_pct >= self.scan_thresholds['min_ema_extension_pct'],
                    fade_atr >= self.scan_thresholds['min_fade_atr'],
                    self.scan_thresholds['min_days_since_high'] <= days_since_high <= self.scan_thresholds['max_days_since_high']
                ])
                
                print(f"\n🎯 FINAL RESULT: {'✅ QUALIFIES as FBO setup' if scan_criteria else '❌ Does NOT qualify'}")
            else:
                print(f"\n🎯 FINAL RESULT: ❌ Does NOT qualify (fade pattern invalid)")
        else:
            print(f"\n❌ TREND ANALYSIS: No valid trend found")

# Example usage
if __name__ == "__main__":
    analyzer = FBOAnalyzer()
    
    # Analyze specific setups
    print("FBO Analyzer - Detailed Parameter Analysis")
    print("Example: analyzer.analyze_setup('CVS', '2025-05-01')")
    
    # You can analyze specific setups like:
    # analyzer.analyze_setup('CVS', '2025-05-01')
    # analyzer.analyze_setup('TQQQ', '2025-01-22')