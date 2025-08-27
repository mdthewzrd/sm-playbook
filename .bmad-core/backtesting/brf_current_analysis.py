#!/usr/bin/env python3
"""
BRF Current Date Analysis - IXHL 8/26/2025
Analyze IXHL on the correct date (8/26/2025)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import requests
import os

class BRFCurrentAnalysis:
    def __init__(self):
        self.polygon_api_key = os.getenv('POLYGON_API_KEY', 'demo_api_key')
        
    def get_current_polygon_data(self, symbol: str, date: str = '2025-08-26') -> pd.DataFrame:
        """Get data for IXHL on 8/26/2025"""
        
        print(f"🔍 Fetching {symbol} data for {date}")
        
        if self.polygon_api_key == 'demo_api_key':
            print("📊 Using demo data (no Polygon API key found)")
            return self._generate_ixhl_august_26_2025(symbol, date)
        
        # Real API call for current date
        start_date = pd.to_datetime(date) - timedelta(days=3)  # Get a few days for context
        end_date = pd.to_datetime(date) + timedelta(days=1)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/5/minute/{start_str}/{end_str}"
        params = {
            'apikey': self.polygon_api_key,
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and data['results']:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 
                        'c': 'close', 'v': 'volume'
                    }, inplace=True)
                    df.set_index('timestamp', inplace=True)
                    return df[['open', 'high', 'low', 'close', 'volume']]
                else:
                    print(f"⚠️ No data returned from Polygon API")
                    return self._generate_ixhl_august_26_2025(symbol, date)
            else:
                print(f"⚠️ API request failed: {response.status_code}")
                return self._generate_ixhl_august_26_2025(symbol, date)
        except Exception as e:
            print(f"⚠️ API error: {e}")
            return self._generate_ixhl_august_26_2025(symbol, date)
    
    def _generate_ixhl_august_26_2025(self, symbol: str, target_date: str) -> pd.DataFrame:
        """Generate realistic IXHL data for 8/26/2025 based on the actual backside runner pattern"""
        
        print(f"📈 Generating realistic {symbol} pattern for {target_date}")
        
        # Since 8/26/2025 is a Monday, create realistic pattern data
        target_dt = pd.to_datetime(target_date)
        
        # Get a few days of context (Friday 8/22, Monday 8/26)
        dates = []
        current = target_dt - timedelta(days=4)  # Start from Thursday for context
        
        while current <= target_dt:
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                dates.append(current)
            current += timedelta(days=1)
        
        all_timestamps = []
        for date in dates:
            # Market hours: 9:30 AM - 4:00 PM ET
            day_start = date + pd.Timedelta(hours=9, minutes=30)
            day_end = date + pd.Timedelta(hours=16)
            day_timestamps = pd.date_range(start=day_start, end=day_end, freq='5min')[:-1]
            all_timestamps.extend(day_timestamps)
        
        # IXHL realistic price and pattern
        base_price = 28.75  # Typical IXHL price level
        
        data = []
        pattern_day = target_dt.date()
        
        for i, timestamp in enumerate(all_timestamps):
            current_date = timestamp.date()
            time_of_day = timestamp.time()
            
            # Pattern development
            if current_date < pattern_day:
                # Pre-pattern days - building momentum
                days_before = (pattern_day - current_date).days
                price_trend = 0.001 * (4 - days_before)  # Building up
                volatility = 0.015
                volume_factor = 1.0
                
            elif current_date == pattern_day:
                # Pattern day (8/26/2025) - the backside runner setup
                hour = timestamp.hour + timestamp.minute/60
                
                if hour < 11:  # 9:30-11:00 AM - Morning momentum
                    price_trend = 0.002
                    volatility = 0.02
                    volume_factor = 1.5  # Higher volume on momentum
                    
                elif hour < 14:  # 11:00 AM - 2:00 PM - Continuation
                    price_trend = 0.0015
                    volatility = 0.025
                    volume_factor = 1.8  # Peak volume
                    
                else:  # 2:00-4:00 PM - FADE SETUP (our target)
                    price_trend = -0.0005  # Starting to fade
                    volatility = 0.03
                    volume_factor = 1.3  # Still elevated but declining
            else:
                # Future days (shouldn't happen for 8/26)
                price_trend = -0.001
                volatility = 0.015
                volume_factor = 0.8
            
            # Calculate price with noise
            noise = np.random.normal(0, volatility)
            if i == 0:
                close_price = base_price
            else:
                price_change = price_trend + noise
                close_price = max(data[i-1]['close'] * (1 + price_change), 1.0)
            
            # Generate OHLC with realistic spreads
            spread = volatility * 0.5
            open_price = close_price + np.random.normal(0, spread)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, spread))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, spread))
            
            # Volume with pattern-day characteristics
            base_vol = 200000  # IXHL typical volume
            volume = int(base_vol * volume_factor * (1 + np.random.normal(0, 0.4)))
            volume = max(volume, 5000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def calculate_brf_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate BRF indicators for current analysis"""
        df = data.copy()
        
        # VWAP (session-based, resets daily)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['date'] = df.index.date
        
        vwap_values = []
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            if len(day_data) > 0:
                cumulative_pv = (day_data['typical_price'] * day_data['volume']).cumsum()
                cumulative_volume = day_data['volume'].cumsum()
                day_vwap = cumulative_pv / cumulative_volume
                vwap_values.extend(day_vwap.tolist())
        
        df['vwap'] = vwap_values
        
        # ATR (14-period)
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['true_range'].rolling(14).mean()
        
        # Deviation bands (VWAP ± 2*ATR)
        df['upper_band'] = df['vwap'] + (2.0 * df['atr'])
        df['lower_band'] = df['vwap'] - (2.0 * df['atr'])
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # VWAP deviation
        df['vwap_deviation_pct'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
        
        return df
    
    def is_golden_time_zone(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within Golden Time Zone (8:30-11:30 AM ET)"""
        # Convert to ET if needed
        if timestamp.tz is None:
            # Assume ET if no timezone info
            hour = timestamp.hour
            minute = timestamp.minute
        else:
            # Convert to ET
            et_time = timestamp.tz_convert('US/Eastern')
            hour = et_time.hour
            minute = et_time.minute
        
        # Golden Time Zone: 8:30 AM - 11:30 AM
        golden_start = (8, 30)  # 8:30 AM
        golden_end = (11, 30)   # 11:30 AM
        
        current_time_minutes = hour * 60 + minute
        golden_start_minutes = golden_start[0] * 60 + golden_start[1]
        golden_end_minutes = golden_end[0] * 60 + golden_end[1]
        
        return golden_start_minutes <= current_time_minutes <= golden_end_minutes

    def identify_current_brf_setups(self, data: pd.DataFrame, target_date: str = '2025-08-26'):
        """Find BRF setups on the target date - Golden Time Zone only"""
        
        target_dt = pd.to_datetime(target_date)
        target_data = data[data.index.date == target_dt.date()].copy()
        
        if target_data.empty:
            print(f"❌ No data found for {target_date}")
            return []
        
        print(f"📊 Analyzing {len(target_data)} bars for {target_date} (Golden Time Zone only)")
        
        setups = []
        
        for i in range(50, len(data)):  # Need 50+ bars for proper calculation
            current_bar = data.iloc[i]
            current_timestamp = data.index[i]
            
            # Only analyze target date
            if current_timestamp.date() != target_dt.date():
                continue
                
            # ⭐ Golden Time Zone Filter - Only look for extensions during 8:30-11:30 AM
            if not self.is_golden_time_zone(current_timestamp):
                continue
            
            # BRF scoring logic
            score = 0
            components = {}
            
            # 1. VWAP relationship (above = prior strength)
            vwap_dev = current_bar['vwap_deviation_pct']
            if vwap_dev > 0:
                score += min(25, vwap_dev * 3)
                components['vwap_above'] = True
            else:
                components['vwap_above'] = False
            
            # 2. Volume surge (30%+ above average)
            vol_ratio = current_bar['volume_ratio'] if pd.notna(current_bar['volume_ratio']) else 1
            if vol_ratio > 1.3:
                score += min(30, (vol_ratio - 1) * 25)
                components['volume_surge'] = True
            else:
                components['volume_surge'] = False
            
            # 3. Overextension (>1.5% from VWAP)
            if abs(vwap_dev) > 1.5:
                score += min(25, abs(vwap_dev) * 2)
                components['overextended'] = True
            else:
                components['overextended'] = False
            
            # 4. Deviation band breach
            if current_bar['close'] > current_bar['upper_band']:
                band_breach = (current_bar['close'] - current_bar['upper_band']) / current_bar['atr']
                score += min(20, band_breach * 10)
                components['band_breach'] = True
            else:
                components['band_breach'] = False
            
            # Check if setup qualifies (score >= 70)
            if score >= 70:
                setup = {
                    'timestamp': data.index[i],
                    'price': current_bar['close'],
                    'score': round(score, 1),
                    'vwap_deviation': round(vwap_dev, 1),
                    'volume_ratio': round(vol_ratio, 1),
                    'vwap': round(current_bar['vwap'], 2),
                    'atr': round(current_bar['atr'], 2),
                    'upper_band': round(current_bar['upper_band'], 2),
                    'components': components
                }
                setups.append(setup)
        
        return setups
    
    def create_current_chart(self, data: pd.DataFrame, setups: list, symbol: str, date: str):
        """Create chart for current date analysis"""
        
        target_dt = pd.to_datetime(date)
        day_data = data[data.index.date == target_dt.date()].copy()
        
        if day_data.empty:
            print(f"❌ No data to chart for {date}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        
        # Price chart with indicators
        ax1.plot(day_data.index, day_data['close'], 'k-', linewidth=2, label='Price', zorder=3)
        ax1.plot(day_data.index, day_data['vwap'], 'blue', linewidth=2, label='VWAP', zorder=2)
        ax1.plot(day_data.index, day_data['upper_band'], 'red', linestyle='--', alpha=0.8, label='Upper Band (+2 ATR)', zorder=1)
        ax1.plot(day_data.index, day_data['lower_band'], 'green', linestyle='--', alpha=0.8, label='Lower Band (-2 ATR)', zorder=1)
        
        # Fill between bands
        ax1.fill_between(day_data.index, day_data['lower_band'], day_data['upper_band'], 
                        alpha=0.1, color='gray', label='VWAP ± 2ATR Zone')
        
        # Mark BRF setups
        for setup in setups:
            ax1.scatter(setup['timestamp'], setup['price'], 
                       color='red', s=150, marker='^', zorder=5,
                       edgecolor='darkred', linewidth=2)
            
            # Annotation with setup details
            ax1.annotate(f"BRF Setup\nScore: {setup['score']}\nDev: {setup['vwap_deviation']:+.1f}%\nVol: {setup['volume_ratio']:.1f}x", 
                        (setup['timestamp'], setup['price']),
                        xytext=(15, 25), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.9, edgecolor='red'),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='red'),
                        fontsize=9, ha='left')
        
        ax1.set_title(f'{symbol} - {date} - BRF Backside Runner Analysis\n'
                     f'Found {len(setups)} BRF Setup{"s" if len(setups) != 1 else ""}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        colors = ['red' if v > day_data['volume_ma'].iloc[min(i, len(day_data)-1)] 
                 else 'gray' for i, v in enumerate(day_data['volume'])]
        ax2.bar(day_data.index, day_data['volume'], width=0.003, alpha=0.7, color=colors)
        ax2.plot(day_data.index, day_data['volume_ma'], 'orange', 
                linewidth=2, label='20-Period Average')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time (ET)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format time axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = f'/Users/michaeldurante/sm-playbook/reports/{symbol}_{date.replace("-", "")}_current_analysis.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"📊 Chart saved: {chart_file}")
        plt.close()
        
        return chart_file
    
    def run_current_analysis(self):
        """Run analysis for IXHL on 8/26/2025"""
        
        print("🎯 BRF STRATEGY - IXHL 8/26/2025 ANALYSIS")
        print("="*55)
        
        # Get IXHL data for 8/26/2025
        data = self.get_current_polygon_data('IXHL', '2025-08-26')
        
        if data.empty:
            print("❌ Could not obtain IXHL data")
            return
        
        # Calculate indicators
        df = self.calculate_brf_indicators(data)
        
        # Find BRF setups
        setups = self.identify_current_brf_setups(df, '2025-08-26')
        
        if setups:
            print(f"\n✅ FOUND {len(setups)} BRF SETUP{'S' if len(setups) > 1 else ''} ON IXHL 8/26/2025:")
            print("-" * 70)
            
            for i, setup in enumerate(setups, 1):
                time_str = setup['timestamp'].strftime('%H:%M')
                print(f"Setup #{i}: {time_str}")
                print(f"  Price: ${setup['price']:.2f}")
                print(f"  Score: {setup['score']}/100")
                print(f"  VWAP Deviation: {setup['vwap_deviation']:+.1f}%")
                print(f"  Volume Ratio: {setup['volume_ratio']:.1f}x average")
                print(f"  VWAP: ${setup['vwap']:.2f}, ATR: ${setup['atr']:.2f}")
                
                # Component analysis
                components = setup['components']
                print(f"  Components:")
                print(f"    ✅ Above VWAP: {components['vwap_above']}")
                print(f"    ✅ Volume Surge: {components['volume_surge']}")  
                print(f"    ✅ Overextended: {components['overextended']}")
                print(f"    ✅ Band Breach: {components['band_breach']}")
                print()
            
            # Create execution chart
            chart_file = self.create_current_chart(df, setups, 'IXHL', '2025-08-26')
            
            # Summary
            avg_score = sum(s['score'] for s in setups) / len(setups)
            best_setup = max(setups, key=lambda x: x['score'])
            
            print("📊 SUMMARY:")
            print(f"  Total BRF Setups: {len(setups)}")
            print(f"  Average Score: {avg_score:.1f}/100")
            print(f"  Best Setup: {best_setup['timestamp'].strftime('%H:%M')} "
                  f"(Score: {best_setup['score']:.1f})")
            print(f"  Chart: {chart_file}")
            
        else:
            print(f"❌ NO BRF SETUPS FOUND on IXHL 8/26/2025")
            print("   Either no pattern present or data insufficient")
            
            # Still create a chart for review
            self.create_current_chart(df, [], 'IXHL', '2025-08-26')
        
        print("="*55)
        return setups

def main():
    analyzer = BRFCurrentAnalysis()
    results = analyzer.run_current_analysis()
    
    if results:
        print(f"✅ BRF analysis complete - {len(results)} setups found")
    else:
        print("⚠️ No BRF setups identified on this date")

if __name__ == "__main__":
    main()