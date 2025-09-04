#!/usr/bin/env python3
"""
Multi-River Flow Prediction Framework with Auto Weather Data
A configurable system for predicting flows on multiple rivers
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Page config
st.set_page_config(
    page_title="Multi-River Flow Predictor",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class WeatherStation:
    name: str
    lat: float
    lon: float
    weight: float = 1.0  # Importance weight for this station
    
@dataclass
class FlowCategory:
    min_flow: float
    max_flow: float
    label: str
    color: str
    description: str
    difficulty: str

@dataclass
class RiverConfig:
    name: str
    usgs_site: str
    state: str
    description: str
    weather_stations: List[WeatherStation]
    flow_categories: List[FlowCategory]
    model_params: Dict  # Prediction model parameters
    seasonal_adjustments: Dict[int, float]  # Month -> multiplier
    
class MultiRiverPredictor:
    def __init__(self):
        self.rivers = self.load_river_configs()
        
    def load_river_configs(self) -> Dict[str, RiverConfig]:
        """Load river configurations"""
        
        # Define flow categories (can be customized per river)
        standard_categories = [
            FlowCategory(0, 800, "Very Low", "#DC2626", "Extremely low - most runs unrunnable", "No Go"),
            FlowCategory(800, 1500, "Low", "#EA580C", "Low water - only shallow runs possible", "Experienced Only"),
            FlowCategory(1500, 3000, "Moderate", "#D97706", "Decent conditions - most runs okay", "Intermediate+"),
            FlowCategory(3000, 6000, "Good", "#059669", "Good conditions - all runs open", "All Levels"),
            FlowCategory(6000, 10000, "High", "#2563EB", "High water - excellent conditions", "Intermediate+"),
            FlowCategory(10000, 20000, "Very High", "#4F46E5", "Very high - advanced runs only", "Advanced Only"),
            FlowCategory(20000, float('inf'), "Flood", "#7C3AED", "Flood stage - DANGEROUS", "STAY OFF RIVER")
        ]
        
        # Define weather stations for each river watershed
        new_river_stations = [
            WeatherStation("Boone, NC", 36.2168, -81.6746, 1.0),
            WeatherStation("Blowing Rock, NC", 36.1343, -81.6787, 0.7),
            WeatherStation("West Jefferson, NC", 36.4043, -81.4926, 0.8)
        ]
        
        french_broad_stations = [
            WeatherStation("Asheville, NC", 35.5951, -82.5515, 1.0),
            WeatherStation("Hot Springs, NC", 35.8973, -82.8278, 1.0),
            WeatherStation("Marshall, NC", 35.7973, -82.6793, 0.9),
            WeatherStation("Burnsville, NC", 35.9154, -82.2968, 0.7)
        ]
        
        nolichucky_stations = [
            WeatherStation("Erwin, TN", 36.1581, -82.4193, 1.0),
            WeatherStation("Elizabethton, TN", 36.3487, -82.2107, 0.9),
            WeatherStation("Burnsville, NC", 35.9154, -82.2968, 0.8),
            WeatherStation("Banner Elk, NC", 36.1626, -81.8712, 0.6)
        ]
        
        watauga_stations = [
            WeatherStation("Boone, NC", 36.2168, -81.6746, 1.0),
            WeatherStation("Elizabethton, TN", 36.3487, -82.2107, 1.0),
            WeatherStation("Banner Elk, NC", 36.1626, -81.8712, 0.8),
            WeatherStation("Mountain City, TN", 36.4734, -81.8065, 0.7)
        ]
        
        rivers = {
            "new_river": RiverConfig(
                name="New River at Fayette, WV",
                usgs_site="03185400",
                state="WV",
                description="Classic Class III-IV whitewater in the New River Gorge",
                weather_stations=new_river_stations,
                flow_categories=standard_categories,
                model_params={
                    'flow_persistence': 0.75,
                    'precip_today': 40,
                    'precip_yesterday': 60,
                    'precip_day2': 35,
                    'precip_day3': 20,
                    'precip_3day': 8,
                    'precip_7day': 3,
                    'base_flow': 200,
                    'min_flow': 300
                },
                seasonal_adjustments={
                    1: 1.0, 2: 1.0, 3: 1.15, 4: 1.15, 5: 1.15,  # Spring snowmelt
                    6: 0.85, 7: 0.85, 8: 0.85,  # Summer ET
                    9: 1.05, 10: 1.05, 11: 1.05, 12: 1.0  # Fall/Winter
                }
            ),
            
            "french_broad": RiverConfig(
                name="French Broad River at Hot Springs, NC",
                usgs_site="03451500",
                state="NC/TN",
                description="Classic Class III-IV river through scenic Appalachian gorge",
                weather_stations=french_broad_stations,
                flow_categories=[
                    FlowCategory(0, 500, "Very Low", "#DC2626", "Too low - rocks everywhere", "No Go"),
                    FlowCategory(500, 1000, "Low", "#EA580C", "Marginal - experienced only", "Experienced Only"),
                    FlowCategory(1000, 2000, "Moderate", "#D97706", "Good conditions - all rapids runnable", "Intermediate+"),
                    FlowCategory(2000, 4000, "Good", "#059669", "Prime conditions - excellent rapids", "All Levels"),
                    FlowCategory(4000, 7000, "High", "#2563EB", "High water - pushy but great", "Intermediate+"),
                    FlowCategory(7000, 12000, "Very High", "#4F46E5", "Big water - advanced only", "Advanced Only"),
                    FlowCategory(12000, float('inf'), "Flood", "#7C3AED", "Flood stage - DANGEROUS", "STAY OFF RIVER")
                ],
                model_params={
                    'flow_persistence': 0.78,  # Fairly stable mountain river
                    'precip_today': 35,
                    'precip_yesterday': 55,
                    'precip_day2': 40,
                    'precip_day3': 25,
                    'precip_3day': 12,
                    'precip_7day': 5,
                    'base_flow': 400,
                    'min_flow': 300
                },
                seasonal_adjustments={
                    1: 1.1, 2: 1.1, 3: 1.2, 4: 1.15, 5: 1.0,  # Spring snowmelt boost
                    6: 0.8, 7: 0.75, 8: 0.8,  # Summer drought
                    9: 0.9, 10: 1.0, 11: 1.05, 12: 1.1  # Fall pickup, winter base
                }
            ),
            
            "nolichucky": RiverConfig(
                name="Nolichucky River at Erwin, TN",
                usgs_site="03467000",
                state="TN/NC",
                description="Technical Class III-IV with stunning gorge scenery",
                weather_stations=nolichucky_stations,
                flow_categories=[
                    FlowCategory(0, 800, "Very Low", "#DC2626", "Too low - not runnable", "No Go"),
                    FlowCategory(800, 1500, "Low", "#EA580C", "Marginal - lots of rocks", "Experienced Only"),
                    FlowCategory(1500, 3000, "Moderate", "#D97706", "Good flow - technical rapids", "Intermediate+"),
                    FlowCategory(3000, 5000, "Good", "#059669", "Excellent conditions", "All Levels"),
                    FlowCategory(5000, 8000, "High", "#2563EB", "High water - big and pushy", "Intermediate+"),
                    FlowCategory(8000, 15000, "Very High", "#4F46E5", "Huge water - experts only", "Advanced Only"),
                    FlowCategory(15000, float('inf'), "Flood", "#7C3AED", "Flood conditions - deadly", "STAY OFF RIVER")
                ],
                model_params={
                    'flow_persistence': 0.72,  # More flashy mountain river
                    'precip_today': 45,
                    'precip_yesterday': 65,  # Peak response 
                    'precip_day2': 45,
                    'precip_day3': 30,
                    'precip_3day': 15,
                    'precip_7day': 8,
                    'base_flow': 600,
                    'min_flow': 400
                },
                seasonal_adjustments={
                    1: 1.0, 2: 1.05, 3: 1.2, 4: 1.15, 5: 1.0,
                    6: 0.75, 7: 0.7, 8: 0.75,  # Very dry summers
                    9: 0.85, 10: 0.95, 11: 1.0, 12: 1.0
                }
            ),
            
            "watauga": RiverConfig(
                name="Watauga River at Sugar Grove, NC",
                usgs_site="03479000",
                state="NC/TN", 
                description="Fun Class II-III with continuous rapids and beautiful scenery",
                weather_stations=watauga_stations,
                flow_categories=[
                    FlowCategory(0, 200, "Very Low", "#DC2626", "Unrunnable - all rocks", "No Go"),
                    FlowCategory(200, 400, "Low", "#EA580C", "Low but possible - scrappy", "Experienced Only"),
                    FlowCategory(400, 800, "Moderate", "#D97706", "Good flow - fun rapids", "Intermediate+"),
                    FlowCategory(800, 1500, "Good", "#059669", "Prime conditions - perfect", "All Levels"),
                    FlowCategory(1500, 2500, "High", "#2563EB", "High water - fast and fun", "Intermediate+"),
                    FlowCategory(2500, 4000, "Very High", "#4F46E5", "Pushy high water", "Advanced Only"),
                    FlowCategory(4000, float('inf'), "Flood", "#7C3AED", "Flood stage", "STAY OFF RIVER")
                ],
                model_params={
                    'flow_persistence': 0.8,  # More stable flow
                    'precip_today': 30,
                    'precip_yesterday': 50,
                    'precip_day2': 35,
                    'precip_day3': 20,
                    'precip_3day': 10,
                    'precip_7day': 6,
                    'base_flow': 300,
                    'min_flow': 150
                },
                seasonal_adjustments={
                    1: 1.05, 2: 1.1, 3: 1.15, 4: 1.1, 5: 0.95,
                    6: 0.8, 7: 0.75, 8: 0.8,  # Summer low water
                    9: 0.9, 10: 1.0, 11: 1.05, 12: 1.05
                }
            )
        }
        
        return rivers

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def fetch_usgs_data(_self, site_id: str) -> Optional[pd.DataFrame]:
        """Fetch USGS data for any site"""
        try:
            url = f"https://waterservices.usgs.gov/nwis/iv/"
            params = {
                'format': 'json',
                'sites': site_id,
                'parameterCd': '00060',
                'period': 'P14D'  # Get 14 days of data for better history
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'timeSeries' in data['value'] and len(data['value']['timeSeries']) > 0:
                values = data['value']['timeSeries'][0]['values'][0]['value']
                df = pd.DataFrame(values)
                df['dateTime'] = pd.to_datetime(df['dateTime'])
                df['flow'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna().sort_values('dateTime')  # Sort by date
                
                # Remove duplicates and keep one reading per day (most recent)
                df['date'] = df['dateTime'].dt.date
                df_daily = df.groupby('date').last().reset_index()
                df_daily['dateTime'] = pd.to_datetime(df_daily['date'])
                
                return df_daily[['dateTime', 'flow']]
            return None
            
        except Exception as e:
            st.error(f"Error fetching USGS data for {site_id}: {e}")
            return None

    @st.cache_data(ttl=3600)  # Cache for 1 hour (forecasts don't change rapidly)
    def fetch_weather_forecast(_self, stations: List[WeatherStation], days_ahead: int = 7) -> Dict:
        """Fetch weather forecast from Open-Meteo API for multiple stations"""
        forecast_data = {}
        
        for i, station in enumerate(stations):
            try:
                # Open-Meteo Forecast API
                start_date = datetime.now().date()
                end_date = start_date + timedelta(days=days_ahead)
                
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    'latitude': station.lat,
                    'longitude': station.lon,
                    'daily': ['precipitation_sum', 'temperature_2m_mean', 'precipitation_probability_max'],
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'timezone': 'America/New_York',
                    'temperature_unit': 'fahrenheit',
                    'precipitation_unit': 'inch'
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'daily' in data:
                    daily = data['daily']
                    
                    # Store forecast data for each day ahead
                    for day_idx in range(min(days_ahead, len(daily['time']))):
                        day_key = f'station_{i}_forecast_day_{day_idx}'
                        
                        if day_idx < len(daily.get('precipitation_sum', [])):
                            precip_value = daily['precipitation_sum'][day_idx]
                            forecast_data[day_key] = precip_value if precip_value is not None else 0.0
                        else:
                            forecast_data[day_key] = 0.0
                        
                        # Precipitation probability
                        prob_key = f'station_{i}_precip_prob_day_{day_idx}'
                        if day_idx < len(daily.get('precipitation_probability_max', [])):
                            prob_value = daily['precipitation_probability_max'][day_idx]
                            forecast_data[prob_key] = prob_value if prob_value is not None else 0
                        else:
                            forecast_data[prob_key] = 0
                    
                    # Store forecast temperatures
                    if daily.get('temperature_2m_mean'):
                        for day_idx in range(min(days_ahead, len(daily['temperature_2m_mean']))):
                            temp_key = f'station_{i}_temp_forecast_day_{day_idx}'
                            temp_value = daily['temperature_2m_mean'][day_idx]
                            forecast_data[temp_key] = temp_value if temp_value is not None else 50.0
                        
            except Exception as e:
                st.warning(f"Could not fetch forecast data for {station.name}: {str(e)}")
                # Fill with zeros as fallback
                for day_idx in range(days_ahead):
                    forecast_data[f'station_{i}_forecast_day_{day_idx}'] = 0.0
                    forecast_data[f'station_{i}_precip_prob_day_{day_idx}'] = 0
                    forecast_data[f'station_{i}_temp_forecast_day_{day_idx}'] = 50.0
        
        return forecast_data

    def calculate_forecast_prediction(self, river_config: RiverConfig, current_flow: float,
                                    historical_precip: Dict, forecast_data: Dict, 
                                    days_ahead: int = 1) -> Tuple[int, int]:
        """Calculate flow prediction for multiple days ahead using forecast data"""
        
        params = river_config.model_params
        predictions = []
        
        # Start with current flow
        predicted_flow = current_flow
        
        for target_day in range(1, days_ahead + 1):
            # Flow persistence decreases over time
            persistence_factor = params['flow_persistence'] * (0.95 ** target_day)
            daily_prediction = predicted_flow * persistence_factor
            
            # Calculate weighted precipitation for this forecast day
            total_weight = sum(station.weight for station in river_config.weather_stations)
            
            # Get forecast precipitation for target day
            forecast_precip = 0
            for j, station in enumerate(river_config.weather_stations):
                station_forecast = forecast_data.get(f'station_{j}_forecast_day_{target_day-1}', 0)
                forecast_precip += station_forecast * station.weight
            forecast_precip = forecast_precip / total_weight if total_weight > 0 else 0
            
            # Apply precipitation effect (stronger for closer days)
            precip_effect = forecast_precip * params['precip_yesterday'] * (0.9 ** (target_day - 1))
            daily_prediction += precip_effect
            
            # Add some influence from recent historical precipitation (decreasing over time)
            for hist_day in range(min(3, len(historical_precip))):
                hist_key = f'station_0_day_{hist_day}'
                if hist_key in historical_precip:
                    hist_influence = historical_precip[hist_key] * 10 * (0.8 ** (target_day + hist_day))
                    daily_prediction += hist_influence
            
            # Seasonal adjustment
            month = datetime.now().month
            seasonal_mult = river_config.seasonal_adjustments.get(month, 1.0)
            daily_prediction *= seasonal_mult
            
            # Temperature effects (forecast)
            forecast_temp = forecast_data.get(f'station_0_temp_forecast_day_{target_day-1}', 50)
            if forecast_temp < 32:
                daily_prediction *= 0.8
            
            # Add base flow and enforce minimum
            daily_prediction += params['base_flow']
            daily_prediction = max(daily_prediction, params['min_flow'])
            
            predictions.append(int(daily_prediction))
            
            # Use this prediction as base for next day
            predicted_flow = daily_prediction * 0.9  # Slight decay for multi-day
            
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_weather_data(_self, stations: List[WeatherStation], days_back: int = 7) -> Dict:
        """Fetch weather data from Open-Meteo API for multiple stations"""
        """Fetch weather data from Open-Meteo API for multiple stations"""
        weather_data = {}
        
        for i, station in enumerate(stations):
            try:
                # Open-Meteo API (free, no API key needed)
                end_date = datetime.now().date()
                start_date = end_date - timedelta(days=days_back)
                
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    'latitude': station.lat,
                    'longitude': station.lon,
                    'daily': ['precipitation_sum', 'temperature_2m_mean'],
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'timezone': 'America/New_York',
                    'temperature_unit': 'fahrenheit',
                    'precipitation_unit': 'inch'
                }
                
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'daily' in data:
                    daily = data['daily']
                    dates = daily['time']
                    
                    # Store data for each day (reverse order so day_0 is most recent)
                    for day_idx in range(min(days_back, len(dates))):
                        precip_idx = len(dates) - 1 - day_idx
                        day_key = f'station_{i}_day_{day_idx}'
                        
                        if precip_idx >= 0 and precip_idx < len(daily.get('precipitation_sum', [])):
                            precip_value = daily['precipitation_sum'][precip_idx]
                            weather_data[day_key] = precip_value if precip_value is not None else 0.0
                        else:
                            weather_data[day_key] = 0.0
                    
                    # Current temperature (most recent)
                    if daily.get('temperature_2m_mean') and len(daily['temperature_2m_mean']) > 0:
                        recent_temp = daily['temperature_2m_mean'][-1]
                        weather_data[f'temp_station_{i}'] = recent_temp if recent_temp is not None else 50.0
                    else:
                        weather_data[f'temp_station_{i}'] = 50.0
                        
            except Exception as e:
                st.warning(f"Could not fetch weather data for {station.name}: {str(e)}")
                # Fill with zeros as fallback
                for day_idx in range(days_back):
                    weather_data[f'station_{i}_day_{day_idx}'] = 0.0
                weather_data[f'temp_station_{i}'] = 50.0
        
        return weather_data

    def get_flow_category(self, river_config: RiverConfig, flow: float) -> FlowCategory:
        """Get flow category for specific river"""
        for cat in river_config.flow_categories:
            if cat.min_flow <= flow < cat.max_flow:
                return cat
        return river_config.flow_categories[0]
    
    def calculate_prediction(self, river_config: RiverConfig, current_flow: float, 
                           flow_history: Dict, precipitation: Dict, temperature: Dict) -> int:
        """Calculate prediction for specific river using its model parameters"""
        
        params = river_config.model_params
        
        # Flow persistence
        predicted_flow = current_flow * params['flow_persistence']
        
        # Add flow history effects
        if flow_history.get('yesterday'):
            predicted_flow += flow_history['yesterday'] * 0.15
        if flow_history.get('day2'):
            predicted_flow += flow_history['day2'] * 0.05
        if flow_history.get('day7'):
            predicted_flow += flow_history['day7'] * 0.02
        
        # Precipitation effects - weight by station importance
        total_weight = sum(station.weight for station in river_config.weather_stations)
        
        # Calculate weighted precipitation
        weighted_precip = {}
        for i in range(7):
            day_precip = 0
            for j, station in enumerate(river_config.weather_stations):
                station_precip = precipitation.get(f'station_{j}_day_{i}', 0)
                day_precip += station_precip * station.weight
            weighted_precip[f'day_{i}'] = day_precip / total_weight if total_weight > 0 else 0
        
        # Apply precipitation effects
        predicted_flow += weighted_precip['day_0'] * params['precip_today']
        predicted_flow += weighted_precip['day_1'] * params['precip_yesterday']
        predicted_flow += weighted_precip['day_2'] * params['precip_day2']
        predicted_flow += weighted_precip['day_3'] * params['precip_day3']
        
        # Cumulative effects
        precip_3day = sum(weighted_precip[f'day_{i}'] for i in range(3))
        precip_7day = sum(weighted_precip[f'day_{i}'] for i in range(7))
        predicted_flow += precip_3day * params['precip_3day']
        predicted_flow += precip_7day * params['precip_7day']
        
        # Seasonal adjustment
        month = datetime.now().month
        seasonal_mult = river_config.seasonal_adjustments.get(month, 1.0)
        predicted_flow *= seasonal_mult
        
        # Temperature effects
        current_temp = temperature.get('current', 50)
        if current_temp < 32:
            predicted_flow *= 0.8
        
        # Add base flow and enforce minimum
        predicted_flow += params['base_flow']
        predicted_flow = max(predicted_flow, params['min_flow'])
        
        return int(predicted_flow)
    
    def calculate_confidence(self, flow_history: Dict, precipitation: Dict) -> int:
        """Calculate prediction confidence"""
        confidence = 0.5
        
        if flow_history.get('yesterday'): confidence += 0.2
        if flow_history.get('day2'): confidence += 0.1
        if flow_history.get('day7'): confidence += 0.05
        
        # Count precipitation data points
        precip_points = sum(1 for key, val in precipitation.items() 
                           if 'day_0' in key or 'day_1' in key and val > 0)
        confidence += min(precip_points * 0.05, 0.15)
        
        return min(int(confidence * 100), 95)

def main():
    # Initialize all variables first to avoid UnboundLocalError
    weather_data = {}
    forecast_data = {}
    current_temp = 50.0
    current_usgs_flow = None
    flow_history_auto = {}
    usgs_data = None
    
    predictor = MultiRiverPredictor()
    
    # Sidebar for river selection
    st.sidebar.header("🏞️ Select River")
    
    river_options = {
        "New River, WV": "new_river",
        "French Broad River, NC": "french_broad", 
        "Nolichucky River, TN": "nolichucky",
        "Watauga River, NC": "watauga"
    }
    
    selected_river_name = st.sidebar.selectbox("Choose a river:", list(river_options.keys()))
    selected_river_key = river_options[selected_river_name]
    river_config = predictor.rivers[selected_river_key]
    
    # Display river info
    st.sidebar.markdown(f"**{river_config.description}**")
    st.sidebar.markdown(f"📍 USGS Site: {river_config.usgs_site}")
    st.sidebar.markdown(f"🗺️ State: {river_config.state}")
    
    # Weather stations info
    st.sidebar.markdown("**Weather Stations:**")
    for station in river_config.weather_stations:
        weight_stars = "⭐" * int(station.weight * 3)
        st.sidebar.markdown(f"• {station.name} {weight_stars}")
    
    # Header
    st.title(f"🌊 {river_config.name}")
    st.markdown(f"**USGS Gauge #{river_config.usgs_site}**")
    st.markdown("Multi-river flow prediction with auto weather data • Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    # Auto-fetch current USGS data
    with st.spinner("Fetching current USGS data..."):
        usgs_data = predictor.fetch_usgs_data(river_config.usgs_site)
    
    if usgs_data is not None and len(usgs_data) > 0:
        current_usgs_flow = usgs_data.iloc[-1]['flow']
        st.success(f"✅ Current Flow: **{current_usgs_flow:.0f} cfs** (auto-updated)")
        
        # Auto-populate flow history from USGS data
        flow_history_auto = {}
        if len(usgs_data) > 1:
            # Get today's date for proper indexing
            today = datetime.now().date()
            
            # Create a lookup dictionary by date
            flow_by_date = {}
            for _, row in usgs_data.iterrows():
                date_key = row['dateTime'].date()
                flow_by_date[date_key] = row['flow']
            
            # Get historical flows by going back specific days
            yesterday_date = today - timedelta(days=1)
            day2_date = today - timedelta(days=2)
            day3_date = today - timedelta(days=3)
            day7_date = today - timedelta(days=7)
            day14_date = today - timedelta(days=14)
            
            # Look up flows for each date
            if yesterday_date in flow_by_date:
                flow_history_auto['yesterday'] = flow_by_date[yesterday_date]
            if day2_date in flow_by_date:
                flow_history_auto['day2'] = flow_by_date[day2_date]
            if day3_date in flow_by_date:
                flow_history_auto['day3'] = flow_by_date[day3_date]
            if day7_date in flow_by_date:
                flow_history_auto['day7'] = flow_by_date[day7_date]
            if day14_date in flow_by_date:
                flow_history_auto['day14'] = flow_by_date[day14_date]
            
            # Debug info (you can remove this later)
            st.sidebar.markdown("**🔍 Debug Info:**")
            st.sidebar.text(f"Data points: {len(usgs_data)}")
            st.sidebar.text(f"Date range: {usgs_data['dateTime'].min().date()} to {usgs_data['dateTime'].max().date()}")
            st.sidebar.text(f"Today: {today}")
            
        flow_history_summary = []
        for key, value in flow_history_auto.items():
            day_names = {
                'yesterday': 'Yesterday',
                'day2': '2 days ago', 
                'day3': '3 days ago',
                'day7': '1 week ago',
                'day14': '2 weeks ago'
            }
            if key in day_names:
                flow_history_summary.append(f"{day_names[key]}: {value:.0f} cfs")
            
        if flow_history_summary:
            st.info(f"✅ **Flow History Auto-Loaded:** {' • '.join(flow_history_summary)}")
        else:
            st.warning("⚠️ Could not extract flow history - dates may not match")
            
    else:
        current_usgs_flow = None
        flow_history_auto = {}
        st.warning("⚠️ Could not fetch current USGS data. Please enter manually.")
    
    # Auto-fetch weather data
    with st.spinner("Fetching weather data..."):
        weather_data = predictor.fetch_weather_data(river_config.weather_stations)
    
    if weather_data:
        # Calculate current temperature (average from stations)
        temp_keys = [key for key in weather_data.keys() if key.startswith('temp_station_')]
        if temp_keys:
            current_temp = sum(weather_data[key] for key in temp_keys) / len(temp_keys)
        else:
            current_temp = 50.0
        
        st.success("✅ Weather data loaded automatically")
        
        # Show weather summary
        recent_precip = sum(weather_data.get(f'station_0_day_{i}', 0) for i in range(3))
        st.info(f"""
        **Automatic Weather Data Summary:**
        - **Current Temperature**: {current_temp:.1f}°F
        - **Recent Precipitation** (last 3 days): {recent_precip:.2f} inches
        - **Data Sources**: {len(river_config.weather_stations)} weather stations in watershed
        """)
    else:
        st.warning("⚠️ Could not fetch weather data. Using defaults.")
        weather_data = {}
        current_temp = 50.0

    # Show forecast summary
    if forecast_data:
        st.success("✅ Weather forecast loaded automatically")
        
        # Calculate forecast summary
        forecast_precip_3day = sum(forecast_data.get(f'station_0_forecast_day_{i}', 0) for i in range(3))
        forecast_precip_7day = sum(forecast_data.get(f'station_0_forecast_day_{i}', 0) for i in range(7))
        
        # Show forecast summary with probability
        forecast_summary = []
        for day in range(3):  # Next 3 days
            day_precip = forecast_data.get(f'station_0_forecast_day_{day}', 0)
            day_prob = forecast_data.get(f'station_0_precip_prob_day_{day}', 0)
            day_names = ["Tomorrow", "Day after", "3 days out"][day]
            if day_precip > 0.1 or day_prob > 30:
                forecast_summary.append(f"{day_names}: {day_precip:.2f}\" ({day_prob}% chance)")
        
        if forecast_summary:
            st.info("🌦️ **Upcoming Weather:** " + " • ".join(forecast_summary))
        else:
            st.info(f"☀️ **Forecast Summary:** Mostly dry next 3 days ({forecast_precip_3day:.2f}\" total)")
    else:
        st.warning("⚠️ Could not fetch forecast data.")
        forecast_data = {}

    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Input Data")
        
        # Current conditions
        with st.expander("🌊 Current River Conditions", expanded=True):
            current_flow = st.number_input(
                "Current Flow (cfs)",
                value=float(current_usgs_flow) if current_usgs_flow else 2500.0,
                min_value=0.0,
                step=100.0,
                help="Current flow from USGS gauge (auto-filled if available)"
            )
            
            display_temp = st.number_input(
                "Current Temperature (°F)",
                value=current_temp,
                min_value=-20.0,
                max_value=100.0,
                step=1.0,
                help="Auto-filled from weather data"
            )
        
        # Flow history (auto-populated but can override)
        with st.expander("📈 Recent Flow History (Auto-Loaded)", expanded=False):
            st.markdown("**✅ Flow history loaded from USGS data**")
            
            if flow_history_auto:
                # Show what was auto-loaded
                col_auto1, col_auto2 = st.columns(2)
                
                with col_auto1:
                    st.markdown("**Auto-Loaded Values:**")
                    for key, value in flow_history_auto.items():
                        day_name = {
                            'yesterday': 'Yesterday',
                            'day2': '2 days ago', 
                            'day3': '3 days ago',
                            'day7': '1 week ago'
                        }.get(key, key)
                        st.text(f"{day_name}: {value:.0f} cfs")
                
                with col_auto2:
                    override_flow_history = st.checkbox("Override with manual input", key="override_flow")
                    if override_flow_history:
                        st.info("Manual inputs enabled below")
            
            # Show manual inputs if no auto data or override selected
            show_manual_flow = not flow_history_auto or st.session_state.get("override_flow", False)
            
            if show_manual_flow:
                st.markdown("**Manual Flow Input:**")
                col_hist1, col_hist2, col_hist3 = st.columns(3)
                
                with col_hist1:
                    yesterday_flow = st.number_input("Yesterday (cfs)", 
                        value=float(flow_history_auto.get('yesterday', 0)), step=100.0, key="manual_yesterday")
                    day2_flow = st.number_input("2 days ago (cfs)", 
                        value=float(flow_history_auto.get('day2', 0)), step=100.0, key="manual_day2")
                
                with col_hist2:
                    day3_flow = st.number_input("3 days ago (cfs)", 
                        value=float(flow_history_auto.get('day3', 0)), step=100.0, key="manual_day3")
                    day7_flow = st.number_input("1 week ago (cfs)", 
                        value=float(flow_history_auto.get('day7', 0)), step=100.0, key="manual_day7")
                
                with col_hist3:
                    day14_flow = st.number_input("2 weeks ago (cfs)", value=0.0, step=100.0, key="manual_day14")
            else:
                # Use auto-loaded values
                yesterday_flow = flow_history_auto.get('yesterday', 0)
                day2_flow = flow_history_auto.get('day2', 0)
                day3_flow = flow_history_auto.get('day3', 0)
                day7_flow = flow_history_auto.get('day7', 0)
                day14_flow = 0

        # Weather override option
        if weather_data:
            with st.expander("🌧️ Weather Data (Auto-Loaded)", expanded=False):
                st.markdown("**✅ Precipitation data loaded automatically**")
                
                # Show summary of loaded data
                col1_weather, col2_weather = st.columns(2)
                
                with col1_weather:
                    st.markdown("**Recent Precipitation:**")
                    for day in range(4):
                        day_name = ["Today", "Yesterday", "2 days ago", "3 days ago"][day]
                        # Average across stations
                        avg_precip = sum(weather_data.get(f'station_{i}_day_{day}', 0) 
                                       for i in range(len(river_config.weather_stations))) / len(river_config.weather_stations)
                        st.text(f"{day_name}: {avg_precip:.2f}\"")
                
                with col2_weather:
                    override_weather = st.checkbox("Override with manual input")
                    if override_weather:
                        st.info("Weather inputs will appear below")

        # Manual weather input (if needed)
        if not weather_data or st.session_state.get("override_weather", False):
            with st.expander("🌧️ Manual Weather Input", expanded=True):
                st.warning("⚠️ Enter precipitation data manually")
                
                # Simple manual input - just primary station
                st.markdown(f"**{river_config.weather_stations[0].name} (Primary)**")
                
                cols = st.columns(4)
                days = ["Today", "Yesterday", "2 days ago", "3 days ago"]
                
                for day_idx, day_name in enumerate(days):
                    with cols[day_idx]:
                        manual_precip = st.number_input(
                            day_name,
                            value=0.0,
                            step=0.01,
                            format="%.2f",
                            key=f"manual_day_{day_idx}"
                        )
                        # Override weather_data with manual input
                        weather_data[f'station_0_day_{day_idx}'] = manual_precip

        # Prediction options
        st.subheader("🔮 Prediction Options")
        
        prediction_type = st.radio(
            "Choose prediction type:",
            ["Single Day (Tomorrow)", "Multi-Day Forecast (7 Days)"],
            help="Single day uses historical data, multi-day uses weather forecast"
        )
        
        # Prediction button
        if st.button("🔮 Generate Flow Prediction", type="primary", use_container_width=True):
            # Prepare data
            flow_history = {
                'yesterday': yesterday_flow if yesterday_flow > 0 else None,
                'day2': day2_flow if day2_flow > 0 else None,
                'day3': day3_flow if day3_flow > 0 else None,
                'day7': day7_flow if day7_flow > 0 else None,
                'day14': day14_flow if day14_flow > 0 else None
            }
            
            temperature = {'current': display_temp}
            
            if prediction_type == "Single Day (Tomorrow)":
                # Single day prediction using existing method
                prediction = predictor.calculate_prediction(river_config, current_flow, flow_history, weather_data, temperature)
                confidence = predictor.calculate_confidence(flow_history, weather_data)
                
                # Store in session state
                st.session_state[f'prediction_{selected_river_key}'] = prediction
                st.session_state[f'confidence_{selected_river_key}'] = confidence
                st.session_state[f'prediction_type_{selected_river_key}'] = "single"
                
            else:
                # Multi-day forecast prediction
                if forecast_data:
                    predictions = predictor.calculate_forecast_prediction(
                        river_config, current_flow, weather_data, forecast_data, days_ahead=7
                    )
                    confidence = predictor.calculate_confidence(flow_history, weather_data)
                    
                    # Store forecast predictions
                    st.session_state[f'predictions_{selected_river_key}'] = predictions
                    st.session_state[f'confidence_{selected_river_key}'] = confidence
                    st.session_state[f'prediction_type_{selected_river_key}'] = "forecast"
                else:
                    st.error("❌ Cannot create forecast - weather forecast data not available")
    
    with col2:
        st.header("🎯 Prediction Results")
        
        prediction_key = f'prediction_{selected_river_key}'
        predictions_key = f'predictions_{selected_river_key}'
        confidence_key = f'confidence_{selected_river_key}'
        prediction_type_key = f'prediction_type_{selected_river_key}'
        
        # Check what type of prediction we have
        if prediction_type_key in st.session_state:
            pred_type = st.session_state[prediction_type_key]
            confidence = st.session_state.get(confidence_key, 50)
            
            if pred_type == "single" and prediction_key in st.session_state:
                # Single day prediction display
                prediction = st.session_state[prediction_key]
                category = predictor.get_flow_category(river_config, prediction)
                
                # Main prediction display
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 20px;
                ">
                    <h2 style="margin: 0; color: white;">Tomorrow's Flow</h2>
                    <h1 style="margin: 10px 0; font-size: 3em; color: white;">{prediction:,}</h1>
                    <h3 style="margin: 0; color: white;">cfs</h3>
                    <p style="margin: 10px 0; font-size: 1.2em; color: white;">
                        <strong>{category.label}</strong><br>
                        {category.difficulty}
                    </p>
                    <p style="margin: 5px 0; opacity: 0.9; color: white;">
                        Confidence: {confidence}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**Conditions:** {category.description}")
                
            elif pred_type == "forecast" and predictions_key in st.session_state:
                # Multi-day forecast display
                predictions = st.session_state[predictions_key]
                
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    color: white;
                    text-align: center;
                    margin-bottom: 20px;
                ">
                    <h2 style="margin: 0; color: white;">7-Day Flow Forecast</h2>
                    <p style="margin: 10px 0; color: white;">Based on weather forecast</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create forecast table
                forecast_data_display = []
                day_names = ["Tomorrow", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
                
                for i, (day_name, pred_flow) in enumerate(zip(day_names, predictions)):
                    category = predictor.get_flow_category(river_config, pred_flow)
                    
                    # Get forecast precipitation for this day
                    forecast_precip = forecast_data.get(f'station_0_forecast_day_{i}', 0)
                    precip_prob = forecast_data.get(f'station_0_precip_prob_day_{i}', 0)
                    
                    forecast_data_display.append({
                        'Day': day_name,
                        'Predicted Flow': f"{pred_flow:,} cfs",
                        'Category': category.label,
                        'Rain Forecast': f"{forecast_precip:.2f}\" ({precip_prob}%)",
                        '🎯': '🎯' if category.label in ["Good", "High"] else ('⚠️' if category.label in ["Very Low", "Low"] else ''),
                    })
                
                # Display forecast table
                forecast_df = pd.DataFrame(forecast_data_display)
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)
                
                # Create forecast chart
                fig = go.Figure()
                
                days = list(range(1, len(predictions) + 1))
                fig.add_trace(go.Scatter(
                    x=days,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Flow',
                    line=dict(color='#2563EB', width=3),
                    marker=dict(size=8)
                ))
                
                # Add category background colors
                for cat in river_config.flow_categories[::-1]:
                    if cat.max_flow != float('inf'):
                        fig.add_hrect(
                            y0=cat.min_flow, y1=cat.max_flow,
                            fillcolor=cat.color, opacity=0.1,
                            annotation_text=cat.label, annotation_position="top left"
                        )
                
                fig.update_layout(
                    title="7-Day Flow Forecast",
                    xaxis_title="Days Ahead",
                    yaxis_title="Flow (cfs)",
                    hovermode='x unified',
                    xaxis=dict(tickmode='array', tickvals=days, ticktext=day_names)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best days recommendation
                best_days = []
                for i, (day_name, pred_flow) in enumerate(zip(day_names, predictions)):
                    category = predictor.get_flow_category(river_config, pred_flow)
                    if category.label in ["Good", "High", "Moderate"]:
                        best_days.append(f"{day_name} ({pred_flow:,} cfs)")
                
                if best_days:
                    st.success(f"🎯 **Best Days to Run:** {', '.join(best_days)}")
                else:
                    st.warning("⚠️ **No optimal days** in the 7-day forecast")
        
        else:
            st.info("👆 Select prediction type and click 'Generate Flow Prediction' to see results")
        
        # River-specific tips
        st.subheader("💡 River-Specific Tips")
        if selected_river_key == "new_river":
            st.markdown("""
            - **Best flows**: 3,000-8,000 cfs
            - **Peak timing**: 24-36 hours after Boone area rain
            - **Season**: Spring (March-May) typically best
            - **Access**: Bridge Rapids, Fayette Station
            - **Class**: III-IV, pool-drop style
            """)
        elif selected_river_key == "french_broad":
            st.markdown("""
            - **Best flows**: 2,000-6,000 cfs
            - **Peak timing**: 12-24 hours after Asheville area rain
            - **Season**: Year-round, spring snowmelt excellent
            - **Access**: Hot Springs to Paint Rock
            - **Class**: III-IV, big water when up
            """)
        elif selected_river_key == "nolichucky":
            st.markdown("""
            - **Best flows**: 2,000-6,000 cfs
            - **Peak timing**: 12-18 hours after TN/NC border rain
            - **Season**: Spring best, very low in summer
            - **Access**: Poplar to Erwin
            - **Class**: III-IV+, technical and beautiful
            """)
        elif selected_river_key == "watauga":
            st.markdown("""
            - **Best flows**: 800-2,000 cfs
            - **Peak timing**: 12-24 hours after High Country rain
            - **Season**: Spring/fall best, summer can be low
            - **Access**: Sugar Grove area runs
            - **Class**: II-III, continuous and fun
            """)
        
        # Data sources
        st.subheader("🔗 Data Sources")
        st.markdown(f"""
        - [USGS Gauge #{river_config.usgs_site}](https://waterdata.usgs.gov/nwis/uv?site_no={river_config.usgs_site})
        - [Open-Meteo Weather API](https://open-meteo.com/) (Auto-updated)
        - [American Whitewater](https://www.americanwhitewater.org/)
        """)
    
    # Show USGS data chart
    if usgs_data is not None and len(usgs_data) > 0:
        st.subheader(f"📈 Recent Flow Data - {river_config.name}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=usgs_data['dateTime'],
            y=usgs_data['flow'],
            mode='lines+markers',
            name='Flow (cfs)',
            line=dict(color='#2563EB', width=2)
        ))
        
        # Add category background colors
        for cat in river_config.flow_categories[::-1]:  # Reverse for layering
            if cat.max_flow != float('inf'):
                fig.add_hrect(
                    y0=cat.min_flow, y1=cat.max_flow,
                    fillcolor=cat.color, opacity=0.1,
                    annotation_text=cat.label, annotation_position="top left"
                )
        
        fig.update_layout(
            title=f"{river_config.name} - Last 7 Days",
            xaxis_title="Date",
            yaxis_title="Flow (cfs)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Multi-river comparison
    st.subheader("🏞️ Quick Multi-River Status")
    if st.button("Check All Rivers"):
        river_status = {}
        for key, config in predictor.rivers.items():
            data = predictor.fetch_usgs_data(config.usgs_site)
            if data is not None and len(data) > 0:
                current_flow = data.iloc[-1]['flow']
                category = predictor.get_flow_category(config, current_flow)
                river_status[config.name] = {
                    'flow': current_flow,
                    'category': category.label,
                    'color': category.color
                }
        
        if river_status:
            cols = st.columns(len(river_status))
            for i, (river_name, status) in enumerate(river_status.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div style="
                        border: 2px solid {status['color']};
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                    ">
                        <strong>{river_name.split(' at ')[0]}</strong><br>
                        {status['flow']:.0f} cfs<br>
                        <span style="color: {status['color']}; font-weight: bold;">
                            {status['category']}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
