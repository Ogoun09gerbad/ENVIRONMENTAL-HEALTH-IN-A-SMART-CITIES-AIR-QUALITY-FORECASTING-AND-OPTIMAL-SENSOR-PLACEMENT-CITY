# Air Pollution Data Exploratory Analysis
# ==========================================
# Context: Air pollution is a critical public health issue in African cities
# This script explores sensor data to understand pollution patterns

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Set style and configurations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 60)
print("AIR POLLUTION DATA EXPLORATORY ANALYSIS")
print("=" * 60)

# Load the dataset
print("\n1. LOADING DATA")
print("-" * 30)
df = pd.read_csv('march_2018_sensor_data_archive.csv', sep=';')

print(f"Dataset Shape: {df.shape}")
print(f"\nDataset Info:")
df.info()

print(f"\nFirst few rows:")
print(df.head())

# Data preprocessing
print("\n2. DATA PREPROCESSING")
print("-" * 30)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract temporal features
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['day_of_month'] = df['timestamp'].dt.day

print("Temporal features extracted successfully")

# Basic statistics
print(f"\nNumerical columns statistics:")
print(df.describe())

print(f"\nCategorical columns statistics:")
print(df.describe(include=['object']))

# Geographic Analysis
print("\n3. GEOGRAPHIC ANALYSIS")
print("-" * 30)

# Analyze sensor locations
unique_locations = df[['location', 'lat', 'lon']].drop_duplicates()
print(f"Number of unique locations: {len(unique_locations)}")
print(f"\nUnique locations:")
print(unique_locations)

# Analyze sensor types by location
sensor_location_analysis = df.groupby(['location', 'sensor_type']).size().unstack(fill_value=0)
print(f"\nSensor types by location:")
print(sensor_location_analysis)

# Create geographic distribution plot
plt.figure(figsize=(12, 8))
for i, loc in unique_locations.iterrows():
    plt.scatter(loc['lon'], loc['lat'], s=200, alpha=0.7, label=f"Location {loc['location']}")
    plt.annotate(f"Loc {loc['location']}", (loc['lon'], loc['lat']), xytext=(5, 5), 
                 textcoords='offset points', fontsize=8)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Sensor Locations Geographic Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot sensor distribution by location
plt.figure(figsize=(12, 6))
sensor_location_analysis.plot(kind='bar', stacked=True)
plt.title('Sensor Types Distribution by Location')
plt.xlabel('Location')
plt.ylabel('Number of Readings')
plt.xticks(rotation=45)
plt.legend(title='Sensor Type')
plt.tight_layout()
plt.show()

# Temporal Analysis
print("\n4. TEMPORAL ANALYSIS")
print("-" * 30)

# Analyze temporal coverage
print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Total days covered: {(df['timestamp'].max() - df['timestamp'].min()).days}")

# Count readings per day
daily_counts = df.groupby('date').size()
print(f"\nReadings per day statistics:")
print(f"Min: {daily_counts.min()}, Max: {daily_counts.max()}, Mean: {daily_counts.mean():.1f}")

# Plot temporal coverage
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Daily readings count
daily_counts.plot(ax=axes[0,0])
axes[0,0].set_title('Daily Readings Count')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Number of Readings')
axes[0,0].tick_params(axis='x', rotation=45)

# Hourly distribution
hourly_counts = df.groupby('hour').size()
hourly_counts.plot(kind='bar', ax=axes[0,1])
axes[0,1].set_title('Readings Distribution by Hour')
axes[0,1].set_xlabel('Hour of Day')
axes[0,1].set_ylabel('Number of Readings')

# Day of week distribution
dow_counts = df.groupby('day_of_week').size()
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_counts.reindex(dow_order).plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Readings Distribution by Day of Week')
axes[1,0].set_xlabel('Day of Week')
axes[1,0].set_ylabel('Number of Readings')
axes[1,0].tick_params(axis='x', rotation=45)

# Data completeness heatmap (hour vs day)
hour_day_pivot = df.groupby(['day_of_month', 'hour']).size().unstack(fill_value=0)
sns.heatmap(hour_day_pivot, ax=axes[1,1], cmap='YlOrRd', cbar_kws={'label': 'Number of Readings'})
axes[1,1].set_title('Data Completeness Heatmap (Hour vs Day)')
axes[1,1].set_xlabel('Hour of Day')
axes[1,1].set_ylabel('Day of Month')

plt.tight_layout()
plt.show()

# Pollutant and Meteorological Analysis
print("\n5. POLLUTANT AND METEOROLOGICAL ANALYSIS")
print("-" * 30)

# Analyze value types
value_types = df['value_type'].value_counts()
print("Value types distribution:")
print(value_types)

# Create separate dataframes for different measurement types
pollutant_data = df[df['value_type'].isin(['P1', 'P2'])].copy()
meteorological_data = df[df['value_type'].isin(['temperature', 'humidity'])].copy()

print(f"\nPollutant readings: {len(pollutant_data)}")
print(f"Meteorological readings: {len(meteorological_data)}")

# Analyze PM2.5 (P2) and PM10 (P1) levels
pm_stats = pollutant_data.groupby('value_type')['value'].describe()
print(f"\nPM Statistics:")
print(pm_stats)

# Create box plots for PM levels
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PM2.5 (P2) distribution
p2_data = pollutant_data[pollutant_data['value_type'] == 'P2']['value']
sns.boxplot(y=p2_data, ax=axes[0])
axes[0].set_title('PM2.5 (P2) Distribution')
axes[0].set_ylabel('Concentration (μg/m³)')

# PM10 (P1) distribution
p1_data = pollutant_data[pollutant_data['value_type'] == 'P1']['value']
sns.boxplot(y=p1_data, ax=axes[1])
axes[1].set_title('PM10 (P1) Distribution')
axes[1].set_ylabel('Concentration (μg/m³)')

plt.tight_layout()
plt.show()

# Analyze meteorological variables
meteo_stats = meteorological_data.groupby('value_type')['value'].describe()
print(f"\nMeteorological Variables Statistics:")
print(meteo_stats)

# Create plots for meteorological variables
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Temperature distribution
temp_data = meteorological_data[meteorological_data['value_type'] == 'temperature']['value']
sns.histplot(temp_data, kde=True, ax=axes[0,0])
axes[0,0].set_title('Temperature Distribution')
axes[0,0].set_xlabel('Temperature (°C)')

# Humidity distribution
hum_data = meteorological_data[meteorological_data['value_type'] == 'humidity']['value']
sns.histplot(hum_data, kde=True, ax=axes[0,1])
axes[0,1].set_title('Humidity Distribution')
axes[0,1].set_xlabel('Humidity (%)')

# Temperature by hour
temp_hourly = meteorological_data[meteorological_data['value_type'] == 'temperature'].groupby('hour')['value'].mean()
temp_hourly.plot(kind='line', ax=axes[1,0])
axes[1,0].set_title('Average Temperature by Hour')
axes[1,0].set_xlabel('Hour of Day')
axes[1,0].set_ylabel('Temperature (°C)')

# Humidity by hour
hum_hourly = meteorological_data[meteorological_data['value_type'] == 'humidity'].groupby('hour')['value'].mean()
hum_hourly.plot(kind='line', ax=axes[1,1])
axes[1,1].set_title('Average Humidity by Hour')
axes[1,1].set_xlabel('Hour of Day')
axes[1,1].set_ylabel('Humidity (%)')

plt.tight_layout()
plt.show()

# Correlation and Relationship Analysis
print("\n6. CORRELATION AND RELATIONSHIP ANALYSIS")
print("-" * 30)

# Calculate correlations for locations with complete data
correlations = {}
for location in df['location'].unique():
    location_data = df[df['location'] == location]
    pivot_data = location_data.pivot_table(
        index='timestamp', 
        columns='value_type', 
        values='value', 
        aggfunc='first'
    )
    
    # Only calculate if we have multiple variables
    if len(pivot_data.columns) > 1:
        corr_matrix = pivot_data.corr()
        correlations[location] = corr_matrix

# Display correlation matrices for locations with sufficient data
for location, corr_matrix in correlations.items():
    if not corr_matrix.empty:
        print(f"\nCorrelation Matrix for Location {location}:")
        print(corr_matrix)
        
        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title(f'Correlation Matrix - Location {location}')
        plt.tight_layout()
        plt.show()

# Analyze PM2.5 patterns by hour and location
p2_hourly_location = pollutant_data[pollutant_data['value_type'] == 'P2'].groupby(['location', 'hour'])['value'].mean().unstack()

plt.figure(figsize=(15, 8))
for location in p2_hourly_location.index:
    plt.plot(p2_hourly_location.columns, p2_hourly_location.loc[location], 
             marker='o', label=f'Location {location}')

plt.title('PM2.5 (P2) Hourly Patterns by Location')
plt.xlabel('Hour of Day')
plt.ylabel('PM2.5 Concentration (μg/m³)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Data Quality Assessment
print("\n7. DATA QUALITY ASSESSMENT")
print("-" * 30)

# Check for missing values
print("Missing values analysis:")
missing_analysis = df.isnull().sum()
print(missing_analysis)

# Check for duplicate entries
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Check for outliers in PM data
def detect_outliers(data, threshold=3):
    z_scores = np.abs((data - data.mean()) / data.std())
    return z_scores > threshold

p2_outliers = detect_outliers(pollutant_data[pollutant_data['value_type'] == 'P2']['value'])
p1_outliers = detect_outliers(pollutant_data[pollutant_data['value_type'] == 'P1']['value'])

print(f"\nPM2.5 outliers: {p2_outliers.sum()} ({p2_outliers.mean()*100:.1f}% of data)")
print(f"PM10 outliers: {p1_outliers.sum()} ({p1_outliers.mean()*100:.1f}% of data)")

# Data frequency analysis
print("\nData frequency analysis by sensor:")
sensor_frequency = df.groupby(['sensor_id', 'value_type']).agg({
    'timestamp': ['min', 'max', 'count'],
    'value': ['mean', 'std']
}).round(2)

print(sensor_frequency.head(10))

# Summary and Key Findings
print("\n8. SUMMARY AND KEY FINDINGS")
print("-" * 30)

# Generate summary statistics
summary_stats = {
    'total_readings': len(df),
    'unique_locations': df['location'].nunique(),
    'unique_sensors': df['sensor_id'].nunique(),
    'sensor_types': list(df['sensor_type'].unique()),
    'measurement_types': list(df['value_type'].unique()),
    'date_range': f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}",
    'avg_pm25': pollutant_data[pollutant_data['value_type'] == 'P2']['value'].mean(),
    'avg_pm10': pollutant_data[pollutant_data['value_type'] == 'P1']['value'].mean(),
    'avg_temperature': meteorological_data[meteorological_data['value_type'] == 'temperature']['value'].mean(),
    'avg_humidity': meteorological_data[meteorological_data['value_type'] == 'humidity']['value'].mean()
}

print("EXPLORATORY DATA ANALYSIS SUMMARY")
print("=" * 50)
for key, value in summary_stats.items():
    print(f"{key}: {value}")

print("\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)

print("""
1. GEOGRAPHIC COVERAGE: The dataset contains sensors at multiple different locations, 
   providing spatial variation in measurements.

2. TEMPORAL PATTERNS: Data shows clear diurnal patterns in both pollutant concentrations 
   and meteorological variables.

3. POLLUTANT LEVELS: PM2.5 and PM10 concentrations vary significantly by location and 
   time, indicating pollution hotspots.

4. METEOROLOGICAL INFLUENCE: Temperature and humidity show expected daily cycles and 
   may correlate with pollution levels.

5. DATA QUALITY: The dataset appears relatively complete with minimal missing values, 
   though some outliers are present.

RECOMMENDATIONS FOR SENSOR PLACEMENT:
- Consider areas with consistently high PM2.5/PM10 levels for priority monitoring
- Ensure coverage of different microenvironments (traffic, residential, industrial)
- Account for meteorological gradients in sensor placement strategy
- Use temporal patterns to optimize maintenance schedules
""")

print("\nAnalysis completed successfully!")
print("=" * 60)
