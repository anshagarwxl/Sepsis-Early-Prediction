import pandas as pd
import numpy as np
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Generate 5000 rows of synthetic vital signs data
data_size = 5000

# Generate vital signs data
respiratory_rate = np.round(np.random.normal(loc=16, scale=3, size=data_size), 0).astype(int)
respiratory_rate = np.clip(respiratory_rate, 8, 30)

spo2 = np.round(np.random.normal(loc=97, scale=2, size=data_size), 1)
spo2 = np.clip(spo2, 85, 100)

systolic_bp = np.round(np.random.normal(loc=120, scale=15, size=data_size), 0).astype(int)
systolic_bp = np.clip(systolic_bp, 80, 200)

heart_rate = np.round(np.random.normal(loc=80, scale=15, size=data_size), 0).astype(int)
heart_rate = np.clip(heart_rate, 40, 150)

temperature = np.round(np.random.normal(loc=37, scale=0.8, size=data_size), 1)
temperature = np.clip(temperature, 35.0, 41.0)

avpu_levels = ['Alert', 'Voice', 'Pain', 'Unresponsive']
consciousness_level = np.random.choice(avpu_levels, size=data_size, p=[0.75, 0.15, 0.05, 0.05])

# Create the initial DataFrame
df = pd.DataFrame({
    'Respiratory Rate': respiratory_rate,
    'SpO₂': spo2,
    'Systolic Blood Pressure': systolic_bp,
    'Heart Rate': heart_rate,
    'Temperature': temperature,
    'Consciousness Level': consciousness_level
})

# Add the 'Sepsis_Risk' target column
sepsis_risk = np.zeros(data_size, dtype=int)
sepsis_risk[(df['Respiratory Rate'] > 20) | (df['SpO₂'] < 95) | (df['Systolic Blood Pressure'] < 100) | (df['Heart Rate'] > 100) | (df['Temperature'] > 38.0)] = 1
sepsis_risk[np.random.rand(data_size) < 0.05] = 1

df['Sepsis_Risk'] = sepsis_risk

# Save the DataFrame to a CSV file at the specified absolute path
file_path = '/Users/sayanbasu/Desktop/Project/Sepsis-Early-Prediction/sepsis-rag-assistant/data_for_ml_training/data.csv'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
df.to_csv(file_path, index=False)

print(f"Successfully generated and saved {data_size} rows of data to {file_path}")
