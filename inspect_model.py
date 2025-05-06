import joblib
import numpy as np

# Load the model and scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Print model coefficients
print("\nModel Coefficients:")
feature_names = ['points', 'rebounds', 'assists', 'steals', 'blocks', 
                 'turnovers', 'fg_pct', 'ft_pct', 'three_pct']
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Test the model with some sample data
test_data = np.array([[
    100,  # points
    40,   # rebounds
    25,   # assists
    8,    # steals
    5,    # blocks
    15,   # turnovers
    0.45, # fg_pct
    0.80, # ft_pct
    0.35  # three_pct
]])

# Scale the test data
scaled_data = scaler.transform(test_data)

# Make prediction
prediction = model.predict(scaled_data)[0]
print(f"\nTest prediction: {prediction:.2f} wins")

# Print scaler info
print("\nScaler mean_:", scaler.mean_)
print("Scaler scale_:", scaler.scale_) 