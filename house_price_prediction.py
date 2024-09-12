import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'train.csv' with your actual file path)
data = pd.read_csv("train.csv")

# Select relevant features (GrLivArea, BedroomAbvGr, FullBath) and the target variable (SalePrice)
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Function to predict house price based on square footage, number of bedrooms, and bathrooms
def predict_house_price(square_footage, num_bedrooms, num_bathrooms):
    # Create a new instance as a DataFrame with feature names
    new_instance = pd.DataFrame({
        'GrLivArea': [square_footage],
        'BedroomAbvGr': [num_bedrooms],
        'FullBath': [num_bathrooms]
    })
    
    # Make the prediction
    predicted_price = model.predict(new_instance)
    
    return predicted_price

# Get user input for square footage, number of bedrooms, and bathrooms
square_footage = float(input("Enter square footage: "))
num_bedrooms = int(input("Enter number of bedrooms: "))
num_bathrooms = int(input("Enter number of bathrooms: "))

# Predict the house price based on user input
predicted_price = predict_house_price(square_footage, num_bedrooms, num_bathrooms)

# Output the predicted price
print(f"Predicted house price for {square_footage} sqft, {num_bedrooms} bedrooms, {num_bathrooms} bathrooms: {predicted_price[0]}")
