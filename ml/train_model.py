import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = "data/processed/us_accidents_risk_sample.csv"
MODEL_PATH = "ml/risk_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    print("Loaded dataset with shape:", df.shape)
    print(df.head())

    X = df.drop(columns=["risk_score"])
    y = df["risk_score"]

    X = pd.get_dummies(X, columns=["time_of_day"], drop_first=True)

    print("Feature matrix shape after encoding:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1738
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=1738
    )

    model.fit(X_train, y_train)
    print("Model training complete.")

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error on test set:", mae)

    joblib.dump(model, MODEL_PATH)
    print (f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()