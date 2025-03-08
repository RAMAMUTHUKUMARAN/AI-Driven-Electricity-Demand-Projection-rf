import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('C:\\Users\\Ramamuthukumaran s\\OneDrive\\Desktop\\AI-Driven-Electricity-Demand-Projection-rf\\AI_DRIVEN_ELECTRICITY_DEMAND_PREDICTION\\prediction_model\\data\\Dataset.csv')

if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.weekday  # Monday=0, Sunday=6

    df.drop(columns=['date'], inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

df_numeric = df.select_dtypes(include=['number'])

corr_matrix = df_numeric.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

print(corr_matrix)
