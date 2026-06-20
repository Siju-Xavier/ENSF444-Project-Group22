import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir):
    """
    Load the NBA games datasets.
    Assumes the Kaggle dataset 'nathanlauga/nba-games' is downloaded to data_dir.
    Files needed: games_details.csv (contains player level game stats) and games.csv
    """
    games_details_path = os.path.join(data_dir, 'games_details.csv')
    games_path = os.path.join(data_dir, 'games.csv')
    
    if not os.path.exists(games_details_path) or not os.path.exists(games_path):
        raise FileNotFoundError(f"Please ensure games.csv and games_details.csv exist in {data_dir}")
        
    details_df = pd.read_csv(games_details_path, low_memory=False)
    games_df = pd.read_csv(games_path)
    return details_df, games_df

def preprocess_data(details_df, games_df):
    """
    Preprocess the data to create the target variable and features.
    Target: 1 if player scored > their historical average, else 0.
    """
    print("Preprocessing data...")
    # Keep relevant columns
    cols_to_keep = ['GAME_ID', 'TEAM_ID', 'PLAYER_ID', 'START_POSITION', 'MIN', 'FGM', 'FGA', 'FG_PCT', 
                    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                    'STL', 'BLK', 'TO', 'PF', 'PTS', 'PLUS_MINUS']
    
    df = details_df[cols_to_keep].copy()
    
    # Drop rows without points data
    df = df.dropna(subset=['PTS'])
    
    # sort data to ensure proper time order
    df = df.sort_values(['PLAYER_ID', 'GAME_ID'])
    
    # compute rolling average (ONLY past games)
    df['AVG_PTS'] = (
        df.groupby('PLAYER_ID')['PTS']
        .expanding()
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
    )
    
    # drop rows where no past average exists (first game per player)
    df = df.dropna(subset=['AVG_PTS'])
    
    # Create target variable: 1 if PTS > AVG_PTS, 0 otherwise
    df['TARGET_ABOVE_AVG'] = (df['PTS'] > df['AVG_PTS']).astype(int)
    
    # Clean 'MIN' (minutes played string like "28:30" or integer)
    def clean_min(m):
        if pd.isna(m): return 0
        if isinstance(m, str) and ':' in m:
            parts = m.split(':')
            return float(parts[0]) + float(parts[1])/60 if len(parts)==2 else 0
        try:
            return float(m)
        except:
            return 0
    df['MIN_NUM'] = df['MIN'].apply(clean_min)
    
    # Merge with games_df to get home/away info
    games_subset = games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    df = pd.merge(df, games_subset, on='GAME_ID', how='left')
    
    # Feature for Home/Away: 1 if Home, 0 if Away
    df['IS_HOME'] = (df['TEAM_ID'] == df['HOME_TEAM_ID']).astype(int)
    
    # Fill any remaining NaNs in numeric columns with 0
    numeric_cols = ['MIN_NUM', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'IS_HOME']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
            
    # Subsample data for a responsive demo (reduce for speed, increase for accuracy)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42)
        
    X = df[numeric_cols]
    y = df['TARGET_ABOVE_AVG']
    
    return X, y

def train_and_evaluate(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # "stratify = y" ensures the class distribution in the training and testing sets matches the original dataset.

    print("Scaling features...")
    scaler = StandardScaler() # Makes all features comparable
    X_train_scaled = scaler.fit_transform(X_train) # fit() → learn mean and std from training data // transform() → scale data
    X_test_scaled = scaler.transform(X_test) # Only transforms test data (DOES NOT fit again)
    # This is important because it prevents data lekeage and keeps test data "unseen"
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred) # % correct predictions
        prec = precision_score(y_test, y_pred, zero_division=0) # how correct positive predictions are
        rec = recall_score(y_test, y_pred, zero_division=0) # how many actual positives you found
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec}
        
        print(f"--- {name} Results ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        
    return results

def plot_model_metrics(results):
    """
    Plot Accuracy, Precision, and Recall for all models in a grouped bar chart.
    """
    print("\nGenerating model comparison plot...")
    
    # Convert results dictionary to a DataFrame
    data = []
    for model_name, metrics in results.items():
        for metric_name, value in metrics.items():
            data.append({
                'Model': model_name,
                'Metric': metric_name,
                'Score': value
            })
    
    df_plot = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Metric', y='Score', hue='Model', data=df_plot, palette='viridis')
    
    # Add labels and title
    plt.title('NBA Player Performance Prediction: Model Comparison', fontsize=16, pad=20)
    plt.ylim(0, 1.0)
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Performance Metrics', fontsize=12)
    plt.legend(title='Machine Learning Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'model_comparison.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved successfully to {plot_path}")
    plt.show() # In some environments plt.show() might not display, but plt.savefig will work.

if __name__ == "__main__":
    # Assuming dataset is in a 'data' folder in the current directory
    data_directory = './data' 
    
    if not os.path.exists(data_directory):
        print(f"Creating directory {data_directory}. Please download the Kaggle dataset into this folder.")
        os.makedirs(data_directory, exist_ok=True)
        print("Expected files: games.csv, games_details.csv")
    else:
        try:
            details_df, games_df = load_data(data_directory)
            X, y = preprocess_data(details_df, games_df)
            results = train_and_evaluate(X, y)
            plot_model_metrics(results)
        except Exception as e:
            print(f"An error occurred: {e}")