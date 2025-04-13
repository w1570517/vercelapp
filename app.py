import os
import pickle
import signal
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tempfile

# Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Data Sources
URLS = {
    'premier': [
        'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2122/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2021/E0.csv'
    ],
    'championship': [
        'https://www.football-data.co.uk/mmz4281/2425/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2324/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2223/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2122/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2021/E1.csv'
    ]
}

# Model Persistence
def save_models(models, team_stats):
    """Save models and team stats to disk"""
    os.makedirs('model_data', exist_ok=True)
    try:
        # Save neural network
        nn_path = os.path.join('model_data', 'nn_model')
        save_model(models['nn']['model'], nn_path)
        
        # Save other components
        data = {
            'rf': models['rf'],
            'nn_scaler': models['nn']['scaler'],
            'nn_selector': models['nn']['selector'],
            'nn_label_encoder': models['nn']['label_encoder'],
            'team_stats': team_stats
        }
        
        with open(os.path.join('model_data', 'models.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
        return True
    except Exception as e:
        print(f"Error saving models: {str(e)}")
        return False

def load_models():
    """Load models and team stats from disk"""
    try:
        model_path = os.path.join('model_data', 'models.pkl')
        nn_path = os.path.join('model_data', 'nn_model')
        
        if not os.path.exists(model_path) or not os.path.exists(nn_path):
            return None, None
            
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        nn_model = load_model(nn_path)
        
        return {
            'rf': data['rf'],
            'nn': {
                'model': nn_model,
                'scaler': data['nn_scaler'],
                'selector': data['nn_selector'],
                'label_encoder': data['nn_label_encoder']
            }
        }, data['team_stats']
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

# Data Processing
def load_data(urls):
    """Load and combine data from multiple URLs"""
    all_data = []
    for league, league_urls in urls.items():
        for url in league_urls:
            try:
                df = pd.read_csv(url, encoding='utf-8')
                df['League'] = league
                all_data.append(df)
                print(f"Loaded data from {url}")
            except Exception as e:
                print(f"Error loading {url}: {str(e)}")
                continue
    return pd.concat(all_data, ignore_index=True) if all_data else None

def preprocess_data(df):
    """Preprocess the raw data"""
    if df is None:
        return None
        
    essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                     'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    available_cols = [col for col in essential_cols if col in df.columns]
    df = df.loc[:, available_cols + ['League']]
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(0)
    
    categorical_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'League']
    for col in categorical_cols:
        if col in df.columns:
            df.loc[:, col] = pd.Categorical(df[col])
    
    return df

def calculate_team_form(df, team_col, result_col, window=5):
    """Calculate team form over last N matches"""
    return (df[result_col] == 'H').rolling(window).mean() if team_col == 'HomeTeam' else \
           (df[result_col] == 'A').rolling(window).mean()

def engineer_features(df):
    """Create enhanced features for modeling"""
    if df is None:
        return None, None, None
        
    df = df.copy()
    df['GoalDifference'] = df['FTHG'] - df['FTAG']
    df['TotalShots'] = df['HS'] + df['AS']
    df['ShotAccuracy'] = (df['HST'] + df['AST']) / (df['HS'] + df['AS'] + 1e-6)
    
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    team_stats = {}
    
    for team in teams:
        home_matches = df[df['HomeTeam'] == team].sort_values('Date')
        home_matches['HomeForm'] = calculate_team_form(home_matches, 'HomeTeam', 'FTR')
        
        away_matches = df[df['AwayTeam'] == team].sort_values('Date')
        away_matches['AwayForm'] = calculate_team_form(away_matches, 'AwayTeam', 'FTR')
        
        team_stats[team] = {
            'home_goals_scored': home_matches['FTHG'].mean(),
            'home_goals_conceded': home_matches['FTAG'].mean(),
            'home_shots': home_matches['HS'].mean(),
            'home_form': home_matches['HomeForm'].iloc[-1] if len(home_matches) > 0 else 0.5,
            'away_goals_scored': away_matches['FTAG'].mean(),
            'away_goals_conceded': away_matches['FTHG'].mean(),
            'away_shots': away_matches['AS'].mean(),
            'away_form': away_matches['AwayForm'].iloc[-1] if len(away_matches) > 0 else 0.5,
        }
    
    features, labels = [], []
    for _, match in df.iterrows():
        home_team, away_team = match['HomeTeam'], match['AwayTeam']
        if home_team not in team_stats or away_team not in team_stats:
            continue
            
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]
        
        features.append([
            home_stats['home_goals_scored'],
            home_stats['home_goals_conceded'],
            home_stats['home_shots'],
            home_stats['home_form'],
            away_stats['away_goals_scored'],
            away_stats['away_goals_conceded'],
            away_stats['away_shots'],
            away_stats['away_form'],
            home_stats['home_goals_scored'] - away_stats['away_goals_conceded'],
            away_stats['away_goals_scored'] - home_stats['home_goals_conceded'],
            1 if match['League'] == 'premier' else 0
        ])
        labels.append(match['FTR'])
    
    return np.array(features), np.array(labels), team_stats

# Model Training
def train_random_forest(X, y):
    """Train optimized Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_selected, y_train)
    
    accuracy = model.score(X_test_selected, y_test)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    calibration_data = calculate_calibration(model, X_test_selected, y_test)
    
    return {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'accuracy': accuracy,
        'calibration': calibration_data
    }

def train_neural_network(X, y):
    """Train optimized Neural Network"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_scaled, y_train_encoded)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_selected.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_selected, y_train_encoded,
        validation_data=(X_test_selected, y_test_encoded),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    _, accuracy = model.evaluate(X_test_selected, y_test_encoded, verbose=0)
    print(f"Neural Network Accuracy: {accuracy:.4f}")
    
    calibration_data = calculate_calibration(model, X_test_selected, y_test, is_nn=True, label_encoder=label_encoder)
    
    return {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'calibration': calibration_data
    }

def calculate_calibration(model, X, y_true, is_nn=False, label_encoder=None):
    """Calculate model calibration by binning predictions"""
    if is_nn:
        y_pred_probs = model.predict(X, verbose=0)
        if label_encoder:
            y_pred = label_encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))
    else:
        y_pred_probs = model.predict_proba(X)
        y_pred = model.classes_[np.argmax(y_pred_probs, axis=1)]
    
    calibration_data = {}
    classes = label_encoder.classes_ if is_nn and label_encoder else model.classes_
    
    for i, cls in enumerate(classes):
        prob_true = y_pred_probs[:, i]
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(prob_true, bins) - 1
        
        actual_probs, bin_counts = [], []
        y_true_cls = (y_true == cls)
        
        for bin_idx in range(len(bins) - 1):
            mask = (bin_indices == bin_idx)
            if np.sum(mask) > 0:
                actual_prob = np.mean(y_true_cls[mask])
                actual_probs.append(actual_prob)
                bin_counts.append(np.sum(mask))
            else:
                actual_probs.append(np.nan)
                bin_counts.append(0)
        
        calibration_data[cls] = {
            'bins': bins[:-1] + 0.05,
            'predicted': bins[:-1] + 0.05,
            'actual': actual_probs,
            'counts': bin_counts
        }
    
    return calibration_data

# Prediction Handling
def predict_match(home_team, away_team, model_type='rf'):
    """Make prediction with proper feature alignment"""
    global models, team_stats
    
    if home_team not in team_stats or away_team not in team_stats:
        return None
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    features = np.array([
        home_stats['home_goals_scored'],
        home_stats['home_goals_conceded'],
        home_stats['home_shots'],
        home_stats['home_form'],
        away_stats['away_goals_scored'],
        away_stats['away_goals_conceded'],
        away_stats['away_shots'],
        away_stats['away_form'],
        home_stats['home_goals_scored'] - away_stats['away_goals_conceded'],
        away_stats['away_goals_scored'] - home_stats['home_goals_conceded'],
        1
    ]).reshape(1, -1)
    
    model_info = models[model_type]
    
    try:
        features_scaled = model_info['scaler'].transform(features)
        features_selected = model_info['selector'].transform(features_scaled)
        
        if model_type == 'rf':
            prediction = model_info['model'].predict(features_selected)[0]
            probabilities = model_info['model'].predict_proba(features_selected)[0]
            classes = model_info['model'].classes_
        else:
            probabilities = model_info['model'].predict(features_selected, verbose=0)[0]
            prediction_idx = np.argmax(probabilities)
            prediction = model_info['label_encoder'].inverse_transform([prediction_idx])[0]
            classes = model_info['label_encoder'].classes_
        
        outcome_mapping = {'H': f"{home_team} win", 'D': "Draw", 'A': f"{away_team} win"}
        reliability = {}
        calibration = model_info['calibration']
        
        for cls, prob in zip(classes, probabilities):
            if cls in calibration:
                bin_idx = np.digitize(prob, calibration[cls]['bins']) - 1
                if 0 <= bin_idx < len(calibration[cls]['actual']):
                    actual_prob = calibration[cls]['actual'][bin_idx]
                    if not np.isnan(actual_prob):
                        reliability[outcome_mapping[cls]] = {
                            'predicted_prob': prob,
                            'actual_prob': actual_prob,
                            'reliability': f"{actual_prob:.1%}",
                            'count': calibration[cls]['counts'][bin_idx]
                        }
        
        prob_percentages = {outcome_mapping[cls]: f"{prob:.1%}" for cls, prob in zip(classes, probabilities)}
        
        return {
            'prediction': outcome_mapping[prediction],
            'probabilities': prob_percentages,
            'max_prob': max(prob_percentages.values()),
            'model_type': 'Random Forest' if model_type == 'rf' else 'Neural Network',
            'reliability': reliability
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    now = datetime.now()
    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        model_type = request.form.get('model_type', 'rf')
        
        result = predict_match(home_team, away_team, model_type)
        
        if result:
            return render_template_string(TEMPLATE,
                                teams=sorted(list(team_stats.keys())),
                                result=result,
                                home_team=home_team,
                                away_team=away_team,
                                model_type=model_type,
                                now=now)
    
    return render_template_string(TEMPLATE,
                         teams=sorted(list(team_stats.keys())),
                         now=now)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    model_type = data.get('model_type', 'rf')
    
    if not home_team or not away_team:
        return jsonify({'error': 'Missing team names'}), 400
    
    result = predict_match(home_team, away_team, model_type)
    
    if not result:
        return jsonify({'error': 'Prediction failed - check team names'}), 400
    
    return jsonify(result)

# HTML Template
TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Predictor Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Poppins', sans-serif; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #e2e8f0; min-height: 100vh; }
        .glass-card { background: rgba(15, 23, 42, 0.7); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.08); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36); }
        .team-selector { transition: all 0.3s ease; }
        .team-selector:hover { transform: translateY(-3px); box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
        .probability-bar { transition: width 1s cubic-bezier(0.65, 0, 0.35, 1); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(74, 222, 128, 0); } 100% { box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); } }
        .floating { animation: floating 6s ease-in-out infinite; }
        @keyframes floating { 0% { transform: translateY(0px); } 50% { transform: translateY(-15px); } 100% { transform: translateY(0px); } }
        .fade-in { animation: fadeIn 0.6s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body class="min-h-screen">
    <div class="container mx-auto px-4 py-8 max-w-6xl">
        <div class="fixed -left-20 top-1/4 w-40 h-40 bg-blue-500 rounded-full filter blur-3xl opacity-20"></div>
        <div class="fixed -right-20 bottom-1/4 w-60 h-60 bg-green-500 rounded-full filter blur-3xl opacity-20"></div>
        
        <header class="mb-12 text-center relative z-10">
            <div class="flex flex-col items-center">
                <div class="relative mb-4">
                    <div class="absolute -inset-4 bg-gradient-to-r from-green-500 to-blue-500 rounded-full blur opacity-75"></div>
                    <div class="relative flex items-center justify-center bg-slate-800 w-20 h-20 rounded-full">
                        <i class="fas fa-futbol text-3xl text-white"></i>
                    </div>
                </div>
                <h1 class="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-blue-500 mb-2">Football Predictor Pro</h1>
                <p class="text-slate-400 mb-1">by Chiogor Ike</p>
                <p class="text-slate-400 max-w-2xl mx-auto">AI-powered match predictions with advanced analytics.</p>
            </div>
        </header>

        <main class="relative z-10">
            <div class="glass-card rounded-2xl overflow-hidden mb-10 fade-in">
                <div class="p-8">
                    <form method="POST" class="space-y-8">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div class="team-selector">
                                <label for="home_team" class="block text-sm font-medium text-slate-300 mb-3">
                                    <i class="fas fa-home mr-2 text-green-400"></i>Home Team
                                </label>
                                <div class="relative">
                                    <select id="home_team" name="home_team" required class="w-full px-5 py-4 text-slate-200 bg-slate-800 border border-slate-700 rounded-xl appearance-none focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all">
                                        <option value="" class="text-slate-400">Select Home Team</option>
                                        {% for team in teams %}
                                            <option value="{{ team }}" {% if home_team == team %}selected{% endif %}>{{ team }}</option>
                                        {% endfor %}
                                    </select>
                                    <div class="absolute inset-y-0 right-0 flex items-center pr-5 pointer-events-none">
                                        <i class="fas fa-chevron-down text-slate-400"></i>
                                    </div>
                                </div>
                            </div>

                            <div class="team-selector">
                                <label for="away_team" class="block text-sm font-medium text-slate-300 mb-3">
                                    <i class="fas fa-route mr-2 text-blue-400"></i>Away Team
                                </label>
                                <div class="relative">
                                    <select id="away_team" name="away_team" required class="w-full px-5 py-4 text-slate-200 bg-slate-800 border border-slate-700 rounded-xl appearance-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all">
                                        <option value="" class="text-slate-400">Select Away Team</option>
                                        {% for team in teams %}
                                            <option value="{{ team }}" {% if away_team == team %}selected{% endif %}>{{ team }}</option>
                                        {% endfor %}
                                    </select>
                                    <div class="absolute inset-y-0 right-0 flex items-center pr-5 pointer-events-none">
                                        <i class="fas fa-chevron-down text-slate-400"></i>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div>
                            <label class="block text-sm font-medium text-slate-300 mb-3">
                                <i class="fas fa-brain mr-2 text-purple-400"></i>Prediction Model
                            </label>
                            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                <div>
                                    <input type="radio" id="model_rf" name="model_type" value="rf" {% if model_type == 'rf' %}checked{% endif %} class="hidden peer">
                                    <label for="model_rf" class="flex items-center justify-between p-5 w-full text-slate-200 bg-slate-800 border border-slate-700 rounded-xl cursor-pointer peer-checked:border-green-500 peer-checked:bg-gradient-to-r peer-checked:from-green-900/30 peer-checked:to-slate-800/50 hover:bg-slate-700/50 transition-all">
                                        <div class="flex items-center">
                                            <div class="w-10 h-10 rounded-lg bg-green-900/50 flex items-center justify-center mr-4">
                                                <i class="fas fa-tree text-green-400"></i>
                                            </div>
                                            <div>
                                                <div class="font-medium">Random Forest</div>
                                                <div class="text-sm text-slate-400">Traditional ML model</div>
                                            </div>
                                        </div>
                                        <i class="fas fa-check-circle text-green-400 opacity-0 peer-checked:opacity-100 transition-opacity"></i>
                                    </label>
                                </div>
                                <div>
                                    <input type="radio" id="model_nn" name="model_type" value="nn" {% if model_type == 'nn' %}checked{% endif %} class="hidden peer">
                                    <label for="model_nn" class="flex items-center justify-between p-5 w-full text-slate-200 bg-slate-800 border border-slate-700 rounded-xl cursor-pointer peer-checked:border-blue-500 peer-checked:bg-gradient-to-r peer-checked:from-blue-900/30 peer-checked:to-slate-800/50 hover:bg-slate-700/50 transition-all">
                                        <div class="flex items-center">
                                            <div class="w-10 h-10 rounded-lg bg-blue-900/50 flex items-center justify-center mr-4">
                                                <i class="fas fa-network-wired text-blue-400"></i>
                                            </div>
                                            <div>
                                                <div class="font-medium">Neural Network</div>
                                                <div class="text-sm text-slate-400">Deep learning approach</div>
                                            </div>
                                        </div>
                                        <i class="fas fa-check-circle text-blue-400 opacity-0 peer-checked:opacity-100 transition-opacity"></i>
                                    </label>
                                </div>
                            </div>
                        </div>

                        <div class="pt-4">
                            <button type="submit" class="w-full flex justify-center items-center px-8 py-4 border border-transparent rounded-xl shadow-lg text-lg font-semibold text-white bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-all duration-300 hover:shadow-xl active:scale-95 transform hover:-translate-y-1 pulse">
                                <i class="fas fa-chart-line mr-3"></i> Predict Match Outcome
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            {% if result %}
            <div class="glass-card rounded-2xl overflow-hidden mb-10 fade-in">
                <div class="p-8">
                    <div class="flex flex-col items-center mb-8">
                        <h2 class="text-3xl font-bold text-white mb-3 text-center">{{ home_team }} vs {{ away_team }}</h2>
                        <div class="flex items-center">
                            <span class="px-4 py-1.5 rounded-full text-sm font-medium {% if result.model_type == 'Random Forest' %}bg-green-900/50 text-green-300 border border-green-800{% else %}bg-blue-900/50 text-blue-300 border border-blue-800{% endif %}">
                                {{ result.model_type }}
                            </span>
                            <span class="mx-3 text-slate-500">•</span>
                            <span class="text-slate-400">Predicted on {{ now.strftime('%Y-%m-%d') }}</span>
                        </div>
                    </div>

                    <div class="bg-gradient-to-r from-green-900/30 to-blue-900/30 rounded-xl p-8 mb-10 relative overflow-hidden">
                        <div class="absolute -right-10 -top-10 w-40 h-40 bg-blue-500 rounded-full filter blur-3xl opacity-10"></div>
                        <div class="absolute -left-10 -bottom-10 w-40 h-40 bg-green-500 rounded-full filter blur-3xl opacity-10"></div>
                        
                        <div class="flex items-center justify-between mb-6">
                            <h3 class="text-xl font-semibold text-white">
                                <i class="fas fa-bullseye mr-3 text-green-400"></i>Predicted Outcome
                            </h3>
                            <span class="px-4 py-1.5 rounded-full text-sm font-medium bg-green-900/50 text-green-300 border border-green-800">
                                {{ result.max_prob }} confidence
                            </span>
                        </div>
                        <p class="text-4xl font-bold text-center py-6 bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-blue-400">
                            {{ result.prediction }}
                        </p>
                    </div>

                    <div class="mb-10">
                        <h3 class="text-xl font-semibold text-white mb-6">
                            <i class="fas fa-percentage mr-3 text-purple-400"></i>Outcome Probabilities
                        </h3>
                        <div class="space-y-6">
                            {% for outcome, prob in result.probabilities.items() %}
                            <div class="space-y-2">
                                <div class="flex justify-between">
                                    <span class="font-medium text-slate-300">{{ outcome }}</span>
                                    <span class="font-semibold text-white">{{ prob }}</span>
                                </div>
                                <div class="w-full bg-slate-800 rounded-full h-3">
                                    <div class="probability-bar h-3 rounded-full {% if 'win' in outcome %}bg-gradient-to-r from-green-400 to-green-600{% else %}bg-gradient-to-r from-yellow-400 to-yellow-600{% endif %}" style="width: {{ prob }}"></div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>

                    <div>
                        <h3 class="text-xl font-semibold text-white mb-6">
                            <i class="fas fa-check-circle mr-3 text-blue-400"></i>Model Reliability Analysis
                        </h3>
                        <div class="text-slate-400 mb-6">
                            Based on historical predictions with similar confidence levels.
                        </div>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                            {% for outcome, data in result.reliability.items() %}
                            <div class="glass-card p-6 rounded-xl">
                                <div class="font-semibold text-white mb-3">{{ outcome }}</div>
                                <div class="text-sm text-slate-400 mb-4">
                                    Model predicted <span class="font-medium text-white">{{ "%.1f"|format(data.predicted_prob * 100) }}%</span>, 
                                    actual outcome was <span class="font-medium text-white">{{ data.reliability }}</span> 
                                    (based on {{ data.count }} similar predictions)
                                </div>
                                <div class="space-y-2">
                                    <div class="flex justify-between text-xs text-slate-400">
                                        <span>Calibration</span>
                                        <span>{{ data.reliability }} accuracy</span>
                                    </div>
                                    <div class="w-full bg-slate-800 rounded-full h-2">
                                        <div class="h-2 rounded-full {% if (data.actual_prob - data.predicted_prob)|abs < 0.1 %}bg-green-500{% elif data.actual_prob > data.predicted_prob %}bg-blue-500{% else %}bg-yellow-500{% endif %}" style="width: {{ (data.actual_prob * 100)|round(1) }}%"></div>
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}
        </main>

        <footer class="mt-16 text-center text-slate-500 text-sm">
            <div class="flex justify-center space-x-6 mb-4">
                <a href="#" class="text-slate-400 hover:text-white transition-colors"><i class="fab fa-twitter"></i></a>
                <a href="#" class="text-slate-400 hover:text-white transition-colors"><i class="fab fa-github"></i></a>
                <a href="#" class="text-slate-400 hover:text-white transition-colors"><i class="fab fa-linkedin"></i></a>
            </div>
            <p>Football Predictor Pro © {{ now.year }} | Created by Chiogor Ike</p>
            <p class="mt-1">Data sourced from football-data.co.uk</p>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(() => {
                document.querySelectorAll('.probability-bar').forEach(bar => {
                    bar.style.width = bar.style.width;
                });
            }, 100);
        });
    </script>
</body>
</html>
'''

# Application Setup
def initialize_application():
    global models, team_stats
    
    print("Loading and preprocessing data...")
    df = load_data(URLS)
    df = preprocess_data(df)
    
    if df is None:
        raise ValueError("Failed to load and preprocess data")
    
    print("Engineering features...")
    X, y, team_stats = engineer_features(df)
    
    if X is None or team_stats is None:
        raise ValueError("Failed to engineer features")
    
    print("Training models...")
    models = {
        'rf': train_random_forest(X, y),
        'nn': train_neural_network(X, y)
    }
    
    if not save_models(models, team_stats):
        raise ValueError("Failed to save models")
    
    print("Models trained and saved successfully")

def handle_shutdown(signum, frame):
    print("\nServer shutting down gracefully...")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_shutdown)
    
    try:
        models, team_stats = load_models()
        if models is None or team_stats is None:
            print("No valid models found, training new ones...")
            initialize_application()
            models, team_stats = load_models()
            
        print("\nServer starting...")
        print("Access the application at: http://localhost:5000")
        app.run(debug=True, host='localhost', port=5000)
    except Exception as e:
        print(f"\nFailed to start application: {str(e)}\n")
        sys.exit(1)