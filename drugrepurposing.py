import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====== LOAD DATA ======
print("Loading CSV...")
df = pd.read_csv(r"C:\Users\eldri\OneDrive\Desktop\finaldataset (1).csv", low_memory=False)

print(f"Columns: {df.columns.tolist()}")
print(f"Initial dataset size: {len(df)}")

# ====== REMOVE DUPLICATES ======
print("\nChecking for duplicates...")
initial_rows = len(df)
df = df.drop_duplicates(subset=['drugbank_id', 'diseasename'])
print(f"Removed {initial_rows - len(df)} duplicate rows")
print(f"Remaining rows: {len(df)}")

# ====== SET TARGET ======
TARGET = "diseasename"

if TARGET not in df.columns:
    raise ValueError(f"TARGET column '{TARGET}' not found in dataframe!")

print(f"\nUsing TARGET = {TARGET}")
print(f"Number of unique diseases: {df[TARGET].nunique()}")

# ====== REDUCE TO TOP DISEASES FOR FASTER TRAINING ======
print("\nReducing to most common diseases for faster training...")
min_samples = 20
vc = df[TARGET].value_counts()

# Keep only top 500 most common diseases
top_diseases = vc.head(500).index
df = df[df[TARGET].isin(top_diseases)].copy()
print(f"Kept top 500 diseases")
print(f"Remaining rows: {len(df)}")
print(f"Remaining classes: {df[TARGET].nunique()}")

# ====== REMOVE RARE CLASSES ======
print("\nRemoving classes with fewer than 20 samples...")
vc = df[TARGET].value_counts()
rare = vc[vc < min_samples].index
df = df[~df[TARGET].isin(rare)].copy()
print(f"Removed {len(rare)} rare classes")
print(f"Final rows: {len(df)}")
print(f"Final classes: {df[TARGET].nunique()}")

# ====== SAMPLE DATASET FOR SPEED ======
print("\nSampling dataset for faster training...")
if len(df) > 30000:
    df = df.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(min(len(x), max(20, int(30000 * len(x) / len(df)))), random_state=42)
    ).reset_index(drop=True)
    print(f"Sampled to {len(df)} rows")
    
    vc_sampled = df[TARGET].value_counts()
    if vc_sampled.min() < min_samples:
        print("Some classes dropped below minimum after sampling. Removing them...")
        rare_after_sample = vc_sampled[vc_sampled < min_samples].index
        df = df[~df[TARGET].isin(rare_after_sample)].copy()
        print(f"Final dataset size: {len(df)}")
        print(f"Final class count: {df[TARGET].nunique()}")

# ====== REMOVE DATA LEAKAGE AND PROBLEMATIC COLUMNS ======
print("\nRemoving problematic columns...")

cols_to_drop = [
    'diseaseid', 
    'diseaseid_norm',
    'chemicalname',
    'chemicalid',
    'pubmedids',
    'description',
    'smiles_smiles',
    'smiles_lipinski',
    'cas',
    'name',
    'omimids',
    'inferencegenesymbol',
    'directevidence',
]

existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
print(f"Dropping columns: {existing_cols_to_drop}")
df = df.drop(columns=existing_cols_to_drop)

# ====== CLEAN MISSING VALUES BEFORE ENCODING ======
print("\nHandling missing values...")
for col in df.columns:
    if col != TARGET and df[col].dtype == "object":
        df[col] = df[col].astype(str).fillna("missing")

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(0)

# ====== ENCODE TARGET ======
print("\nEncoding target variable...")
le = LabelEncoder()
df[TARGET] = le.fit_transform(df[TARGET].astype(str))

# Verify class distribution after encoding
print(f"Class distribution after encoding:")
encoded_vc = df[TARGET].value_counts()
print(f"Number of classes: {len(encoded_vc)}")
print(f"Min samples per class: {encoded_vc.min()}")
print(f"Max samples per class: {encoded_vc.max()}")

# ====== DEFINE FEATURES ======
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# ====== IDENTIFY AND PROCESS CATEGORICAL FEATURES ======
print("\nProcessing categorical features...")
cat_cols = [c for c in X.columns if X[c].dtype == "object"]

cols_to_remove = []
for c in cat_cols:
    X[c] = X[c].astype(str)
    nunique = X[c].nunique()
    print(f"  {c}: {nunique} unique values")
    
    if nunique > 500:
        print(f"    ⚠️  Marking {c} for removal - too many unique values ({nunique})")
        cols_to_remove.append(c)

if cols_to_remove:
    print(f"\nRemoving {len(cols_to_remove)} high-cardinality columns...")
    X = X.drop(columns=cols_to_remove)
    cat_cols = [c for c in cat_cols if c not in cols_to_remove]

cat_features_indices = [X.columns.get_loc(c) for c in cat_cols]

print(f"\nFinal categorical features count: {len(cat_features_indices)}")
print(f"Final numerical features count: {len(X.columns) - len(cat_features_indices)}")
print(f"Final feature count: {len(X.columns)}")
if cat_cols:
    print(f"Categorical features: {cat_cols}")

# Force garbage collection
gc.collect()

# ====== FINAL CLASS CHECK BEFORE SPLIT ======
print("\n" + "="*50)
print("FINAL CLASS DISTRIBUTION CHECK")
print("="*50)
final_vc = y.value_counts()
print(f"Total samples: {len(y)}")
print(f"Total classes: {len(final_vc)}")
print(f"Min samples per class: {final_vc.min()}")
print(f"Max samples per class: {final_vc.max()}")
print(f"\nTop 10 classes by count:")
print(final_vc.head(10))

# ====== TRAIN/TEST SPLIT WITH ERROR HANDLING ======
print("\n" + "="*50)
print("Splitting data...")
print("="*50)

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Split successful!")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    
except ValueError as e:
    print(f"❌ Stratified split failed: {e}")
    print("\nAttempting non-stratified split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    print(f"✓ Non-stratified split successful!")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

print(f"\nClasses in train set: {len(np.unique(y_train))}")
print(f"Classes in test set: {len(np.unique(y_test))}")

X_train = X_train.copy()
X_test = X_test.copy()

# ====== CATBOOST TRAIN WITH FASTER SETTINGS ======
print("\n" + "="*50)
print("Initializing CatBoost model (faster settings)...")
print("="*50)

model = CatBoostClassifier(
    iterations=50,
    depth=4,
    learning_rate=0.15,
    loss_function="MultiClass",
    random_seed=42,
    verbose=10
)

print("\nCreating data pools...")
train_pool = Pool(
    X_train, 
    y_train, 
    cat_features=cat_features_indices if cat_features_indices else None
)

test_pool = Pool(
    X_test, 
    y_test, 
    cat_features=cat_features_indices if cat_features_indices else None
)

print("\nTraining CatBoost model...")
print("(This should take 30-60 minutes...)")

try:
    model.fit(
        train_pool, 
        eval_set=test_pool,
        early_stopping_rounds=10
    )
    
    print("\n" + "="*50)
    print("✓ Training complete!")
    print("="*50)
    
    # ====== EVALUATION ======
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Calculate Top-K accuracies
    top_3_acc = top_k_accuracy_score(y_test, y_pred_proba_test, k=3)
    top_5_acc = top_k_accuracy_score(y_test, y_pred_proba_test, k=5)
    top_10_acc = top_k_accuracy_score(y_test, y_pred_proba_test, k=10)
    top_20_acc = top_k_accuracy_score(y_test, y_pred_proba_test, k=20)
    top_50_acc = top_k_accuracy_score(y_test, y_pred_proba_test, k=50)
    
    # Calculate random baseline
    num_classes = len(np.unique(y_train))
    random_baseline_top1 = 1.0 / num_classes
    random_baseline_top3 = min(3.0 / num_classes, 1.0)
    random_baseline_top5 = min(5.0 / num_classes, 1.0)
    random_baseline_top10 = min(10.0 / num_classes, 1.0)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Total disease classes: {num_classes}")
    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Overfitting gap: {abs(train_accuracy - test_accuracy):.4f}")
    print(f"\nTop-3 Test Accuracy:  {top_3_acc:.4f}")
    print(f"Top-5 Test Accuracy:  {top_5_acc:.4f}")
    print(f"Top-10 Test Accuracy: {top_10_acc:.4f}")
    print(f"Top-20 Test Accuracy: {top_20_acc:.4f}")
    print(f"Top-50 Test Accuracy: {top_50_acc:.4f}")
    
    # ====== FEATURE IMPORTANCE ======
    print(f"\n{'='*50}")
    print("TOP 15 MOST IMPORTANT FEATURES")
    print(f"{'='*50}")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(15).to_string(index=False))
    
    # ====== CREATE VISUALIZATIONS ======
    print(f"\n{'='*50}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*50}")
    
    # Create figure directory
    import os
    os.makedirs('model_visualizations', exist_ok=True)
    
    # 1. TOP-K ACCURACY COMPARISON PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values = [1, 3, 5, 10, 20, 50]
    accuracies = [test_accuracy, top_3_acc, top_5_acc, top_10_acc, top_20_acc, top_50_acc]
    random_baselines = [random_baseline_top1, random_baseline_top3, random_baseline_top5, 
                       random_baseline_top10, min(20.0/num_classes, 1.0), min(50.0/num_classes, 1.0)]
    
    x_pos = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, accuracies, width, label='Model Accuracy', color='#2E86AB')
    bars2 = ax.bar(x_pos + width/2, random_baselines, width, label='Random Baseline', color='#A23B72')
    
    ax.set_xlabel('Top-K Predictions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Top-K Accuracy Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Top-{k}' for k in k_values])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_visualizations/top_k_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: top_k_accuracy.png")
    plt.close()
    
    # 2. FEATURE IMPORTANCE BAR CHART (Top 15)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_features = feature_importance.head(15)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.1f}", 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.close()
    
    # 3. TRAIN VS TEST ACCURACY COMPARISON
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Training', 'Testing', 'Overfitting Gap']
    values = [train_accuracy, test_accuracy, abs(train_accuracy - test_accuracy)]
    colors_acc = ['#06D6A0', '#118AB2', '#EF476F']
    
    bars = ax.bar(categories, values, color=colors_acc, width=0.6)
    ax.set_ylabel('Accuracy / Gap', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Testing Performance', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_visualizations/train_vs_test.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: train_vs_test.png")
    plt.close()
    
    # 4. FEATURE IMPORTANCE PIE CHART BY CATEGORY
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Categorize features
    def categorize_feature(feat):
        feat_lower = feat.lower()
        if any(x in feat_lower for x in ['logp', 'solubility', 'molecular', 'weight']):
            return 'Chemical Properties'
        elif any(x in feat_lower for x in ['pka', 'acidic', 'basic']):
            return 'Ionization Properties'
        elif any(x in feat_lower for x in ['hba', 'hbd', 'ro5']):
            return 'Lipinski Descriptors'
        elif any(x in feat_lower for x in ['score', 'evidence']):
            return 'Evidence Scores'
        else:
            return 'Other Features'
    
    feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
    category_importance = feature_importance.groupby('category')['importance'].sum().sort_values(ascending=False)
    
    colors_pie = plt.cm.Set3(range(len(category_importance)))
    wedges, texts, autotexts = ax.pie(category_importance.values, 
                                       labels=category_importance.index,
                                       autopct='%1.1f%%',
                                       colors=colors_pie,
                                       startangle=90,
                                       textprops={'fontsize': 11})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Feature Importance by Category', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('model_visualizations/feature_categories.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_categories.png")
    plt.close()
    
    # 5. TOP-K ACCURACY LINE PLOT
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_values_extended = [1, 3, 5, 10, 20, 50]
    accuracies_extended = [test_accuracy, top_3_acc, top_5_acc, top_10_acc, top_20_acc, top_50_acc]
    
    ax.plot(k_values_extended, accuracies_extended, marker='o', linewidth=3, 
           markersize=10, color='#06D6A0', label='Model Performance')
    ax.plot(k_values_extended, [random_baseline_top1, random_baseline_top3, 
                                random_baseline_top5, random_baseline_top10,
                                min(20.0/num_classes, 1.0), min(50.0/num_classes, 1.0)], 
           marker='s', linewidth=2, linestyle='--', markersize=8, 
           color='#EF476F', label='Random Baseline')
    
    ax.set_xlabel('K (Number of Predictions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Top-K Accuracy Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for k, acc in zip(k_values_extended, accuracies_extended):
        ax.annotate(f'{acc:.3f}', xy=(k, acc), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('model_visualizations/topk_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: topk_curve.png")
    plt.close()
    
    # 6. SUMMARY METRICS DASHBOARD
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Key Metrics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    metrics_text = f"""
    MODEL SUMMARY - Drug Repurposing Prediction
    {'='*80}
    
    Dataset Statistics:
    • Total Disease Classes: {num_classes}
    • Training Samples: {len(X_train):,}
    • Testing Samples: {len(X_test):,}
    • Features Used: {len(X.columns)}
    
    Performance Metrics:
    • Training Accuracy: {train_accuracy:.4f}
    • Test Accuracy: {test_accuracy:.4f}
    • Top-3 Accuracy: {top_3_acc:.4f} (vs {random_baseline_top3:.4f} random)
    • Top-5 Accuracy: {top_5_acc:.4f} (vs {random_baseline_top5:.4f} random)
    • Top-10 Accuracy: {top_10_acc:.4f} (vs {random_baseline_top10:.4f} random)
    
    Model Quality:
    • Overfitting Gap: {abs(train_accuracy - test_accuracy):.4f} ({'Mild' if abs(train_accuracy - test_accuracy) < 0.1 else 'Moderate'})
    • Performs {(test_accuracy/random_baseline_top1):.1f}x better than random guessing
    """
    ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Subplot 2: Top accuracies bar
    ax2 = fig.add_subplot(gs[1, 0])
    top_k_names = ['Top-1', 'Top-3', 'Top-5', 'Top-10']
    top_k_vals = [test_accuracy, top_3_acc, top_5_acc, top_10_acc]
    bars = ax2.bar(top_k_names, top_k_vals, color=['#e63946', '#f77f00', '#06d6a0', '#118ab2'])
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Top-K Accuracy Summary', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot 3: Feature importance top 10
    ax3 = fig.add_subplot(gs[1, 1])
    top10 = feature_importance.head(10)
    ax3.barh(range(len(top10)), top10['importance'], color=plt.cm.plasma(np.linspace(0.2, 0.8, len(top10))))
    ax3.set_yticks(range(len(top10)))
    ax3.set_yticklabels(top10['feature'], fontsize=9)
    ax3.set_xlabel('Importance', fontweight='bold')
    ax3.set_title('Top 10 Features', fontweight='bold')
    ax3.invert_yaxis()
    
    # Subplot 4: Train vs Test
    ax4 = fig.add_subplot(gs[2, 0])
    performance = ['Train', 'Test']
    perf_vals = [train_accuracy, test_accuracy]
    bars = ax4.bar(performance, perf_vals, color=['#06d6a0', '#118ab2'], width=0.5)
    ax4.set_ylabel('Accuracy', fontweight='bold')
    ax4.set_title('Training vs Testing', fontweight='bold')
    ax4.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 5: Category breakdown
    ax5 = fig.add_subplot(gs[2, 1])
    if len(category_importance) > 0:
        ax5.pie(category_importance.values, labels=category_importance.index,
               autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3(range(len(category_importance))))
        ax5.set_title('Feature Categories', fontweight='bold')
    
    plt.suptitle('Drug Repurposing Model - Complete Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('model_visualizations/complete_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: complete_dashboard.png")
    plt.close()
    
    print(f"\n{'='*50}")
    print("✓ All visualizations saved to 'model_visualizations/' folder")
    print(f"{'='*50}")
    
    # ====== CLASSIFICATION REPORT ======
    print(f"\n{'='*50}")
    print("CLASSIFICATION REPORT (Top 10 classes)")
    print(f"{'='*50}")
    num_classes_to_show = min(10, len(le.classes_))
    target_names = le.inverse_transform(range(num_classes_to_show))
    print(classification_report(
        y_test, 
        y_pred_test, 
        target_names=target_names,
        labels=range(num_classes_to_show),
        zero_division=0
    ))
    
    print("\n" + "="*50)
    print("✓ Model training and evaluation complete!")
    print("="*50)
    
    # ====== SAVE MODEL ======
    save_choice = input("\nSave model? (yes/no): ").strip().lower()
    if save_choice == 'yes':
        model.save_model("drug_repurposing_model.cbm")
        print("✓ Model saved to 'drug_repurposing_model.cbm'")
        
        # Save label encoder
        import pickle
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        print("✓ Label encoder saved to 'label_encoder.pkl'")
        
        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("✓ Feature importance saved to 'feature_importance.csv'")

except MemoryError as e:
    print(f"\n❌ Memory Error: {str(e)}")
    print("\nReduce dataset size or increase RAM")
    raise
    
except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    raise