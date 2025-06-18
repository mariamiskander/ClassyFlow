#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from datetime import datetime
import base64
from io import BytesIO

import pickle
import xgboost as xgb
import dataframe_image as dfi
from jinja2 import Environment, FileSystemLoader

## Static Variables: File Formatting
classColumn = "${params.classifed_column_name}" # 'Classification'
leEncoderFile = "${params.output_dir}/models/classes.npy"

############################ HTML REPORTING ############################
def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    return base64.b64encode(plot_data).decode()

def image_to_base64(image_path):
    """Convert image file to base64 string for embedding"""
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file '{image_path}' not found.")
        return None

def create_default_letterhead():
    """Create a default letterhead image"""
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    
    ax.text(5, 1.2, 'CLASSYFLOW PIPELINE', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            color='#667eea')
    ax.text(5, 0.7, 'Machine Learning Model Evaluation', 
            fontsize=14, ha='center', va='center',
            color='#495057', style='italic')
    
    ax.axhline(y=0.3, color='#667eea', linewidth=3, alpha=0.7)
    ax.axhline(y=1.7, color='#667eea', linewidth=3, alpha=0.7)
    
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    return plot_to_base64(fig)

def generate_html_report(model_name, results):
    """Generate HTML report using template"""
    template_dir = "${projectDir}/html_templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("holdout_evaluation_template.html")
    
    # Handle letterhead
    letterhead_path = "${projectDir}/images/ClassyFlow_Letterhead.PNG"
    letterhead_image = image_to_base64(letterhead_path)
    if not letterhead_image:
        letterhead_image = create_default_letterhead()
    
    # Prepare template data
    report_data = {
        'model_name': model_name.replace('.pkl', ''),
        'report_title': 'Holdout Evaluation: XGBoost',
        'pipeline_version': '1.0.0',
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'letterhead_image': letterhead_image,
        # Performance metrics
        'accuracy': results['accuracy'],
        'balanced_accuracy': results['balanced_accuracy'],
        'f1_score': results['f1_weighted'],
        'max_auc_score': results['max_auc'][1],
        'max_auc_class': results['max_auc_class'],
        'min_auc_score': results['min_auc'][1],
        'min_auc_class': results['min_auc_class'],
        'n_classes': results['n_classes'],
        # Plots
        'class_distribution_plot': results['class_distribution_plot'],
        'confusion_matrix_plot': results['confusion_matrix_plot'],
        'roc_curves_plot': results['roc_curves_plot'],
        'auc_rankings_table': results['auc_rankings_table'],
        'class_imbalance_detected': results['class_imbalance_detected'],
    }
    
    return template.render(**report_data)
############################ HTML REPORTING ############################

def create_class_distribution_plot(unique_values, counts, label_hash):
    """Create class distribution bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = [label_hash[val] for val in unique_values]
    
    # Create horizontal bar chart
    bars = ax.barh(class_names, counts, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of Cells')
    #ax.set_title('Class Distribution in Holdout Dataset')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
               f'{count}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return plot_to_base64(fig)

def create_confusion_matrix_plot(cm_df, class_names):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               ax=ax, cbar_kws={'label': 'Number of Predictions'})
    
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    #ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return plot_to_base64(fig)

def create_roc_curves_plot(y_true_binarized, y_pred_binarized, label_hash, n_classes):
    """Create ROC curves for all classes"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute AUC for each class and sort
    auc_scores = {}
    for i in range(n_classes):
        if i < y_true_binarized.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[i] = roc_auc
    
    # Sort by AUC score (descending)
    sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Plot ROC curves
    colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_auc_scores)))
    for (class_index, roc_auc), color in zip(sorted_auc_scores, colors):
        if class_index in label_hash:
            fpr, tpr, _ = roc_curve(y_true_binarized[:, class_index], y_pred_binarized[:, class_index])
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{label_hash[class_index]} (AUC = {roc_auc:.3f})')
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    #ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)

def create_auc_rankings_table(sorted_auc_scores, label_hash):
    """Create a DataFrame with AUC rankings"""
    rankings_data = []
    for rank, (class_index, auc_score) in enumerate(sorted_auc_scores, 1):
        if class_index in label_hash:
            # Determine performance level
            if auc_score >= 0.9:
                performance = '<span class="auc-score auc-excellent">Excellent</span>'
            elif auc_score >= 0.8:
                performance = '<span class="auc-score auc-good">Good</span>'
            elif auc_score >= 0.7:
                performance = '<span class="auc-score auc-fair">Fair</span>'
            else:
                performance = '<span class="auc-score auc-poor">Poor</span>'
            
            rankings_data.append({
                'Rank': rank,
                'Class': label_hash[class_index],
                'AUC Score': f'{auc_score:.3f}',
                'Performance': performance
            })
    
    rankings_df = pd.DataFrame(rankings_data)
    return rankings_df.to_html(classes='table', table_id='auc_rankings', escape=False, index=False)


def detect_class_imbalance(counts, threshold=0.1):
    """Detect if there's significant class imbalance"""
    total_samples = sum(counts)
    min_ratio = min(counts) / total_samples
    max_ratio = max(counts) / total_samples
    
    # Consider imbalanced if smallest class is less than threshold of total
    # or if ratio between largest and smallest is > 10:1
    return min_ratio < threshold or (max_ratio / min_ratio) > 10

def check_holdout(toCheckDF, xgbM):
    all_results = {}
    X_holdout = toCheckDF[list(toCheckDF.select_dtypes(include=[np.number]).columns.values)]
    X_holdout = X_holdout[xgbM.feature_names]

    le = preprocessing.LabelEncoder()
    le.classes_ = np.load(leEncoderFile, allow_pickle=True)
    print(le.classes_.tolist())
    y_holdout = le.transform(toCheckDF[classColumn])

    # Make predictions
    dmatrix = xgb.DMatrix(X_holdout)
    y_pred_proba = xgbM.predict(dmatrix)

    all_results['accuracy'] = accuracy_score(y_holdout, y_pred_proba)
    all_results['f1_weighted'] = f1_score(y_holdout, y_pred_proba, average='weighted')
    all_results['balanced_accuracy'] = balanced_accuracy_score(y_holdout, y_pred_proba)

    # Get unique values and their counts
    unique_values, counts = np.unique(y_holdout, return_counts=True)
    n_classes = len(unique_values)
    all_results['n_classes'] = n_classes
    print(n_classes)
    uniqNames = le.inverse_transform(unique_values)   
    lableHash = dict(zip(unique_values, uniqNames))

    # Detect class imbalance
    all_results['class_imbalance_detected'] = detect_class_imbalance(counts)

    # Create class distribution plot
    all_results['class_distribution_plot'] = create_class_distribution_plot(
        unique_values, counts, lableHash)

    # Calculate confusion matrix
    cm = confusion_matrix(y_holdout, y_pred_proba)
    cm_df = pd.DataFrame(cm, columns=uniqNames, index=uniqNames)
    all_results['confusion_matrix_plot'] = create_confusion_matrix_plot(cm_df, uniqNames)

    # Multiclass case
    y_true_binarized = label_binarize(y_holdout, classes=np.arange(n_classes))
    y_pred_binarized = label_binarize(y_pred_proba, classes=np.arange(n_classes))

    # Compute AUC scores and rankings
    auc_scores = {}
    for i in range(n_classes):
        if i < y_true_binarized.shape[1]:
            fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[i] = roc_auc

    # Sort AUC scores
    sorted_auc_scores = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
    
    all_results['max_auc'] = sorted_auc_scores[0]
    all_results['min_auc'] = sorted_auc_scores[-1]
    all_results['max_auc_class'] = lableHash[sorted_auc_scores[0][0]]
    all_results['min_auc_class'] = lableHash[sorted_auc_scores[-1][0]]

    # Create ROC curves plot
    all_results['roc_curves_plot'] = create_roc_curves_plot(
        y_true_binarized, y_pred_binarized, lableHash, n_classes)

    # Create AUC rankings table
    all_results['auc_rankings_table'] = create_auc_rankings_table(
        sorted_auc_scores, lableHash)
    
    return all_results

if __name__ == "__main__":
    modelName = "${model_pickle}" #"XGBoost_Model_Second.pkl"
    myData = pd.read_pickle("${holdoutDataframe}") 

    with open("${select_features_csv}", 'r') as file:
        next(file) # Skip header
        featureList = file.readlines()

    featureList = list(set([line.strip() for line in featureList]))
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(classColumn)
    focusData = myData[featureList]

    with open(modelName, 'rb') as file:
        xgbMdl = pickle.load(file)
    
    eval_results = check_holdout(focusData, xgbMdl)
    
    # Generate HTML report instead of PDF
    html_report = generate_html_report(modelName, eval_results)
    
    # Save HTML report
    html_filename = "Holdout_on_{}.html".format(modelName.replace(".pkl",""))
    with open(html_filename, "w") as f:
        f.write(html_report)
    
    print(f"HTML report generated: {html_filename}")
    
    # Create a DataFrame from the presence dictionary (same as original)
    performanceDF = pd.DataFrame({
        "Model": modelName,
        "Accuracy": eval_results['accuracy'], 
        "Max AUC": eval_results['max_auc'][-1], 
        "Min AUC": eval_results['min_auc'][-1] }, index=[0])

    # Save the presence DataFrame to a CSV file (same as original)
    performanceDF.to_csv("holdout_{}.csv".format(modelName.replace(".pkl","")), index=False)