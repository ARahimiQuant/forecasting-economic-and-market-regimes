import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score


def plot_feature_importance(feat_impo_df: pd.DataFrame,
                            group_impo_df: pd.DataFrame,
                            plot_title: str,
                            bar_plot_title: str,
                            pie_plot_title: str,
                            start_feat_idx: int, 
                            max_feat_no: int) -> go.Figure:
    """
    Create a composite plot displaying feature importance as bar charts and group importance as a pie chart.
    
    Parameters:
        feat_impo_df (pd.DataFrame): DataFrame containing feature importance data.
        group_impo_df (pd.DataFrame): DataFrame containing group importance data.
        plot_title (str): Title of the composite plot.
        bar_plot_title (str): Title of the bar plot.
        pie_plot_title (str): Title of the pie plot.
        start_feat_idx (int): Starting index of selected features.
        max_feat_no (int): Maximum number of features to include.
    
    Returns:
        go.Figure: A Plotly figure containing the composite plot.
    """
    # Define group colors
    group_colors = {'Output and Income': '#1f77b4',
                    'Labor Market': '#ff7f0e',
                    'Housing': '#2ca02c',
                    'Consumption, Orders and Inventories': '#d62728',
                    'Money and Credit': '#9467bd',
                    'Interest and Exchange Rates': '#8c564b',
                    'Prices': '#e377c2',
                    'Stock Market': '#7f7f7f'}

    # Create subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1, column_widths=[0.85, 0.15], 
                        specs=[[{"type": "bar"}, {"type": "pie"}]], 
                        subplot_titles=(f'<b> {bar_plot_title} </b>', f'<b> {pie_plot_title} </b>'))

    # Feature importance bar plot
    end_feature = start_feat_idx + max_feat_no - 1
    selected_features = feat_impo_df['Feature'][start_feat_idx - 1:end_feature]
    selected_importance = 100 * feat_impo_df['Importance'][start_feat_idx - 1:end_feature]
    selected_group = feat_impo_df['Group'][start_feat_idx - 1:end_feature]
    for idx, group in enumerate(selected_group):
        color = group_colors[group]
        fig.add_trace(go.Bar(x=[selected_features.iloc[idx]], y=[selected_importance.iloc[idx]], showlegend=False,
                             marker_color=color, name=group), row=1, col=1)
    
    # Bar plot axis names
    fig.update_xaxes(title_text='<b>Features</b>', row=1, col=1)
    fig.update_yaxes(title_text='<b>Importance (%)</b>', row=1, col=1)

    # Group importance pie chart
    fig.add_trace(go.Pie(labels=group_impo_df['Group'], 
                         values=group_impo_df['Importance'],
                         marker_colors=[group_colors[group] for group in group_impo_df['Group']],
                         hole=0.5, textinfo='percent+label',
                         insidetextfont={'size': 14}, showlegend=False, 
                         ), row=1, col=2)

    # Update layout and formatting
    fig.update_layout(title=f'<b> {plot_title} </b>', title_x=0.5, title_font_size=22)
    fig.update_layout(template='simple_white', width=1300, height=600)
    
    return fig


def evaluation_metrics(pred_df: pd.DataFrame, 
                       model_cols: list,
                       y_true_col: str,
                       show_plot: bool = True,
                       plot_width: int = 700, 
                       plot_height: int = 400) -> tuple:
    """
    Calculate and visualize evaluation metrics for different models.

    Parameters:
        pred_df (pd.DataFrame): DataFrame containing predicted probabilities and true labels.
        model_cols (list): List of column names representing different models in pred_df.
        y_true_col (str): Column name representing true class labels in pred_df.
        show_plot (bool, optional): Whether to show the plot. Default is True.
        plot_width (int, optional): Width of the plot. Default is 700.
        plot_height (int, optional): Height of the plot. Default is 400.
    
    Returns:
        tuple: A tuple containing the metrics DataFrame and the Plotly figure.
    """
    # List of metric functions to calculate
    metric_functions = [accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score]
    metric_results = {}

    # Calculate metrics for each model
    for model in model_cols:
        # Convert predicted probabilities to predicted classes (0 or 1)
        predicted_class = np.where(pred_df[model] > 0.5, 1, 0)
        true_class = pred_df[y_true_col]
        # Calculate each metric for the current model
        metric_results[model] = [metric(true_class, predicted_class) for metric in metric_functions]

    # Create a DataFrame to store the calculated metrics
    metrics_df = pd.DataFrame(metric_results, index=['Accuracy', 'B-Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'ROC-AUC'])

    # Create traces for each model's metric values
    traces = [go.Bar(x=metrics_df.index, y=metrics_df[model], name=model, orientation='v') for model in model_cols]
    # Create the layout for the figure
    layout = go.Layout(barmode='group', autosize=False, width=plot_width, height=plot_height, template='simple_white', title_text='<b> Evaluation Metrics </b>', title_x=0.5)
    # Create the Plotly figure
    fig = go.Figure(data=traces, layout=layout)
    
    # Show the plot if requested
    if show_plot:
        fig.show()
        
    # Return the metrics DataFrame and the Plotly figure
    return metrics_df, fig


def plot_confusion_matrix(y_true: list, 
                          y_pred: list, 
                          class_names: list = ['Normal', 'Recession'], 
                          title: str = 'RF',
                          width: int = 450,
                          height: int = 450,
                          annotation_text_size: int = 14) -> go.Figure:
    """
    Create an annotated heatmap of the confusion matrix.
    
    Parameters:
        y_true (list): True class labels.
        y_pred (list): Predicted class labels.
        class_names (list, optional): Names of classes. Default is ['Normal', 'Recession'].
        title (str, optional): Plot title. Default is 'RF'.
        width (int, optional): Width of the plot. Default is 450.
        height (int, optional): Height of the plot. Default is 450.
        annotation_text_size (int, optional): Font size of annotations. Default is 14.
    
    Returns:
        go.Figure: A Plotly figure representing the annotated heatmap.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Assign class names
    x = class_names
    y = class_names

    # Convert confusion matrix elements to strings for annotations
    cm_text = [[str(y) for y in x] for x in cm]

    # Create a figure for annotated heatmap
    fig = ff.create_annotated_heatmap(cm, x=x, y=y, annotation_text=cm_text, colorscale='Blues')
    
    # Adjust x and y axis labels and title
    fig.update_layout(xaxis=dict(title="<b> Predicted Class </b>", side="bottom"), 
                      yaxis=dict(title="<b> Actual Class </b>"),
                      title_text=f'<b> {title} </b>', title_x=0.53)
    
    # Set figure size and theme
    fig.update_layout(autosize=False, width=width, height=height, template='simple_white')
    
    # Change annotation text size
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = annotation_text_size
        
    return fig