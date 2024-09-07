# plot functions
import plotly.graph_objects as go
import pandas as pd


def plot_scores(score_df: pd.DataFrame, x_col: str, y_col: str, label_col: str, xaxis_type: str='log') -> go.Figure:
    """
    plot scores with respect to parameters and labels
    @param score_df: dataframe with scores
    @param x_col: column name for x-axis
    @param y_col: column name for y-axis
    @param label_col: column name to label separate lines
    @param xaxis_type: type of x-axis: linear, log, or reverselog
    @return: plotly figure
    """
    fig = go.Figure()

    reverselog = xaxis_type == 'reverselog'

    if reverselog:
        # get x values
        x_values = score_df[x_col].unique()
        max_x = round(max(x_values))

    for label in score_df[label_col].unique():
        label_df = score_df[score_df[label_col] == label]
        label_name = f"{label_col}={label}"

        if reverselog:
            x_plot = max_x - label_df[x_col]
        else:
            x_plot = label_df[x_col]

        fig.add_trace(go.Scatter(x=x_plot, y=label_df[y_col], mode='lines+markers', name=label_name))
            
    fig.update_layout(
        title=f'{y_col} vs {x_col} for different {label_col} values',
        xaxis_type=xaxis_type if xaxis_type != 'reverselog' else 'log',
        xaxis_title=x_col,
        yaxis_title=y_col
    )

    if reverselog:
        fig.update_xaxes(autorange='reversed', ticktext=x_values, tickvals=max_x - x_values)

    return fig
