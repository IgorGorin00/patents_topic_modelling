# copypasted from github
# to customize the layout of the plots
# make it show all at once
# and return the

import itertools
import numpy as np
from typing import List

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_barchart(topic_model,
                       topics: List[int] = None,
                       top_n_topics: int = 8,
                       n_words: int = 5,
                       custom_labels: bool = False,
                       title: str = "<b>Topic Word Scores</b>",
                       width: int = 250,
                       height: int = 250,
                       columns_n: int = 5) -> go.Figure:
    """ Visualize a barchart of selected topics
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics: A selection of topics to visualize.
        top_n_topics: Only select the top n most frequent topics.
        n_words: Number of words to show in a topic
        custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
        title: Title of the plot.
        width: The width of each figure.
        height: The height of each figure.
    Returns:
        fig: A plotly figure
    Examples:
    To visualize the barchart of selected topics
    simply run:
    ```python
    topic_model.visualize_barchart()
    ```
    Or if you want to save the resulting figure:
    ```python
    fig = topic_model.visualize_barchart()
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/bar_chart.html"
    style="width:1100px; height: 660px; border: 0px;""></iframe>
    """
    colors = itertools.cycle(["#17baad", "#44546a", "#50b46e", "#79919c", "#55690f"])

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list()[0:6])

    # Initialize figure
    if topic_model.custom_labels_ is not None and custom_labels:
        subplot_titles = [topic_model.custom_labels_[topic + topic_model._outliers] for topic in topics]
    else:
        subplot_titles = [f"Topic {topic}" for topic in topics]
    columns = columns_n
    rows = int(np.ceil(len(topics) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    words_all = []
    scores_all = []
    row = 1
    column = 1
    for topic in topics:
        words = [word + "  " for word, _ in topic_model.get_topic(topic)][:n_words][::-1]
        scores = [score for _, score in topic_model.get_topic(topic)][:n_words][::-1]

        words_all.append(words)
        scores_all.append(scores)

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': f"{title}",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width*4,
        height=height*rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig, words_all, scores_all


import pandas as pd
from typing import List
import plotly.graph_objects as go
from sklearn.preprocessing import normalize


def visualize_topics_over_time(topic_model,
                               topics_over_time: pd.DataFrame,
                               top_n_topics: int = None,
                               topics: List[int] = None,
                               normalize_frequency: bool = False,
                               custom_labels: bool = False,
                               title: str = "<b>Topics over Time</b>",
                               width: int = 1250,
                               height: int = 450) -> go.Figure:
    """ Visualize topics over time
    Arguments:
        topic_model: A fitted BERTopic instance.
        topics_over_time: The topics you would like to be visualized with the
                          corresponding topic representation
        top_n_topics: To visualize the most frequent topics instead of all
        topics: Select which topics you would like to be visualized
        normalize_frequency: Whether to normalize each topic's frequency individually
        custom_labels: Whether to use custom topic labels that were defined using
                       `topic_model.set_topic_labels`.
        title: Title of the plot.
        width: The width of the figure.
        height: The height of the figure.
    Returns:
        A plotly.graph_objects.Figure including all traces
    Examples:
    To visualize the topics over time, simply run:
    ```python
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    topic_model.visualize_topics_over_time(topics_over_time)
    ```
    Or if you want to save the resulting figure:
    ```python
    fig = topic_model.visualize_topics_over_time(topics_over_time)
    fig.write_html("path/to/file.html")
    ```
    <iframe src="../../getting_started/visualization/trump.html"
    style="width:1000px; height: 680px; border: 0px;""></iframe>
    """
    colors = ["#17baad", "#44546a", "#50b46e", "#79919c", "#55690f"]

    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        selected_topics = list(topics)
    elif top_n_topics is not None:
        selected_topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        selected_topics = sorted(freq_df.Topic.to_list())

    # Prepare data
    if topic_model.custom_labels_ is not None and custom_labels:
        topic_names = {key: topic_model.custom_labels_[key + topic_model._outliers] for key, _ in topic_model.topic_labels_.items()}
    else:
        topic_names = {key: value[:40] + "..." if len(value) > 40 else value
                       for key, value in topic_model.topic_labels_.items()}
    topics_over_time["Name"] = topics_over_time.Topic.map(topic_names)
    data = topics_over_time.loc[topics_over_time.Topic.isin(selected_topics), :].sort_values(["Topic", "Timestamp"])

    # Add traces
    x_all, y_all, words_all = [], [], []
    fig = go.Figure()
    for index, topic in enumerate(data.Topic.unique()):
        trace_data = data.loc[data.Topic == topic, :]
        topic_name = trace_data.Name.values[0]
        words = trace_data.Words.values
        if normalize_frequency:
            y = normalize(trace_data.Frequency.values.reshape(1, -1))[0]
        else:
            y = trace_data.Frequency
        fig.add_trace(go.Scatter(x=trace_data.Timestamp, y=y,
                                 mode='lines',
                                 marker_color=colors[index % 7],
                                 hoverinfo="text",
                                 name=topic_name,
                                 hovertext=[f'<b>Topic {topic}</b><br>Words: {word}' for word in words]))
        x_all.append(trace_data.Timestamp)
        y_all.append(y)
        words_all.append(words)
    # Styling of the visualization
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_layout(
        yaxis_title="Normalized Frequency" if normalize_frequency else "Frequency",
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.40,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        template="simple_white",
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        legend=dict(
            title="<b>Global Topic Representation",
        )
    )
    return fig, x_all, y_all, words_all
