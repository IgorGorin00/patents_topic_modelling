import pandas as pd

def prepare_df(df):
    df['date'] = pd.to_datetime(df['date'])
    df['title'] = df['title'].str.lower()
    return df


def get_df_from_txt(path) -> pd.DataFrame:
    """
    Was originally used to read data from txt file
    and convert it to pd.DataFrame,
    not used currently, but i keep it,
    because why not
    """
    import re
    dates_all, titles_all = [], []
    for filename in os.listdir(path):
        if 'fips' in filename:
            with open(f'/{path}/{filename}', 'r') as f:
                text = f.read()
                dates = (re.findall(r'\(\d+\.\d+\.\d+\)', text))
                titles = (re.findall(r'\)\n.*\nЗИЗ', text))
                assert len(dates) == len(titles)
                dates_all.extend(dates)
                titles_all.extend(titles)
    df = pd.DataFrame({'date': dates_all, 'title': titles_all})

    dates_new = []
    for date in df['date']:
        dates_new.append(date[1:-1])
    df['date'] = dates_new
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

    new_titles = []
    for title in df['title']:
        new_titles.append(title[2:-4])
    df['title'] = new_titles
    df['title'] = df['title'].str.lower()
    return df


def get_words_scores_df(words_all: list, scores_all: list):
    words_to_df = []
    scores_to_df = []

    for words, scores in zip(words_all, scores_all):
        words.append('##############')
        words_to_df.extend(words)
        scores.append('##############')
        scores_to_df.extend(scores)

    return pd.DataFrame({'words': words_to_df, 'scores': scores_to_df})


def get_over_time_values(x, y, words):
    dfs_over_time = []
    for i in range(len(x)):
        over_time = pd.concat([x[i], y[i]], axis=1).reset_index(drop=True)
        over_time[f'words_{i}'] = pd.Series(words[i])
        dfs_over_time.append(over_time)
    return pd.concat(dfs_over_time, axis=1)
