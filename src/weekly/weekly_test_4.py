import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import random

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

euro12 = pd.read_csv("C:/Users/Atti60/Documents/GitHub/ECOPY_23241/data/Euro_2012_stats_TEAM.csv")

def number_of_participants(input_df: pd.DataFrame) -> int:
    return len(input_df)

def goals(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df[['Team', 'Goals']]

def sorted_by_goal(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.sort_values(by='Goals', ascending=False)

def avg_goal(input_df: pd.DataFrame) -> float:
    return input_df['Goals'].mean()

def countries_over_five(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df[input_df['Goals'] >= 6]

def countries_starting_with_g(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df[input_df['Team'].str.startswith('G')]

def first_seven_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.iloc[:, :7]

def every_column_except_last_three(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.iloc[:, :-3]

def sliced_view(input_df: pd.DataFrame, columns_to_keep: List[str], column_to_filter: str, rows_to_keep: List[str]) -> pd.DataFrame:
    return input_df[columns_to_keep + [column_to_filter]][input_df[column_to_filter].isin(rows_to_keep)]

def generate_quartile(input_df: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        (input_df['Goals'] >= 6) & (input_df['Goals'] <= 12),
        (input_df['Goals'] == 5),
        (input_df['Goals'] >= 3) & (input_df['Goals'] <= 4),
        (input_df['Goals'] >= 0) & (input_df['Goals'] <= 2)
    ]
    values = [1, 2, 3, 4]
    quartile_values = [values[i] for i in range(len(input_df)) if conditions[i]]
    input_df['Quartile'] = quartile_values
    return input_df

def average_yellow_in_quartiles(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby('Quartile')['Yellow Cards'].mean().reset_index()

def minmax_block_in_quartile(input_df: pd.DataFrame) -> pd.DataFrame:
    return input_df.groupby('Quartile')['Blocks'].agg(['min', 'max']).reset_index()

def scatter_goals_shots(input_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(input_df['Goals'], input_df['Shots on target'])
    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')
    ax.set_title('Goals and Shot on target')
    return fig

def scatter_goals_shots_by_quartile(input_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for quartile in input_df['Quartile'].unique():
        quartile_data = input_df[input_df['Quartile'] == quartile]
        ax.scatter(quartile_data['Goals'], quartile_data['Shots on target'], label=f'Quartile {quartile}')

    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')
    ax.set_title('Goals and Shot on target')
    ax.legend(title='Quartiles')
    return fig

def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []
    for _ in range(number_of_trajectories):
        random_numbers = [pareto_distribution(1, 1) for _ in range(length_of_trajectory)]
        cumulative_average = [sum(random_numbers[:i + 1]) / (i + 1) for i in range(length_of_trajectory)]
        trajectories.append(cumulative_average)
    return trajectories



