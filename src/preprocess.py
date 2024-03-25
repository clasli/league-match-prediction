import pandas as pd

match_data_2024_file = '../data/2024/2024_LoL_esports_match_data_from_OraclesElixir.csv'
match_data_2023_file = '../data/2023/2023_Lol_esports_match_data_from_OraclesElixir.csv'

def create_df(filename):
    df = pd.read_csv(filename)
    return df

def create_league_df(df, league_name):
    # only select rows with the league name
    df_league = df[df['league'] == league_name]
    return df_league

def export_df_to_csv(df, filename):
    df.to_csv(filename)

def split_feature_target(df):
    # split the data into features and target
    X = df.drop('result', axis=1)
    y = df['result']
    return X, y

def extract_team_stats(df):
    # only take df rows where postion is 'team'
    df_team = df[df['position'] == 'team']
    return df_team

if __name__ == "__main__":
    df_2023 = create_df(match_data_2023_file)
    
    df_2023_lck_individual = create_league_df(df_2023,'LCK')
    export_df_to_csv(df_2023_lck_individual, '../data/2023/2023_LCK_match_data_individual.csv')

    df_2023_lck_team = extract_team_stats(df_2023_lck_individual)
    export_df_to_csv(df_2023_lck_team, '../data/2023/2023_LCK_match_data_team.csv')

    LCK_2023_X, LCK_2023_y = split_feature_target(df_2023_lck_team)

    ### DEBUG ###
    # print(LCK_2024_X.head())
    # print(LCK_2024_y.head())

    