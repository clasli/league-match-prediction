import pandas as pd

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

def extract_individual_stats(df):
    # only take df rows where postion is NOT 'team'
    df_team = df[df['position'] != 'team']
    return df_team

if __name__ == "__main__":
    match_data_2021_file = '../data/2021/2021_LoL_esports_match_data_from_OraclesElixir.csv'
    match_data_2022_file = '../data/2022/2022_LoL_esports_match_data_from_OraclesElixir.csv'
    match_data_2023_file = '../data/2023/2023_Lol_esports_match_data_from_OraclesElixir.csv'
    match_data_2024_file = '../data/2024/2024_LoL_esports_match_data_from_OraclesElixir.csv'


    df_2023 = create_df(match_data_2023_file)
    # df_2021 = create_df(match_data_2021_file)
    df_2022 = create_df(match_data_2022_file)
    df_2024 = create_df(match_data_2024_file)
    temp_df = pd.DataFrame()

    # temp_df = pd.concat([temp_df, df_2021], ignore_index=True)
    temp_df = pd.concat([temp_df, df_2022], ignore_index=True)
    temp_df = pd.concat([temp_df, df_2023], ignore_index=True)
    df_2023 = pd.concat([temp_df, df_2024], ignore_index=True)

    df_2023_lck = create_league_df(df_2023,'LCK')

    df_2023_lck_team = extract_team_stats(df_2023_lck)
    export_df_to_csv(df_2023_lck_team, '../data/output/LCK_match_data_team.csv')

    df_2023_lck_individual = extract_individual_stats(df_2023_lck)
    export_df_to_csv(df_2023_lck_individual, '../data/output/LCK_match_data_individual.csv')

    df_2023_all_team = extract_individual_stats(df_2023)
    export_df_to_csv(df_2023_all_team, '../data/output/ALL_match_data_individual.csv')

    LCK_2023_X, LCK_2023_y = split_feature_target(df_2023_lck_team)

    ### DEBUG ###
    # print(LCK_2024_X.head())
    # print(LCK_2024_y.head())

    