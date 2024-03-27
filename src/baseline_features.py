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

def calculate_winrate_separate(df):
    win_rates = {}
    curr_split = ""

    for index, row in df.iterrows():
        team = row['teamname']
        result = row['result']
        split = row['split']

        # Start of new split
        if split != curr_split:
            win_rates = {}
            curr_split = split
        
        # First game of the team for the split
        if team not in win_rates:
            win_rates[team] = {'wins': 0, 'games': 0}
            win_rate = -1
        # Calculate win rate up to this point
        else:
            win_rate = win_rates[team]['wins'] / win_rates[team]['games']

        # Update win and game counts
        if result == 1:
            win_rates[team]['wins'] += 1
        win_rates[team]['games'] += 1
        
        # Add win rate to the DataFrame
        df.at[index, 'win_rate'] = win_rate
    
    return df[['gameid', 'league', 'year', 'split', 'date', 'side', 'teamname', 'win_rate', 'result']]

def calculate_wr(input_df):
    combined_headers = ['gameid', 'split', 'blue_team', 'red_team', 'blue_wr', 'red_wr', 'result']
    output_df = pd.DataFrame(columns=combined_headers)

    win_rates = {}
    curr_split = ""

    for gameid in input_df['gameid'].unique():
        game_df = input_df[input_df['gameid'] == gameid]

        # get current split
        split = game_df['split'].iloc[0]

        # Start of new split
        if split != curr_split:
            # win_rates = {}
            curr_split = split

        # in game_df, find teamname
        blue_team = game_df['teamname'].iloc[0]
        red_team = game_df['teamname'].iloc[1]

        # determine which team won the game (check boolean result column, and output the team code (from team_code_dict) using winning team name)
        result = game_df['result'].iloc[0]

        # update blue wr
        if blue_team not in win_rates:
            win_rates[blue_team] = {'wins': 0, 'games': 0}
            blue_wr = 0.5
        else:
            blue_wr = win_rates[blue_team]['wins'] / win_rates[blue_team]['games']

        # update red wr
        if red_team not in win_rates:
            win_rates[red_team] = {'wins': 0, 'games': 0}
            red_wr = 0.5
        else:
            red_wr = win_rates[red_team]['wins'] / win_rates[red_team]['games']

        # write the data to the output dataframe (columns: gameid, blue_team, red_team, result + (10 columsn to be populated with 0s)) ... use concate instead of append
        new_df = pd.DataFrame([[gameid, split, blue_team, red_team, blue_wr, red_wr, result]], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True)

        # Update win and game counts
        if result == 1:
            win_rates[blue_team]['wins'] += 1
        else:
            win_rates[red_team]['wins'] += 1
        win_rates[blue_team]['games'] += 1
        win_rates[red_team]['games'] += 1

    return output_df

if __name__ == "__main__":
    match_data_2022_file = '../data/2022/2022_LoL_esports_match_data_from_OraclesElixir.csv'
    match_data_2023_file = '../data/2023/2023_Lol_esports_match_data_from_OraclesElixir.csv'
    match_data_2024_file = '../data/2024/2024_LoL_esports_match_data_from_OraclesElixir.csv'


    df_2022 = create_df(match_data_2022_file)
    df_2023 = create_df(match_data_2023_file)
    df_2024 = create_df(match_data_2024_file)
    
    df_2022_lck_individual = create_league_df(df_2022,'LCK')
    df_2023_lck_individual = create_league_df(df_2023,'LCK')
    df_2024_lck_individual = create_league_df(df_2024,'LCK')

    df_2022_lck_team = extract_team_stats(df_2022_lck_individual)
    df_2023_lck_team = extract_team_stats(df_2023_lck_individual)
    df_2024_lck_team = extract_team_stats(df_2024_lck_individual)

    df_2022_lck_with_winrate_separate = calculate_wr(df_2022_lck_team)
    df_2023_lck_with_winrate_separate = calculate_wr(df_2023_lck_team)
    df_2024_lck_with_winrate_separate = calculate_wr(df_2024_lck_team)

    # add df-2023 and df-2024 rows on top of each other
    df_2022_2023_2024 = df_2022_lck_with_winrate_separate
    df_2022_2023_2024 = pd.concat([df_2022_2023_2024, df_2023_lck_with_winrate_separate], ignore_index=True)
    df_2022_2023_2024 = pd.concat([df_2022_2023_2024, df_2024_lck_with_winrate_separate], ignore_index=True)
    export_df_to_csv(df_2022_2023_2024, '../data/2023/2023_LCK_match_data_with_winrate.csv')

    LCK_2023_X, LCK_2023_y = split_feature_target(df_2023_lck_team)

    


    