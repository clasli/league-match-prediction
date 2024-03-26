import pandas as pd

filename = '../data/2023/2023_LCK_match_data_team.csv'

def initialize_team_code_dict():
    team_code_dict = {}
    team_code_dict['BRION'] = 'BRION'
    team_code_dict['Dplus KIA'] = 'DK'
    team_code_dict['DRX'] = 'DRX'
    team_code_dict['Gen.G'] = 'GEN'
    team_code_dict['Hanwha Life Esports'] = 'HLE'
    team_code_dict['Kwangdong Freecs'] = 'KDF'
    team_code_dict['KT Rolster'] = 'KT'
    team_code_dict['Liiv SANDBOX'] = 'LSB'
    team_code_dict['Nongshim RedForce'] = 'NS'
    team_code_dict['OKSavingsBank BRION'] = 'BRION'
    team_code_dict['T1'] = 'T1'

    return team_code_dict

def get_list_team_code_dict(team_code_dict):
    return list(team_code_dict.values())

def create_df(filename):
    df = pd.read_csv(filename)
    return df

def get_unique_teams(df):
    teams = df['teamname'].unique()
    return teams

def refine_data(df):
    # ensure that all data is complete (if we will use that row for match prediction)
    df = df[df['datacompleteness'] == 'complete']

    # remove "datacompleteness" column
    df = df.drop(columns=['datacompleteness'])

    # remove "url" column, "league" column, "date" column, "ban1", "ban2", "ban3", "ban4", "ban5" columns, 
    df = df.drop(columns=['url'])

    # create new dataframes for each split
    df_spring = df[df['split'] == 'spring']
    df_summer = df[df['split'] == 'summer']

    # create new dataframes for non-playoffs and playoffs
    df_spring_reg_season = df_spring[df_spring['playoffs'] == False]
    df_spring_playoffs = df_spring[df_spring['playoffs'] == True]
    df_summer_reg_season = df_summer[df_summer['playoffs'] == False]
    df_summer_playoffs = df_summer[df_summer['playoffs'] == True]

    # (1) consider 2 entries (each team) per game ID
    # (2) 
    
def create_output_csv(teamlist):
    output_df = pd.DataFrame(teamlist)
    output_df.to_csv('../data/2023/2023_LCK_teams.csv', index=False)

def export_df_to_csv(df, filename):
    df.to_csv(filename)

########################
### FEATURE CREATION ###
########################
    
""" 
* FEATURE 1: Standardized Win Score *
> Relevant Headers: 
    (1) gameid
    (2) game (1 or 2) id, series id??
    (3) blue or red side
    (4) teamname, team id
    (5) result
> Intuition:    Normally, we calculate a cumulative win pc for each team after each game
                and assume that for a given upcoming game, the team with higher win pc will win. 
                However, this is a naive approach that does not consider the strength of the
                opponent. Suppose that the (true) 7th place team has 3 wins (to date) as they beat the 8th,
                9th, and 10th place teams. Suposse the (true) 2nd place team has 2 wins and 1 loss 
                as they beat the 3rd and 4th place teams but lost to the 1st place team. The naive 
                approach would suggest that the 7th place team is stronger because they have more wins.
                However, the 2nd place team has a higher win pc against stronger opponents. So to measure 
                the strength of a team, we need to consider the strength of the opponent.
> Method:       We will calculate a standardized win score for each team after each game. The standardized
                win score will be calculated as follows:
                (1) Calculate the win pc of the team for all games played to date
                (2) Calculate the average win pc of the opponents for all games played to date
                (3) Calculate the standardized win score as the difference between the team's win pc and the 
                    average win pc of the opponents
                (4) The standardized win score will be calculated for each game played to date
                (5) We will use the standardized win score as a feature in the logistic regression model

> Formula:     standardized_win_score = (team_win_pc - avg_opponent_win_pc) / standardized_av_opponent_win_pc
> Why this works: 
                This formula works because it measures the strength of the team relative to the strength of the
                opponents. If the team has a higher win pc than the average win pc of the opponents, then the
                team is stronger than the opponents. If the team has a lower win pc than the average win pc of the
                opponents, then the team is weaker than the opponents. If the team has a win pc equal to the average

> What advantage does dividing by standard deviation give us? 
                Dividing by the standard deviation gives us a standardized score that is independent of the scale
                of the win pc. This is important because the win pc can range from 0 to 1, and the scale of the win pc
                can vary depending on the number of games played. By dividing by the standard deviation, we can compare
                the standardized win score across different teams and different games.

"""
# implements the above explanation of feature
def create_F1_standardized_win_score(input_df, team_code_dict, F1_output_filename, export_csv):

    # create the headers for the output dataframe
    unique_team_codes = list(set(get_list_team_code_dict(team_code_dict)))
    combined_headers = ['match_id', 'gameid', 'game', 'blue_team', 'red_team', 'result'] + unique_team_codes

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    # create dict data structure to store in-progress total games, wins, and losses for each team, using team_code_dict values (each team code should have 3 vals)
    F1_wr_dict = {team_code: [0, 0, 0] for team_code in unique_team_codes}

    # grab all the unique gameids from input_df
    gameids = input_df['gameid'].unique()

    # setup relative match_id
    match_id = 1

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = input_df[input_df['gameid'] == gameid] 

        # determine which game it is (1 or 2)
        game = "game_" + str(game_df['game'].iloc[0])

        # in game_df, find teamname and determine team code from team_code_dict
        blue_team = team_code_dict[game_df['teamname'].iloc[0]]
        red_team = team_code_dict[game_df['teamname'].iloc[1]]

        # determine which team won the game (check boolean result column, and output the team code (from team_code_dict) using winning team name)
        if bool(game_df['result'].iloc[0]) == 1:
            result = team_code_dict[game_df['teamname'].iloc[0]]
            # print(result + " won" + " gameid: " + str(gameid))
        elif bool(game_df['result'].iloc[1]) == 1:
            result = team_code_dict[game_df['teamname'].iloc[1]]
            # print(result + " won" + " gameid: " + str(gameid))

        # update the in-progress total games, wins, and losses for each team
            
        # update total games played
        F1_wr_dict[blue_team][0] += 1
        F1_wr_dict[red_team][0] += 1

        if result == blue_team: # update blue team win, red team loss
            F1_wr_dict[blue_team][1] += 1
            F1_wr_dict[red_team][2] += 1

        elif result == red_team: # update red team win, blue team loss
            F1_wr_dict[red_team][1] += 1
            F1_wr_dict[blue_team][2] += 1

        # store the match data in a list
        match_data = [match_id, gameid, game, blue_team, red_team, result]
            
        # calculate the standardized win score per team (and store in a list)
        team_cwr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        team_num = 0
        for team_code in unique_team_codes:
            if F1_wr_dict[team_code][0] == 0: # edge case where team has not played any games yet
                team_cwr[team_num] = 0
            else:
                team_cwr[team_num] = round(F1_wr_dict[team_code][1] / F1_wr_dict[team_code][0], 2)
            team_num += 1

        # add the row_data list to the output dataframe
        row_data = [x for x in match_data] + [y for y in team_cwr]
        new_df = pd.DataFrame([row_data], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True)

        # increase match_id
        match_id += 1

    if export_csv:
        export_df_to_csv(output_df, F1_output_filename)

    return output_df

def create_F2_region_champ_wr(input_df, wip_df, team_code_dict):
    '''
    Feature 2: Champion winrate per region per split.

    Take the average of the winrates of all 5 champions.
    Winrate defaults to 0.5 (for first-time pick)

    Things to consider:
        - fewer picks might not be as accurate
    '''
    # create the headers for the output dataframe
    # unique_team_codes = list(set(get_list_team_code_dict(team_code_dict)))
    # combined_headers = ['match_id', 'game', 'blue_team', 'red_team', 'result']

    # create dict data structure to store in-progress champ wr
    region_champ_wr = {}

    # add new headers to wip_df
    champ_headers = ["region_top_wr", "region_jg_wr", "region_mid_wr", "region_adc_wr", "region_sup_wr"]
    for header in champ_headers:
        wip_df[header] = None

    # grab all the unique gameids from input_df
    gameids = input_df['gameid'].unique()

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = input_df[input_df['gameid'] == gameid] 

        # get index of wip_df
        index = wip_df[wip_df['gameid'] == gameid].index.tolist()

        # calculate the win rate difference per lane (B-R)
        champ_data = []
        picks = ['pick1', 'pick2', 'pick3', 'pick4', 'pick5']
        for p in picks:
            b_champ = game_df[p].iloc[0]
            r_champ = game_df[p].iloc[1]
            
            # edge cases where champ has never been picked
            if b_champ not in region_champ_wr: 
                region_champ_wr[b_champ] = {'wins': 0, 'games': 0}
                b_wr = 0.5
            else:
                b_wr = region_champ_wr[b_champ]['wins'] / region_champ_wr[b_champ]['games']
            if r_champ not in region_champ_wr:
                region_champ_wr[r_champ] = {'wins': 0, 'games': 0} 
                r_wr = 0.5
            else:
                r_wr = region_champ_wr[r_champ]['wins'] / region_champ_wr[r_champ]['games']

            champ_data.append(round(b_wr - r_wr, 3))

            # update total games played and won per champ
            region_champ_wr[b_champ]['games'] += 1
            region_champ_wr[r_champ]['games'] += 1

            if game_df['result'].iloc[0] == 1:
                region_champ_wr[b_champ]['wins'] += 1
            else:
                region_champ_wr[r_champ]['games'] += 1

        # add columns to the output dataframe
        for header, val in zip(champ_headers, champ_data):
            wip_df.iloc[index, wip_df.columns.get_loc(header)] = val

def create_F3_player_champ_wr(input_df, team_code_dict):
    '''
    Feature 3: Champion winrate per player per split.

    Take the average of the winrates of all 5 champions.
    Winrate defaults to 0.5 (for first-time pick)

    Things to consider:
        - maybe this can be per year instead of per split? 
            since player mastery should be retained
    '''
    pass

def create_F4_patch_champ_wr(input_df, team_code_dict):
    # dates up to curr patch, cross region
    pass

if __name__ == '__main__':
    team_code_dict = initialize_team_code_dict()

    game_data_df = create_df(filename)
    # print(game_data_df.head())
    
    F1_output_filename = "../data/2023/2023_LCK_LogReg_F1_standardized_win_score.csv"
    output_df = create_F1_standardized_win_score(game_data_df, team_code_dict, F1_output_filename, True)
    create_F2_region_champ_wr(game_data_df, output_df, team_code_dict)

    export_df_to_csv(output_df, F1_output_filename)






    # note that "BRION" and "OKSavingsBank BRION" are the same team, should replace to "BRION" for consistency
    # team_list = [team.replace('OKSavingsBank BRION', 'BRION') for team in team_list]

