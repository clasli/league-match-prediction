import pandas as pd
import statistics
import sys

############
### UTIL ###
############
def initialize_team_code_dict():
    team_code_dict = {}
    team_code_dict['BRION'] = 'BRION'
    team_code_dict['Fredit BRION'] = 'BRION'
    team_code_dict['OKSavingsBank BRION'] = 'BRION'
    team_code_dict['Dplus KIA'] = 'DK'
    team_code_dict['DWG KIA'] = 'DK'
    team_code_dict['DRX'] = 'DRX'
    team_code_dict['Gen.G'] = 'GEN'
    team_code_dict['Hanwha Life Esports'] = 'HLE'
    team_code_dict['Kwangdong Freecs'] = 'KDF'
    team_code_dict['KT Rolster'] = 'KT'
    team_code_dict['Liiv SANDBOX'] = 'LSB'
    team_code_dict['FearX'] = 'LSB'
    team_code_dict['Nongshim RedForce'] = 'NS'
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

################
### CLEAN UP ### 
################
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
def create_F1_standardized_win_score(general_input_df, team_code_dict):

    # create the headers for the output dataframe
    unique_team_codes = list(set(get_list_team_code_dict(team_code_dict)))
    combined_headers = ['gameid', 'result', 'blue_wr', 'red_wr']

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    # create dict data structure to store in-progress total games, wins, and losses for each team, using team_code_dict values (each team code should have 3 vals)
    # F1_wr_dict = {team_code: [total_games, total_wins, total_losses]}
    F1_wr_dict = {team_code: [0, 0, 0] for team_code in unique_team_codes}

    # keep track of current split
    curr_split = ""

    # grab all the unique gameids f rom input_df
    gameids = general_input_df['gameid'].unique()

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = general_input_df[general_input_df['gameid'] == gameid] 

        # start of new split, rst wr
        split = game_df['split'].iloc[0]
        if split != curr_split:
            F1_wr_dict = {team_code: [0, 0, 0] for team_code in unique_team_codes}
            curr_split = split

        # # skip this gameid if this is NOT a spring split game
        # if game_df['split'].iloc[0] != 'Spring':
        #     continue

        # # skip this gameid if this is NOT a regular season game
        # if game_df['playoffs'].iloc[0] == 1:
        #     continue

        # in game_df, find teamname and determine team code from team_code_dict
        blue_team = team_code_dict[game_df['teamname'].iloc[0]]
        red_team = team_code_dict[game_df['teamname'].iloc[1]]

        # determine which team won the game 
        result = game_df['result'].iloc[0]

        # store the match data in a list
        match_data = [gameid, result]
            
        # Calculate the team Weighted Win Percentage (WWP)
        # Formula: Weighted Win Percentage (WWP) = TWP (Team Win Percentage) * SOS (Strength of Schedule)
        epsilon = 1e-6
        dec_pts = 4

        # TWP Updates to Total Games, Wins, Losses
        # update total games played
        F1_wr_dict[blue_team][0] += 1
        F1_wr_dict[red_team][0] += 1

        # F1_wr_dict[team_code][0] >= 1 symbolizes that a team has played at least one game... loop + determine how many teams have played at least one game and store into "effective_teams"
        effective_teams = len([team_code for team_code in unique_team_codes if F1_wr_dict[team_code][0] >= 1])
        
        if effective_teams > 10:
            print("Error: More than 10 teams have played at least one game")
            sys.exit()

        # (if) update blue team win, red team loss
        if result == 1: 
            F1_wr_dict[blue_team][1] += 1
            F1_wr_dict[red_team][2] += 1

        # (elif) update red team win, blue team loss
        elif result == 0: 
            F1_wr_dict[red_team][1] += 1
            F1_wr_dict[blue_team][2] += 1

        # TWP Calculation  
        twp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        team_num = 0
        for team_code in unique_team_codes:
            if F1_wr_dict[team_code][0] == 0: # edge case where team has not played any games yet
                twp[team_num] = 0
            else: # if a team has played at minimum 1 game
                twp[team_num] = round(F1_wr_dict[team_code][1] / F1_wr_dict[team_code][0], dec_pts) # calculate (wins/total_games) # ROUND
            team_num += 1

        # print("twp: ", twp) # debug

        # SOS Calculation - Average Opponent TWP (to date)
        sos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        team_num = 0
        for team_code in unique_team_codes:
            sos[team_num] = round((sum(twp) - twp[team_num]) / (effective_teams - 1), dec_pts) # can calculate even if the team has not played yet # ROUND
            team_num += 1

        # print("sos: ", sos) # debug
                        
        # WWP Calculation
        wwp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        team_num = 0
        for team_code in unique_team_codes:
            # if total games for a team is 0, then os_twp = win_percentage * strength_of_schedule / (total_games + epsilon)
            if F1_wr_dict[team_code][0] == 0:
                WWP = twp[team_num] * sos[team_num] / (F1_wr_dict[team_code][0] + epsilon)
                wwp[team_num] = round(WWP, dec_pts)
            else:
                WWP = twp[team_num] * sos[team_num]
                wwp[team_num] = round(WWP, dec_pts)

            team_num += 1

        # print("wwp: ", wwp) # debug
        
        # Standardized WWP Calculation
        standardized_wwp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        WWP_mean = statistics.mean(wwp)
        # print("WWP Mean: " + str(WWP_mean)) # debug
        WWP_std = statistics.stdev(wwp)
        team_num = 0
        for team_code in unique_team_codes:
            if (WWP_std == 0):
                standardized_wwp[team_num] = 0
            else: 
                standardized_wwp[team_num] = round((wwp[team_num] - WWP_mean) / (WWP_std), dec_pts)
            team_num += 1

        # print("standardized_wwp: ", standardized_wwp, "\n") # debug

        # DEBUG: create new dictionary mapping team code to twp, sort and print by descending twp
        # twp_dict = {team_code: twp[i] for i, team_code in enumerate(unique_team_codes)}
        # sorted_twp = sorted(twp_dict.items(), key=lambda x: x[1], reverse=True)
        # print("sorted twp: ", sorted_twp)

        # DEBUG: create new dictionary mapping team code to standardized_wwp, sort and print by descending
        # standardized_wwp_dict = {team_code: standardized_wwp[i] for i, team_code in enumerate(unique_team_codes)}
        # sorted_standardized_wwp = sorted(standardized_wwp_dict.items(), key=lambda x: x[1], reverse=True)
        # print("sorted standardized_wwp: ", sorted_standardized_wwp)

        # add the row_data list to the output dataframe
        blue_idx = unique_team_codes.index(blue_team)
        red_idx = unique_team_codes.index(red_team)
        row_data = [x for x in match_data] + [standardized_wwp[blue_idx], standardized_wwp[red_idx]]
        new_df = pd.DataFrame([row_data], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True)

    return output_df

def create_F2_region_champ_wr(input_df):
    '''
    Feature 2: Champion winrate per region per split.

    Take the average of the winrates of all 5 champions.
    Winrate defaults to 0.5 (for first-time pick)

    Things to consider:
        - fewer picks might not be as accurate
    '''

    # create dict data structure to store in-progress champ wr
    region_champ_wr = {}

    # create a new df
    champ_headers = ["region_top_wr", "region_jg_wr", "region_mid_wr", "region_adc_wr", "region_sup_wr"]
    output_df = pd.DataFrame(columns=champ_headers)

    # grab all the unique gameids from input_df
    gameids = input_df['gameid'].unique()

    # keep track of current split
    curr_split = ""

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = input_df[input_df['gameid'] == gameid] 

        # get current split
        split = game_df['split'].iloc[0]

        # start of new split
        if split != curr_split:
            region_champ_wr = {}
            curr_split = split

        # should be size 10 for 10 players
        if game_df.shape[0] != 10:
            raise Exception('Improperly formed csv. Game {} does not have 10 players.'.format(gameid))

        # get a list of [player, champ, win/loss]
        players = []
        for _, row in game_df.iterrows():
            players.append([row['playerid'], row['champion'], row['result']])

        # calculate the win rate difference per lane (B-R)
        champ_data = []
        for r in range(5):
            b_champ = players[r][1]
            r_champ = players[r+5][1]
            
            
            if b_champ not in region_champ_wr:
                region_champ_wr[b_champ] = {'wins': 0, 'games': 0}
                b_wr = 0.5
            else: # only calculate wr if game is from desired region
                b_wr = region_champ_wr[b_champ]['wins'] / region_champ_wr[b_champ]['games']
            
            if r_champ not in region_champ_wr:
                region_champ_wr[r_champ] = {'wins': 0, 'games': 0}
                r_wr = 0.5
            else: # only calculate wr if game is from desired region
                r_wr = region_champ_wr[r_champ]['wins'] / region_champ_wr[r_champ]['games']
        
            champ_data.append(round(b_wr - r_wr, 3))

            # update total games played and won per champ
            region_champ_wr[b_champ]['games'] += 1
            region_champ_wr[r_champ]['games'] += 1

            if players[r][2] == 1:
                region_champ_wr[b_champ]['wins'] += 1
            else:
                region_champ_wr[r_champ]['wins'] += 1

        # add new row to the output dataframe
        row_data = [y for y in champ_data]
        new_df = pd.DataFrame([row_data], columns=champ_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True)

    return output_df

def create_F3_player_champ_wr(input_df):
    '''
    Feature 3: Champion winrate per player per year.

    Take the average of the winrates of all 5 champions.
    Winrate defaults to 0.5 (for first-time pick)

    Things to consider:
        - maybe this can be per year instead of per split? 
            since player mastery should be retained
    '''
    # create dict data structure to store in-progress champ wr
    player_champ_wr = {}

    # create a new df
    champ_headers = ["player_top_wr", "player_jg_wr", "player_mid_wr", "player_adc_wr", "player_sup_wr"]
    output_df = pd.DataFrame(columns=champ_headers)

    # grab all the unique gameids from input_df
    gameids = input_df['gameid'].unique()

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = input_df[input_df['gameid'] == gameid] 

        # should be size 10 for 10 players
        if game_df.shape[0] != 10:
            raise Exception('Improperly formed csv. Game {} does not have 10 players.'.format(gameid))

        # get a list of [player, champ, win/loss]
        players = []
        for _, row in game_df.iterrows():
            players.append([row['playerid'], row['champion'], row['result']])

        # calculate the win rate difference per lane (B-R)
        champ_data = []
        roles = ['top',]
        for r in range(5):
            b_player = players[r][0]
            b_champ = players[r][1]
            r_player = players[r+5][0]
            r_champ = players[r+5][1]
            
            # edge cases where champ has never been picked
            if b_player not in player_champ_wr: 
                player_champ_wr[b_player] = {}
                player_champ_wr[b_player][b_champ] = {'wins': 0, 'games': 0}
                b_wr = 0.5
            elif b_champ not in player_champ_wr[b_player]:
                player_champ_wr[b_player][b_champ] = {'wins': 0, 'games': 0}
                b_wr = 0.5
            else:
                b_wr = player_champ_wr[b_player][b_champ]['wins'] / player_champ_wr[b_player][b_champ]['games']
            
            if r_player not in player_champ_wr: 
                player_champ_wr[r_player] = {}
                player_champ_wr[r_player][r_champ] = {'wins': 0, 'games': 0}
                r_wr = 0.5
            elif r_champ not in player_champ_wr[r_player]:
                player_champ_wr[r_player][r_champ] = {'wins': 0, 'games': 0}
                r_wr = 0.5
            else:
                r_wr = player_champ_wr[r_player][r_champ]['wins'] / player_champ_wr[r_player][r_champ]['games']
            
            champ_data.append(round(b_wr - r_wr, 3))

            # update total games played and won per champ
            player_champ_wr[b_player][b_champ]['games'] += 1
            player_champ_wr[r_player][r_champ]['games'] += 1

            if players[r][2] == 1:
                player_champ_wr[b_player][b_champ]['wins'] += 1
            else:
                player_champ_wr[r_player][r_champ]['wins'] += 1

        # add new row to the output dataframe
        row_data = [y for y in champ_data]
        new_df = pd.DataFrame([row_data], columns=champ_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True)

    return output_df

def create_F4_patch_champ_wr(input_df, one_region_input_df):
    '''
    Feature 4: Champion winrate per patch, across all regions.

    Things to consider:
        - maybe this can be per year instead of per split? 
            since player mastery should be retained
    '''
    # create dict data structure to store in-progress champ wr
    patch_champ_wr = {}

    # create a new df
    champ_headers = ["patch_top_wr", "patch_jg_wr", "patch_mid_wr", "patch_adc_wr", "patch_sup_wr"]
    output_df = pd.DataFrame(columns=champ_headers)

    # grab all the unique gameids from input_df
    gameids = input_df['gameid'].unique()

    # grab all the unique gameids from the one_region_input_df
    gameid_subset = one_region_input_df['gameid'].unique()

    # track current patch
    curr_patch = ""

    for gameid in gameids: # iterate through all unique game_ids

        # get all rows with the same gameid
        game_df = input_df[input_df['gameid'] == gameid] 

        # get current patch
        patch = game_df['patch'].iloc[0]

        # start of new split
        if patch != curr_patch:
            patch_champ_wr = {}
            curr_patch = patch

        # should be size 10 for 10 players
        if game_df.shape[0] != 10:
            raise Exception('Improperly formed csv. Game {} does not have 10 players.'.format(gameid))

        # get a list of [player, champ, win/loss]
        players = []
        for _, row in game_df.iterrows():
            players.append([row['playerid'], row['champion'], row['result']])

        # calculate the win rate difference per lane (B-R)
        champ_data = []
        for r in range(5):
            b_champ = players[r][1]
            r_champ = players[r+5][1]
            
            
            if b_champ not in patch_champ_wr:
                patch_champ_wr[b_champ] = {'wins': 0, 'games': 0}
                b_wr = 0.5
            elif gameid in gameid_subset: # only calculate wr if game is from desired region
                b_wr = patch_champ_wr[b_champ]['wins'] / patch_champ_wr[b_champ]['games']
            
            if r_champ not in patch_champ_wr:
                patch_champ_wr[r_champ] = {'wins': 0, 'games': 0}
                r_wr = 0.5
            elif gameid in gameid_subset: # only calculate wr if game is from desired region
                r_wr = patch_champ_wr[r_champ]['wins'] / patch_champ_wr[r_champ]['games']
        
            if gameid in gameid_subset: # only calculate wr if game is from desired region
                champ_data.append(round(b_wr - r_wr, 3))

            # update total games played and won per champ
            patch_champ_wr[b_champ]['games'] += 1
            patch_champ_wr[r_champ]['games'] += 1

            if players[r][2] == 1:
                patch_champ_wr[b_champ]['wins'] += 1
            else:
                patch_champ_wr[r_champ]['wins'] += 1

        # add new row to the output dataframe (only if game is from desired region)
        if gameid in gameid_subset:
            row_data = [y for y in champ_data]
            new_df = pd.DataFrame([row_data], columns=champ_headers)
            output_df = pd.concat([output_df, new_df], ignore_index=True)

    return output_df

if __name__ == '__main__':
    # init all the team codes for each team name
    team_code_dict = initialize_team_code_dict()

    # pull data from input csvs
    filename = '../data/2023/2023_LCK_match_data_team.csv'
    general_input_df = create_df(filename) 

    filename = '../data/2023/2023_LCK_match_data_individual.csv'
    individual_input_df = create_df(filename) 

    filename = '../data/2023/2023_ALL_match_data_individual.csv'
    all_regions_input_df = create_df(filename) 
    
    # specify output df and csv
    output_filename = "../data/2023/2023_LCK_LogReg_Dataset.csv"

    # add feature data to output df
    F1_output_df = create_F1_standardized_win_score(general_input_df, team_code_dict)

    # F2 result (per region champion wr)
    F2_output_df = create_F2_region_champ_wr(individual_input_df)

    # F3 result (per player champion wr)
    F3_output_df = create_F3_player_champ_wr(individual_input_df)

    # F4 result (per patch champion wr)
    F4_output_df = create_F4_patch_champ_wr(all_regions_input_df, general_input_df)

    # Adding features to the desired final df 
    output_df = pd.DataFrame()
    output_df = pd.concat([output_df, F1_output_df], axis=1)
    output_df = pd.concat([output_df, F2_output_df], axis=1)
    output_df = pd.concat([output_df, F3_output_df], axis=1)
    output_df = pd.concat([output_df, F4_output_df], axis=1)

    # export output df as csv
    export_df_to_csv(output_df, output_filename)

    # note that "BRION" and "OKSavingsBank BRION" are the same team, should replace to "BRION" for consistency
    # team_list = [team.replace('OKSavingsBank BRION', 'BRION') for team in team_list]



## INIT ##
# (1) Utility Class for Re-Useable Functions
# (2) Feature Class (init method, create method, update method)
# (3) Refine Data Function 