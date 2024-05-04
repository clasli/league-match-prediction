import pandas as pd
import statistics
import sys

def create_df(filename):
    df = pd.read_csv(filename)
    return df

def create_output_csv(teamlist):
    output_df = pd.DataFrame(teamlist)
    output_df.to_csv('../data/2023/2023_LCK_teams.csv', index=False)

def export_df_to_csv(df, filename):
    df.to_csv(filename)

########################
### FEATURE CREATION ###
########################
   
def create_F0_results(general_input_df):
    combined_headers = ['gameid', 'result']

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    gameids = general_input_df['gameid'].unique()

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = general_input_df[general_input_df['gameid'] == gameid] 

        # determine which team won the game 
        result = game_df['result'].iloc[0]

        # store the match data in a list
        match_data = [gameid, result]

        new_df = pd.DataFrame([match_data], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True) 

    return output_df

def create_F0_5_win_score(general_input_df):
    
    # create the headers for the output dataframe
    combined_headers = ['blue_wr_simple', 'blue_wr_rel_simple']

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    # create dict data structure to store in-progress total games, wins, and losses for each team, using team_code_dict values (each team code should have 3 vals)
    # F1_wr_dict = {team_code: [total_games, total_wins, total_losses]}
    F1_wr_dict = {}

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
            F1_wr_dict = {}
            curr_split = split

        # in game_df, find teamname and determine team code from team_code_dict
        blue_team = game_df['teamname'].iloc[0]
        red_team = game_df['teamname'].iloc[1]

        # determine which team won the game 
        result = game_df['result'].iloc[0]

        # TWP Updates to Total Games, Wins, Losses
        # update total games played
        if blue_team not in F1_wr_dict:
            F1_wr_dict[blue_team] = [0, 0, 0]
        if red_team not in F1_wr_dict:
            F1_wr_dict[red_team] = [0, 0, 0]

        F1_wr_dict[blue_team][0] += 1
        F1_wr_dict[red_team][0] += 1

        # F1_wr_dict[team_code][0] >= 1 symbolizes that a team has played at least one game... loop + determine how many teams have played at least one game and store into "effective_teams"
        effective_teams = len(F1_wr_dict)
        
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
        twp = {}
        for team_code in F1_wr_dict.keys():
            twp[team_code] = round(F1_wr_dict[team_code][1] / F1_wr_dict[team_code][0], 4)

        # add the row_data list to the output dataframe
        row_data = [(twp[blue_team]), round(twp[blue_team] - twp[red_team], 4)]
        new_df = pd.DataFrame([row_data], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True) 

    
    print(output_df['blue_wr_simple'].min())
    print(output_df['blue_wr_simple'].max())
    print(output_df['blue_wr_rel_simple'].min())
    print(output_df['blue_wr_rel_simple'].max())

    return output_df

# implements the above explanation of feature
def create_F1_standardized_win_score(general_input_df):

    # create the headers for the output dataframe
    combined_headers = ['blue_wr', 'blue_wr_rel']

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    # create dict data structure to store in-progress total games, wins, and losses for each team, using team_code_dict values (each team code should have 3 vals)
    # F1_wr_dict = {team_code: [total_games, total_wins, total_losses]}
    F1_wr_dict = {}

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
            F1_wr_dict = {}
            curr_split = split

        # in game_df, find teamname and determine team code from team_code_dict
        blue_team = game_df['teamname'].iloc[0]
        red_team = game_df['teamname'].iloc[1]

        # determine which team won the game 
        result = game_df['result'].iloc[0]

        # Calculate the team Weighted Win Percentage (WWP)
        # Formula: Weighted Win Percentage (WWP) = TWP (Team Win Percentage) * SOS (Strength of Schedule)
        epsilon = 1e-6
        dec_pts = 4

        # TWP Updates to Total Games, Wins, Losses
        # update total games played
        if blue_team not in F1_wr_dict:
            F1_wr_dict[blue_team] = [0, 0, 0]
        if red_team not in F1_wr_dict:
            F1_wr_dict[red_team] = [0, 0, 0]

        F1_wr_dict[blue_team][0] += 1
        F1_wr_dict[red_team][0] += 1

        # F1_wr_dict[team_code][0] >= 1 symbolizes that a team has played at least one game... loop + determine how many teams have played at least one game and store into "effective_teams"
        effective_teams = len(F1_wr_dict)
        
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
        # twp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        twp = {}
        
        # team_num = 0
        for team_code in F1_wr_dict.keys():
            # twp[team_num] = round(F1_wr_dict[team_code][1] / F1_wr_dict[team_code][0], dec_pts) # calculate (wins/total_games) # ROUND
            twp[team_code] = round(F1_wr_dict[team_code][1] / F1_wr_dict[team_code][0], dec_pts)
            # team_num += 1

        # print("twp: ", twp) # debug

        # SOS Calculation - Average Opponent TWP (to date)
        # sos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        sos = {}
        # team_num = 0
        for team_code in F1_wr_dict.keys():
            # sos[team_num] = round((sum(twp) - twp[team_num]) / (len(F1_wr_dict)-1), dec_pts) # can calculate even if the team has not played yet # ROUND
            sos[team_code] = round((sum(twp.values()) - twp[team_code]) / (len(F1_wr_dict)-1), dec_pts)
            # team_num += 1

        # print("sos: ", sos) # debug
                        
        # WWP Calculation
        # wwp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        wwp = {}
        # team_num = 0
        for team_code in F1_wr_dict.keys():
            # if total games for a team is 0, then os_twp = win_percentage * strength_of_schedule / (total_games + epsilon)
            # WWP = twp[team_num] * sos[team_num]
            # wwp[team_num] = round(WWP, dec_pts)
            WWP = twp[team_code] * sos[team_code]
            wwp[team_code] = round(WWP, dec_pts)

        # Standardized WWP Calculation
        # standardized_wwp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        standardized_wwp = {}
        WWP_mean = statistics.mean(wwp.values())
        # print("WWP Mean: " + str(WWP_mean)) # debug
        WWP_std = statistics.stdev(wwp.values())
        # team_num = 0
        for team_code in F1_wr_dict.keys():
            if (WWP_std == 0):
                # standardized_wwp[team_num] = 0
                standardized_wwp[team_code] = 0
            else: 
                # standardized_wwp[team_num] = round((wwp[team_num] - WWP_mean) / (WWP_std), dec_pts)
                standardized_wwp[team_code] = round((wwp[team_code] - WWP_mean) / (WWP_std), dec_pts)
            # team_num += 1

        # add the row_data list to the output dataframe
        row_data = [(standardized_wwp[blue_team])/1.8, (standardized_wwp[blue_team] - standardized_wwp[red_team])/3]
        new_df = pd.DataFrame([row_data], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True) 

    
    print(output_df['blue_wr'].min())
    print(output_df['blue_wr'].max())
    print(output_df['blue_wr_rel'].min())
    print(output_df['blue_wr_rel'].max())

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
    champ_headers2 = ["region_top_wr", "region_top_wr_rel", "region_jg_wr", "region_jg_wr_rel", "region_mid_wr", \
                     "region_mid_wr_rel", "region_adc_wr", "region_adc_wr_rel", "region_sup_wr", "region_sup_wr_rel"]
    output_df = pd.DataFrame(columns=champ_headers2)

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
        
            champ_data.append(round(b_wr, 3))
            champ_data.append(round((b_wr - r_wr), 3))

            # update total games played and won per champ
            region_champ_wr[b_champ]['games'] += 1
            region_champ_wr[r_champ]['games'] += 1

            if players[r][2] == 1:
                region_champ_wr[b_champ]['wins'] += 1
            else:
                region_champ_wr[r_champ]['wins'] += 1

        # add new row to the output dataframe
        row_data = [y for y in champ_data]
        new_df = pd.DataFrame([row_data], columns=champ_headers2)
        output_df = pd.concat([output_df, new_df], ignore_index=True)

    for col in champ_headers:
        mean = output_df[col].mean()
        std = output_df[col].std()
        output_df[col] = (output_df[col]-mean)/std/3.2

        print(output_df[col].min())
        print(output_df[col].max())

    

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
    champ_headers2 = ["player_top_wr", "player_top_wr_rel", "player_jg_wr", "player_jg_wr_rel", "player_mid_wr", \
                     "player_mid_wr_rel", "player_adc_wr", "player_adc_wr_rel", "player_sup_wr", "player_sup_wr_rel"]
    
    champ_headers = ["player_top_wr", "player_jg_wr", "player_mid_wr", "player_adc_wr", "player_sup_wr"]

    output_df = pd.DataFrame(columns=champ_headers2) # TODO: fix champ headers

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
            
            champ_data.append(round(b_wr, 3)) 
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
        new_df = pd.DataFrame([row_data], columns=champ_headers2) # TODO: fix champ headers
        output_df = pd.concat([output_df, new_df], ignore_index=True)

    for col in champ_headers:
        mean = output_df[col].mean()
        std = output_df[col].std()
        output_df[col] = (output_df[col]-mean)/std/3.2

        print(output_df[col].min())
        print(output_df[col].max())

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

# def create_F5_team_momentum(input_df):
#     '''
#     Feature 5: Team Momentum (Indicator Features)

#     Momentum is related to "if a team won a previous game, how does that affect current game"
#     Momentum 0 indicates that the team has no past results in a series
#     Momentum 1 indicates that the team WON/LOSS their previous game
#     Momentum 2 indicates that the team WON/LOSS their (2) previous games
#     Positive Momentum indicates that Blue Team has momentum
#     Negative Momentum indicates that Red Team has momentum

#     Things to consider:
#         - does a team perform better with no past results in a series? 
#         - does a team perform better if they win a game? 
#         - does a team perform worse if they lose 2 games in a row? are they not likely to win? 
#     '''

#     # create a new df
#     momentum_headers = ["momentum"]
#     output_df = pd.DataFrame(columns=momentum_headers)    

#     # create dict data structure to store in-progress total games, wins, and losses for each team, using team_code_dict values (each team code should have 3 vals)
#     # F1_wr_dict = {team_code: [total_games, total_wins, total_losses]}
#     unique_team_codes = list(set(get_list_team_code_dict(team_code_dict)))
#     F5_wr_dict = {team_code: [0, 0, 0] for team_code in unique_team_codes}

#     gameids = input_df['gameid'].unique()

#     # keep track of current split
#     curr_split = ""

#     # keep track of a list of game_df
#     game_df_list = []

#     for gameid in gameids: # iterate through all unique game_ids
        
#         # get all rows with the same gameid
#         game_df = input_df[input_df['gameid'] == gameid] 

#         # get current split
#         split = game_df['split'].iloc[0]

#         # start of new split
#         if split != curr_split:
#             curr_split = split

#         # get the "game" of the game_df
#         game = game_df['game'].iloc[0]

#         # in game_df, find teamname and determine team code from team_code_dict
#         blue_team = team_code_dict[game_df['teamname'].iloc[0]]
#         red_team = team_code_dict[game_df['teamname'].iloc[1]]        

#         # if game is the first game of the series, then momentum is 0
#         F5_wr_dict[blue_team][0] = 0  # start by momentum being 0 because will modify it later
#         F5_wr_dict[red_team][0] = 0 

#         if game >= 2:
#             # get the result of the previous game
#             # identify the previous iterated game_df from game_df_list
#             prev_game_df = game_df_list[-1]

#             # make sure the "game" field of prev_game_df is 1 less than the current "game" value from game_df
#             if (prev_game_df['game'].iloc[0]) != (game - 1):
#                 prev_game_df = game_df_list[-2]
#                 if (prev_game_df['game'].iloc[0]) != (game - 1):
#                     raise Exception('Improperly formed csv. Game {} does not have a previous game.'.format(gameid))

#             if team_code_dict[prev_game_df['teamname'].iloc[0]] == blue_team:
#                 prev_game_result = prev_game_df['result'].iloc[0]
#             else:
#                 prev_game_result = prev_game_df['result'].iloc[1]

#             if prev_game_result == 1:
#                 F5_wr_dict[blue_team][0] = 0.05
#                 F5_wr_dict[red_team][0] = -0.05
#             else:
#                 F5_wr_dict[blue_team][0] = -0.05
#                 F5_wr_dict[red_team][0] = 0.05
        
#         if game >= 3:
#             # get the result of the last two games
#             prev_game_df = game_df_list[-1]
#             prev_game_2_df = game_df_list[-2]

#             if team_code_dict[prev_game_df['teamname'].iloc[0]] == blue_team:
#                 prev_game_result = prev_game_df['result'].iloc[0]
#             else:
#                 prev_game_result = prev_game_df['result'].iloc[1]

#             if team_code_dict[prev_game_2_df['teamname'].iloc[0]] == blue_team:
#                 prev_game_2_result = prev_game_2_df['result'].iloc[0]
#             else:
#                 prev_game_2_result = prev_game_2_df['result'].iloc[1]

#             if prev_game_result == 1 and prev_game_2_result == 1:
#                 F5_wr_dict[blue_team][0] = 0.1
#                 F5_wr_dict[red_team][0] = -0.1
#             elif prev_game_result == 0 and prev_game_2_result == 0:
#                 F5_wr_dict[blue_team][0] = -0.1
#                 F5_wr_dict[red_team][0] = 0.1
#             else:
#                 pass # nothing happens, as momentum is already set to 1 or -1

#             # if gameid == "ESPORTSTMNT02_2673605":
#             #     print("Team: {}, {} and {}".format(blue_team, prev_game_result, prev_game_2_result))


#         # add new row to the output dataframe
#         row_data = {"momentum": [F5_wr_dict[blue_team][0]]}
#         new_df = pd.DataFrame(row_data)
#         output_df = pd.concat([output_df, new_df], ignore_index=True)

#         # add game_df to game_df_list
#         game_df_list.append(game_df)
        
#         # reset the momentum for the next game
#         F5_wr_dict[blue_team][0] = 0
#         F5_wr_dict[red_team][0] = 0

#     return output_df

def create_F6_blueside_team(general_input_df, unique_teams):
    combined_headers = unique_teams

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    gameids = general_input_df['gameid'].unique()

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = general_input_df[general_input_df['gameid'] == gameid] 

        # get blue team name
        blue_team = game_df['teamname'].iloc[0]

        # one-hot team encoding
        team_list = [1 if header == blue_team else 0 for header in combined_headers]

        new_df = pd.DataFrame([team_list], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True) 

    return output_df

def create_F7_blueside_team(general_input_df, unique_teams):
    combined_headers = unique_teams

    # create the output dataframe
    output_df = pd.DataFrame(columns=combined_headers)

    gameids = general_input_df['gameid'].unique()

    for gameid in gameids: # iterate through all unique game_ids
        
        # get all rows with the same gameid
        game_df = general_input_df[general_input_df['gameid'] == gameid] 

        # get red team name
        red_team = game_df['teamname'].iloc[1]

        # one-hot team encoding
        team_list = [1 if header == red_team else 0 for header in combined_headers]

        new_df = pd.DataFrame([team_list], columns=combined_headers)
        output_df = pd.concat([output_df, new_df], ignore_index=True) 

    return output_df

if __name__ == '__main__':
    # pull data from input csvs
    filename = '../data/output/LCK_match_data_team.csv'
    general_input_df = create_df(filename) 

    filename = '../data/output/LCK_match_data_individual.csv'
    individual_input_df = create_df(filename) 

    filename = '../data/output/ALL_match_data_individual.csv'
    all_regions_input_df = create_df(filename) 
    
    # specify output df and csv
    output_filename = "../data/output/LCK_LogReg_Dataset_No_F4.csv"

    # Get unique team names
    unique_teams = general_input_df['teamname'].unique()

    # Add gameid and result
    F0_output_df = create_F0_results(general_input_df)

    # F0.5 simple win percentage
    F0_5_output_df = create_F0_5_win_score(general_input_df)

    # F1 standardized win score
    F1_output_df = create_F1_standardized_win_score(general_input_df)

    # F2 result (per region champion wr)
    F2_output_df = create_F2_region_champ_wr(individual_input_df)

    # F3 result (per player champion wr)
    F3_output_df = create_F3_player_champ_wr(individual_input_df)

    # F4 result (per patch champion wr)
    # F4_output_df = create_F4_patch_champ_wr(all_regions_input_df, general_input_df)

    # F5 result (team momentum)
    # F5_output_df = create_F5_team_momentum(general_input_df)

    # F6 result (per region champion wr)
    F6_output_df = create_F6_blueside_team(general_input_df, unique_teams)

    # F7 result (per region champion wr)
    F7_output_df = create_F7_blueside_team(general_input_df, unique_teams)

    # Adding features to the desired final df 
    output_df = pd.DataFrame()
    output_df = pd.concat([output_df, F0_output_df], axis=1)
    output_df = pd.concat([output_df, F0_5_output_df], axis=1)
    output_df = pd.concat([output_df, F1_output_df], axis=1)
    output_df = pd.concat([output_df, F2_output_df], axis=1)
    output_df = pd.concat([output_df, F3_output_df], axis=1)
    # output_df = pd.concat([output_df, F4_output_df], axis=1)
    # output_df = pd.concat([output_df, F5_output_df], axis=1)
    output_df = pd.concat([output_df, F6_output_df], axis=1)
    output_df = pd.concat([output_df, F7_output_df], axis=1)

    # export output df as csv
    export_df_to_csv(output_df, output_filename)


## INIT ##
# (1) Utility Class for Re-Useable Functions
# (2) Feature Class (init method, create method, update method)
# (3) Refine Data Function 