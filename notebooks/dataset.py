import pandas as pd
import numpy as np
from spektral.data import Dataset, Graph

class DotaV1(Dataset):
    def __init__(self, df_combined: pd.DataFrame, features: pd.DataFrame,  **kwargs):
        '''Generates a list of Graph objects
        df_combined: match result (radiant_win), hero picks (hero0_pick ...), hero slots (hero0_slot...) [dataframe]
        features: all hero ids (hero_id), desired features (feature1,2,3,etc., name not important) [dataframe]
        '''
        graphs_radiant = [self.get_graph(index, match, features, 'radiant') for index, match in df_combined.iterrows()]
        graphs_dire = [self.get_graph(index, match, features, 'dire') for index, match in df_combined.iterrows()]
        self.graphs = graphs_radiant + graphs_dire
        # self.graphs = graphs_radiant
        super().__init__(**kwargs)
        
    def get_graph(self, index, match, features, team):
        '''Generates a single graph based on a single match'''
        # Status
        if (index+1)%1000==0:
            print(f'Graph {team} {index+1}')

        # Extract hero ids and match result, depending on radiant or dire perspective
        # Reduce match df to columns: hero0_slot, ..., hero9_slot
        slots = match[[f'hero{i}_slot' for i in range(0,10)]]

        # Based on team: select slots filter , assign match result
        if team=='radiant':
            slots = slots[slots<5] # radiant slots 0,1,2,3,4
            #heroes = match[['hero0','hero1','hero2','hero3','hero4']].values
            y = float(match['radiant_win'])
        elif team=='dire':
            slots = slots[slots>127] # dire slots 128,129,130,131,132
            #heroes = match[['hero5','hero6','hero7','hero8','hero9']].values
            y = 1-float(match['radiant_win'])
        else:
            raise ValueError('Incorrect team specified in "get_graph" matchod. Use "radiant" or "dire"')

        # Pick columns based on team slots determined above
        pick_columns = [f'{herox[:5]}_pick' for herox in slots.index]
        heroes = match[pick_columns].values
        heroes = [hero for hero in heroes if hero!=0] # remove hero id 0 (these are invalid)
        
        # Create feature matrix
        x = features.loc[heroes].iloc[:,3:]
        x = x.to_numpy(dtype='float')
        
        # Adjacency matrix
        a = np.ones([5,5], dtype='float32')

        g = Graph(x=x, a=a, y=y)
        return g
        
    def read(self):
        return self.graphs