import pandas as pd
import numpy as np
import random

class ItemList:
    def __init__(self, length, condition="Temporal", locations=None, wordpool=None):
        self.length = length
        self.condition = condition
        self.items = (
            np.array([f"item_{i}" for i in range(length)])
            if wordpool is None
            else np.random.choice(wordpool, size=length, replace=False)
        )
        if locations is None:
            self.pos = None
        else:
            self.pos = locations.copy()
            # np.random.shuffle(self.pos)
        self.vals = np.zeros(self.length)
        if condition == "Temporal": self.temporalCondition()
        elif condition == "Random": self.randomCondition()

    def temporalCondition(self, decay_factor=1.0, effect_strength=1.0, mean_range=(30, 70), var_range=(13, 17)):
        """
        Generates a temporally correlated list of values using a GP-like covariance.
        
        Parameters:
        - decay_factor: controls correlation decay; smaller = faster decay, larger = slower.
        - effect_strength: multiplies the resulting correlation effect.
        - mean_range: tuple for uniform sampling of mean of final values.
        - var_range: tuple for uniform sampling of variance scaling.
        """
        n = self.length
        # Serial positions
        positions = np.arange(n)
        
        # Covariance matrix with exponential decay (like GP)
        cov = np.exp(-np.square(positions.reshape(-1,1) - positions) / (2 * decay_factor**2))
        cov += np.eye(n) * 1e-5  # numerical stability
        
        # Sample from multivariate normal
        vals = np.random.multivariate_normal(mean=np.zeros(n), cov=cov)
        
        # Apply effect strength
        vals *= effect_strength
        
        # Standardize
        vals = (vals - np.mean(vals)) / np.std(vals)
        
        # Optionally scale to desired mean/variance
        point_mean = np.random.uniform(*mean_range)
        point_var = np.random.uniform(*var_range)
        vals = vals * point_var + point_mean
        
        # Round and clip to 0-9
        # vals = np.clip(np.rint(vals), 0, 9).astype(int)
        
        self.vals = vals
        return self.vals


    def randomCondition(self):
        vals = self.temporalCondition().copy()
        np.random.shuffle(vals)
        self.vals = vals
        return self.vals
    
    def __str__(self):
        s = f"{self.condition} Associated List, Length: {self.length}\n"
        s += f"Names: {self.items}\n"
        s += f"Values: {getattr(self, 'vals', None)}\n"
        if self.pos is not None:
            s += f"Store Locations: {self.pos}\n"
        return s
    
    __repr__ = __str__
    
    def _repr_html_(self):
        return f"<pre>{str(self)}</pre>"

        

class SimulatedSubjectData:
    def __init__(self, subject, primacy_rate=0.2, recency_rate=0.2, t_cluster_rate=0.1, s_cluster_rate=0.1, val_acc_rate=0.5, recall_rate=0.705, intrusion_rate=0.1, item_lists=None):
        self.resetDF()
        self.subject = subject
        self.primacy_rate = primacy_rate
        self.recency_rate = recency_rate
        self.t_cluster_rate = t_cluster_rate
        self.s_cluster_rate = s_cluster_rate
        self.val_acc_rate = val_acc_rate
        self.recall_rate = recall_rate
        self.intrusion_rate = intrusion_rate
        self.item_lists = item_lists
    
    def resetDF(self):
        columns = [
            'experiment', 'intrusion', 'item_name', 'list',
            'recalled', 'recallpos', 'serialpos', 'subject', 'type',
            'val_list_avg', 'val_guess', 'pos_x', 'pos_y'
        ]
        self.df = pd.DataFrame(columns=columns)
        
    
    def generateLocations(self, r, n):
        return [(random.uniform(0, r), random.uniform(0, r)) for _ in range(n)]
    
    def generateLists(self, list_len, num_lists, wordpool=None, pos=None):
        """
        Generate a set of ItemLists for this subject.

        Parameters:
        - list_len: number of items per list
        - num_lists: total number of lists to generate
        - wordpool: optional list of words to use

        Returns:
        - item_lists: list of ItemList objects
        - conditions: list of strings, either "Temporal" or "Random"
        """
        # Half Temporal, half Random
        conditions = ["Temporal"] * (num_lists // 2) + ["Random"] * (num_lists // 2)
        random.shuffle(conditions)

        # Generate ItemList objects
        item_lists = [ItemList(list_len, condition, wordpool=wordpool) for condition in conditions]
        
        if pos is not None:
            for item_list in item_lists:
                item_list.pos = pos
        
        return item_lists, conditions

    def computeRecallProbabilities(
        self,
        list_len,
        item_locs=None,
        prev_serialpos=None
    ):
        """
        Generate a probability for each item in a list of length `list_len`.

        Parameters:
        - primacy_rate: weight for early items
        - recency_rate: weight for late items
        - t_cluster_rate: weight for temporal clustering (influenced by previous recall)
        - s_cluster_rate: weight for spatial clustering (influenced by previous recall position)
        - item_locs: list of (x, y) tuples for item positions
        - prev_serialpos: serial position of last recalled item (for clustering effects)

        Returns:
        - prob: np.array of probabilities (length `list_len`) normalized to [0,1]
        """
        # Base probability for all items
        prob = np.ones(list_len)

        # Primacy effect
        primacy_weights = np.linspace(1 + self.primacy_rate, 1, list_len)

        # Recency effect
        recency_weights = np.linspace(1, 1 + self.recency_rate, list_len)

        prob = prob * primacy_weights * recency_weights

        # Temporal clustering
        if prev_serialpos is not None:
            # print(list_len, flush=True)
            # print(prev_serialpos, flush=True)
            t_weights = 1 + self.t_cluster_rate / (1 + np.abs(np.arange(list_len) - prev_serialpos))
            prob = prob * t_weights

        # Spatial clustering
        if self.s_cluster_rate > 0 and prev_serialpos is not None and item_locs is not None:
            prev_pos = np.array(item_locs[prev_serialpos])
            distances = np.linalg.norm(np.array(item_locs) - prev_pos, axis=1)  # Euclidean distance
            s_weights = 1 + self.s_cluster_rate / (1 + distances)  # closer items get higher probability
            prob = prob * s_weights

        # Normalize to [0,1]
        prob = prob / np.max(prob)
        prob = prob ** 2
        # Scale down so max recall chance is base_recall_rate
        prob = prob * self.recall_rate

        return prob
        
    def generateData(self, list_len, num_lists, wordpool=None, gen_pos=False, reset=True):
        if reset:
            self.resetDF()
            self.item_lists = None
        
        experiment = "VCsim"
        conditions =None
        if self.item_lists is None:
            pos = generateLocations(1, list_len)
            item_lists, conditions = self.generateLists(list_len, num_lists, wordpool, pos=pos)
            self.item_lists = item_lists

        encoding_rows = []
        recall_rows = []
        for list_num, item_list in enumerate(self.item_lists):
            prev_recalled_pos = None  # reset for each list
            list_mean = np.mean(item_list.vals)

            # Guess based on list average and accuracy
            max_val_range = 70
            sd = (1 - self.val_acc_rate) * (max_val_range / 2)
            val_guess = int(np.clip(
                np.rint(np.random.normal(loc=list_mean, scale=sd)), 0, max_val_range
            ))

            for serialpos, item_name in enumerate(item_list.items):
                # Compute recall probabilities using the previous recalled item's serialpos
                if item_list.pos is None: locs = None
                else: locs = item_list.pos
                probs = self.computeRecallProbabilities(
                    list_len,
                    item_locs=locs,
                    prev_serialpos=prev_recalled_pos
                )

                # Spatial positions
                if item_list.pos is None:
                    pos_x, pos_y = np.nan, np.nan
                else:
                    pos_x, pos_y = item_list.pos[serialpos] if isinstance(item_list.pos[0], (list, tuple)) else (item_list.pos[serialpos], 0)

                # Decide recall
                is_intrusion = np.random.choice([0,1], p=[1 - self.intrusion_rate,self.intrusion_rate])
                recalled = np.random.rand() < probs[serialpos]

                # Intrusion handling
                recall_item_name = item_name
                if not recalled and is_intrusion:
                    other_items = [item for j, jl in enumerate(self.item_lists) if j != list_num for item in jl.items]
                    recall_item_name = random.choice(other_items)

                
                if recalled or is_intrusion:
                    # --- Recall row ---
                    recall_row = {
                        'experiment': experiment,
                        'intrusion': is_intrusion,
                        'item_name': recall_item_name,
                        'list': list_num,
                        'recalled': int(recalled),
                        'recallpos': serialpos,
                        # 'recallpos': serialpos if recalled else -1,
                        'serialpos': serialpos + 1,
                        'subject': self.subject,
                        'type': "REC_WORD",
                        'val_list_avg': list_mean,
                        'val_guess': val_guess,
                        'pos_x': pos_x,
                        'pos_y': pos_y
                    }
                    recall_rows.append(recall_row)

                # --- Encoding row ---
                encoding_row = {
                    'experiment': experiment,
                    'intrusion': -1,
                    'item_name': item_name,
                    'list': list_num,
                    'recalled': int(recalled),
                    'recallpos': -1,
                    'serialpos': serialpos + 1,
                    'subject': self.subject,
                    'type': "WORD",
                    'val_list_avg': list_mean,
                    'val_guess': val_guess,
                    'pos_x': pos_x,
                    'pos_y': pos_y
                }
                encoding_rows.append(encoding_row)

                # Update counters
                if recalled:
                    prev_recalled_pos = serialpos  # update last recalled position


        # Combine encoding and recall events
        self.df = pd.concat([pd.DataFrame(encoding_rows), pd.DataFrame(recall_rows)], ignore_index=True)
        return self.df.copy()

def main():
    # Create a simulated subject
    subject1 = SimulatedSubjectData("subject1")
    
    # Generate data: 100 lists, each of length 15
    df = subject1.generateData(list_len=15, num_lists=100)
    
    # Show first few rows
    print(df.head())
    
    # Optionally, return the dataframe
    return df

# Standard Python entry point
if __name__ == "__main__":
    df = main()
