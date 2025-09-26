import pandas as pd
import numpy as np

# ------------------ ItemList ------------------
class ItemList:
    def __init__(self, length, condition="Temporal", locations=None, wordpool=None, complex_params=None, simple_params=None, rng=None):
        self.length = length
        self.condition = condition
        if rng is None:
            seed = int(np.random.SeedSequence().entropy)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

        # Wordpool handling
        self.items = (
            np.array([f"item_{i}" for i in range(length)])
            if wordpool is None
            else self.rng.choice(wordpool, size=length, replace=False)
        )

        # Locations
        self.pos = None if locations is None else np.array(locations.copy())

        # complex Temporal parameters
        default_complex_params = {
            "decay_factor": 1.0,
            "effect_strength": 1.0,
            "mean_range": (4, 10),
            "var_range": (2, 6),
            "val_range": (0,15)
        }
        self.complex_params = default_complex_params if complex_params is None else {**default_complex_params, **complex_params}
        
        # complex Temporal parameters
        default_simple_params = {
            "val_range": (0,20),
            "recency_buf": 3,
            "primacy_buf": 2, 
            "num_chosen": 4
        }
        self.simple_params = default_simple_params if simple_params is None else {**default_simple_params, **simple_params}

        self.vals = np.zeros(self.length)
        if condition == "Temporal":
            self.temporalConditionComplex()
        elif condition == "Random":
            self.randomCondition()

            
#     def temporalConditionSimple(self):
#         n = self.length
#         # decay_factor = self.simple_params["decay_factor"]
#         # effect_strength = self.simple_params["effect_strength"]
#         # mean_range = self.simple_params["mean_range"]
#         # var_range = self.gp_psimple_paramsarams["var_range"]
#         val_range = self.simple_params["val_range"]
        
#         # number of items from start of list that should not be included
#         primacy_buf = self.simple_params['primacy_buf']
#         # number of items from end of list that should not be included
#         recency_buf = self.simple_params['recency_buf']
        
#         positions = np.arange(n)
        
        
    
    def temporalConditionComplex(self):
        n = self.length
        decay_factor = self.complex_params["decay_factor"]
        effect_strength = self.complex_params["effect_strength"]
        mean_range = self.complex_params["mean_range"]
        var_range = self.complex_params["var_range"]
        val_range = self.complex_params["val_range"]

        positions = np.arange(n)
        cov = np.exp(-np.square(positions.reshape(-1, 1) - positions) / (2 * decay_factor**2))
        cov += np.eye(n) * 1e-5

        vals = self.rng.multivariate_normal(mean=np.zeros(n), cov=cov)
        vals *= effect_strength
        vals = (vals - np.mean(vals)) / np.std(vals)

        point_mean = self.rng.uniform(*mean_range)
        point_var = self.rng.uniform(*var_range)
        vals = vals * point_var + point_mean

        # ensure non-negative integers
        vals = np.clip(vals, val_range[0], val_range[1])
        vals = np.rint(vals).astype(int)

        self.vals = vals
        return self.vals


    def randomCondition(self, complex=True):
        if complex:
            vals = self.temporalConditionComplex().copy()
        self.rng.shuffle(vals)
        self.vals = vals
        return self.vals
    
    def __str__(self):
        s = f"ItemList ({self.condition}) - Length: {self.length}\n"
        s += f"Items: {self.items.tolist()}\n"
        s += f"Values: {np.round(self.vals, 2).tolist()}\n"
        if self.pos is not None:
            s += f"Positions: {self.pos.tolist()}\n"
        s += f"Simple params: {self.simple_params}\n"
        s += f"Complex params: {self.complex_params}\n"
        s += f"RNG: {self.rng}\n"
        return s

    __repr__ = __str__

    def _repr_html_(self):
        return f"<pre>{str(self)}</pre>"

# ------------------ ItemProcessor ------------------
class ItemProcessor:
    def __init__(self, itemlist, rng=None):
        self.items = itemlist.items.copy()
        self.vals = itemlist.vals.copy()
        self.pos = itemlist.pos.copy() if itemlist.pos is not None else None
        self.length = itemlist.length
        self.serialpos = list(range(itemlist.length))
        if rng is None:
            seed = int(np.random.SeedSequence().entropy)
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng

    def hasItems(self):
        return self.length > 0

    def pickFirstItem(self, probs):
        if len(probs) == 0:
            raise ValueError("first_recall probs empty")

        idx = self.rng.choice(np.arange(len(probs)), p=probs)
        item = self.items[idx]
        val = self.vals[idx]
        pos = self.pos[idx] if self.pos is not None else (np.nan, np.nan)
        serialpos = self.serialpos[idx]

        self._removeItem(idx)
        return item, val, pos, serialpos

    def pickItem(self, lag_probs, prev_serialpos):
        if not lag_probs:
            raise ValueError("lag_probs dict is empty")

        lags = np.array(list(lag_probs.keys()))
        probs = np.array(list(lag_probs.values()), dtype=float)
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs.sum()

        # Precompute valid serial positions
        valid_serials = np.array([s for s in self.serialpos if (s - prev_serialpos) in lags])

        if len(valid_serials) == 0:
            # fallback: pick any remaining item uniformly
            item_idx = self.rng.integers(0, len(self.serialpos))
            chosen_lag = self.serialpos[item_idx] - prev_serialpos
        else:
            # map valid serials to their corresponding lags
            valid_lags = valid_serials - prev_serialpos
            # get probabilities for these valid lags
            valid_probs = np.array([lag_probs.get(lag, 0) for lag in valid_lags], dtype=float)
            # filter out zero-prob lags
            nonzero_mask = valid_probs > 0
            valid_lags = valid_lags[nonzero_mask]
            valid_serials = valid_serials[nonzero_mask]
            valid_probs = valid_probs[nonzero_mask]

            if len(valid_serials) == 0:
                # fallback: uniform over remaining items
                item_idx = self.rng.integers(0, len(self.serialpos))
                chosen_lag = self.serialpos[item_idx] - prev_serialpos
            else:
                valid_probs /= valid_probs.sum()
                item_idx_in_valid = self.rng.choice(len(valid_serials), p=valid_probs)
                item_idx = self.serialpos.index(valid_serials[item_idx_in_valid])
                chosen_lag = valid_lags[item_idx_in_valid]

        item = self.items[item_idx]
        val = self.vals[item_idx]
        pos = self.pos[item_idx] if self.pos is not None else (np.nan, np.nan)
        serialpos = self.serialpos[item_idx]

        self._removeItem(item_idx)
        return item, val, pos, serialpos, chosen_lag



    def _removeItem(self, idx):
        self.items = np.delete(self.items, idx)
        self.vals = np.delete(self.vals, idx)
        if self.pos is not None:
            self.pos = np.delete(self.pos, idx, axis=0)
        self.serialpos.pop(idx)
        self.length -= 1
        
    def __str__(self):
        s = f"ItemProcessor - {self.length} items remaining\n"
        s += f"Items: {self.items.tolist()}\n"
        s += f"Values: {np.round(self.vals,2).tolist()}\n"
        s += f"Serial positions: {self.serialpos}\n"
        if self.pos is not None:
            s += f"Positions: {self.pos.tolist()}\n"
        s += f"RNG: {self.rng}\n"
        return s

    __repr__ = __str__
    def _repr_html_(self):
        return f"<pre>{str(self)}</pre>"

# ------------------ SimulatedSubjectData ------------------
class SimulatedSubjectData:
    def __init__(self, subject, first_recall, lag_crp, recall_rate, value_acc, 
                 item_lists=None, simple_params=None,complex_params=None, seed=None):
        self.resetDF()
        self.subject = subject
        self.first_recall = first_recall
        self.lag_crp = lag_crp
        self.item_lists = item_lists
        self.recall_rate = recall_rate
        self.value_acc = value_acc

        # complex Temporal parameters
        default_complex_params = {
            "decay_factor": 1.0,
            "effect_strength": 1.0,
            "mean_range": (4, 10),
            "var_range": (2, 6),
            "val_range": (0,15)
        }
        self.complex_params = default_complex_params if complex_params is None else {**default_complex_params, **complex_params}
        
        # complex Temporal parameters
        default_simple_params = {
            "val_range": (0,20),
            "recency_buf": 3,
            "primacy_buf": 2, 
            "num_chosen": 4
        }
        self.simple_params = default_simple_params if simple_params is None else {**default_simple_params, **simple_params}

        # RNG setup
        if seed is None:
            seed = int(np.random.SeedSequence().entropy)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def resetDF(self):
        columns = [
            'experiment', 'item_name', 'item_val', 'list',
            'recalled', 'recallpos', 'serialpos', 'subject', 'type',
            'val_list_avg', 'val_guess', 'pos_x', 'pos_y'
        ]
        self.df = pd.DataFrame(columns=columns)

    def generateLocations(self, r, n):
        return [(self.rng.uniform(0, r), self.rng.uniform(0, r)) for _ in range(n)]

    def generateLists(self, list_len, num_lists, wordpool=None, pos=None):
        half = num_lists // 2
        conditions = ["Temporal"] * half + ["Random"] * half
        if num_lists % 2 == 1:
            conditions.append(self.rng.choice(["Temporal", "Random"]))
        self.rng.shuffle(conditions)

        item_lists = [
            ItemList(list_len, condition, wordpool=wordpool, locations=pos, 
                     complex_params=self.complex_params, simple_params=self.simple_params, rng=self.rng)
            for condition in conditions
        ]
        self.item_lists = item_lists
        return item_lists, conditions

    def _guessByMean(self, subset_mean, max_val_range=15):
        sd = (1 - self.value_acc) * (max_val_range / 2)
        val_guess = int(np.clip(np.rint(self.rng.normal(loc=subset_mean, scale=sd)), 0, max_val_range))
        return val_guess

    def convertCRP(self, crp):
        crp = np.array(crp, dtype=float)
        center = len(crp) // 2
        lag_probs = {i - center: crp[i] for i in range(len(crp)) if i != center}
        return self.normalizeDict(lag_probs)

    def normalizeDict(self, lag_probs):
        total = sum(lag_probs.values())
        if total <= 0:
            n = len(lag_probs)
            return {lag: 1/n for lag in lag_probs}
        return {lag: p/total for lag, p in lag_probs.items()}

    def generateData(self, list_len, num_lists, wordpool=None, gen_pos=False, guess_by_subset=True, reset=True):
        if reset:
            self.resetDF()
            self.item_lists = None

        if self.item_lists is None:
            pos = self.generateLocations(1, list_len) if gen_pos else None
            self.generateLists(list_len=list_len, num_lists=num_lists, wordpool=wordpool, pos=pos)

        encoding_rows, recall_rows = [], []
        experiment = "VCsim"

        for list_num, item_list in enumerate(self.item_lists):
            recall_dict = {}
            first_item = True
            list_mean = np.mean(item_list.vals)
            item_proc = ItemProcessor(item_list, rng=self.rng)

            lag_probs = self.convertCRP(self.lag_crp.copy())
            prev_serialpos = -1
            recall_pos = 0

            while item_proc.hasItems():
                if first_item:
                    item, val, pos, serialpos = item_proc.pickFirstItem(self.first_recall)
                    first_item = False
                    # print("picked first item")
                else:
                    item, val, pos, serialpos, chosen_lag = item_proc.pickItem(lag_probs, prev_serialpos)
                    # print(f"picked {recall_pos} item")
                
                if self.rng.random() > self.recall_rate:
                    continue

                prev_serialpos = serialpos
                recall_row = {
                    'experiment': experiment,
                    'item_name': item,
                    'item_val': val,
                    'list': list_num,
                    'recalled': 1,
                    'recallpos': recall_pos + 1,
                    'serialpos': serialpos + 1,
                    'subject': self.subject,
                    'type': "REC_WORD",
                    'val_list_avg': list_mean,
                    'val_guess': np.nan,
                    'pos_x': pos[0],
                    'pos_y': pos[1]
                }
                recall_dict[item] = recall_row
                recall_pos += 1

            # Value guess
            if guess_by_subset and recall_dict:
                recalled_vals = [row['item_val'] for row in recall_dict.values()]
                val_guess = self._guessByMean(np.mean(recalled_vals))
            else:
                val_guess = self._guessByMean(list_mean)

            for serialpos, item in enumerate(item_list.items):
                recalled = item in recall_dict
                recall_row = recall_dict[item] if recalled else None
                encoding_row = {
                    'experiment': experiment,
                    'item_name': item,
                    'item_val': item_list.vals[serialpos],
                    'list': list_num,
                    'recalled': int(recalled),
                    'recallpos': recall_row['recallpos'] if recalled else -1,
                    'serialpos': serialpos + 1,
                    'subject': self.subject,
                    'type': "WORD",
                    'val_list_avg': list_mean,
                    'val_guess': val_guess,
                    'pos_x': item_list.pos[serialpos][0],
                    'pos_y': item_list.pos[serialpos][1]
                }
                encoding_rows.append(encoding_row)
                if recalled:
                    recall_row['val_guess'] = val_guess
                    recall_rows.append(recall_row)

        self.df = pd.concat([pd.DataFrame(encoding_rows), pd.DataFrame(recall_rows)], ignore_index=True)
        return self.df.copy()

    # BROKEN
    def getSubjectLagCRP(self, list_len):
        data = self.df.copy()
        center = list_len - 1
        min_lag = -center
        max_lag = center + 1
        actual = {lag: 0 for lag in range(min_lag, max_lag)}
        possible = {lag: 0 for lag in range(min_lag, max_lag)}
        for session_id, session_data in data.groupby('session'):
            recalls = session_data[session_data.type == 'REC_WORD']
            # print(recalls)
            words = session_data[session_data.type == 'WORD']
            if recalls.empty or words.empty:
                print(f"session {session_id} has no events")
                continue
            # print(recalls.intruded)
            recalls = recalls[(recalls['trial'] != -999)]
            word_to_pos = dict(zip(words['item'], words['serialpos']))
            # print(word_to_pos)
            # print(recalls)
            for trial in recalls['trial'].unique():
                trial_words = words[words['trial'] == trial]['item'].tolist()
                trial_recalls = (recalls[recalls['trial'] == trial]
                                 .sort_values('rectime')
                                 .drop_duplicates('item'))

                if len(trial_recalls) < 2:
                    print(f"session {session_id}, trial {trial} doesn't have enough events")
                    continue
                trial_recalls = trial_recalls[trial_recalls['item'].isin(trial_words)]
                recall_pos = [word_to_pos[w] for w in trial_recalls['item']]
                # print(recall_pos)
                for i, cur in enumerate(recall_pos[:-1]):
                    lag = recall_pos[i+1] - cur
                    if min_lag <= lag <= max_lag and lag != 0:
                        actual[lag] += 1
                    for pos in set(range(1, list_len+1)) - set(recall_pos[:i+1]):
                        pl = pos - cur
                        if min_lag <= pl <= max_lag and pl != 0:
                            possible[pl] += 1

        # build CRP array
        full_len = 2*list_len - 1
        crp = np.full(full_len, np.nan)
        center = list_len - 1
        for lag in range(min_lag, max_lag):
            idx = center + lag
            if 0 <= idx < full_len:
                crp[idx] = (actual[lag] / possible[lag]) if possible[lag] > 0 else np.nan
        crp[center] = 0.0
        self.lag_crp = crp.copy()
        return crp
    
    def __str__(self):
        s = f"SimulatedSubjectData (Subject: {self.subject})\n"
        s += f"First recall probs: {self.first_recall}\n"
        s += f"Lag CRP: {self.lag_crp}\n"
        s += f"Recall rate: {self.recall_rate}\n"
        s += f"Value accuracy: {self.value_acc}\n"
        s += f"Simple params: {self.simple_params}\n"
        s += f"Complex params: {self.complex_params}\n"
        s += f"Seed: {self.seed}\n"
        s += f"RNG: {self.rng}\n"
        if self.item_lists is not None:
            s += f"{len(self.item_lists)} ItemList(s), each length: {self.item_lists[0].length}\n"
        else:
            s += "Item lists: None\n"
        s += f"DataFrame shape: {self.df.shape}\n"
        return s

    __repr__ = __str__
    def _repr_html_(self):
        return f"<pre>{str(self)}</pre>"
