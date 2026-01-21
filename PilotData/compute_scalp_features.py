#!/usr/bin/env python

import sys

import sys
import cmlreaders as cml
import numpy as np
import seaborn as sns
import pandas as pd
from ptsa.data.filters import MorletWaveletFilter, ButterworthFilter, MonopolarToBipolarMapper
from ptsa.data.timeseries import TimeSeries
from scipy.stats import zscore
import xarray
from ptsa.data.concat import concat
from cmldask import CMLDask as da

def free_epochs(times, duration, pre, post, start=None, end=None):
    # (list(vector(int))*int*int*int) -> list(vector(int))
    """
    Given a list of event times, find epochs between them when nothing is happening
    Parameters:
    -----------
    times:
        An iterable of 1-d numpy arrays, each of which indicates event times
    duration: int
        The length of the desired empty epochs
    pre: int
        the time before an upcoming vocalization to exclude
    post: int
        The time after a preceding vocalization to exclude
    """
    n_trials = len(times) #len of a list of arrays is the number of arrays the list contains
    epoch_times = []
    for i in range(n_trials): #for each trial in a session
        ext_times = times[i] #grab the timing of recall events for this trial
        if start is not None: #if there is a start, add the timing of the start to the times
            ext_times = np.append([start[i]], ext_times)
        if end is not None: #if there is an end, add the timing of the end to the times
            ext_times = np.append(ext_times, [end[i]])
        pre_times = ext_times - pre #array of the times occuring 1000ms prior to each event 
        post_times = ext_times + post #array of the times occuring 1000ms after each event
        #subtract the post-event timing of one event from the pre-event timing of the next event, aka inter-event timing.
        interval_durations = pre_times[1:] - post_times[:-1] 
        #free_intervals is an array of integers indicating the iloc of silent periods (beginning at the iloc of the post-time)
        free_intervals = np.where(interval_durations > duration)[0]
        trial_epoch_times = []
        for interval in free_intervals:
            begin = post_times[interval] #start of silence is after word offset
            finish = pre_times[interval + 1] - duration #end of silence is the pre-timing onset minus duration
            interval_epoch_times = range(int(begin), int(finish), int(duration)) 
            trial_epoch_times.extend(interval_epoch_times) #array of the start and ends of durations of silence
        epoch_times.append(np.array(trial_epoch_times)) #list of arrays of start and end of durations for each trial
    #select the trial with the most periods of silence, create an array of trial x silence periods, fill it with random values
    epoch_array = np.empty((n_trials, max([len(x) for x in epoch_times])))
    epoch_array[...] = -np.inf #set all the values to negative infinity
    for i, epoch in enumerate(epoch_times): #for each array in the list of arrays with silence markers
        epoch_array[i, :len(epoch)] = epoch #fill in all of the values where there is a value avaiable
    #aka you end up with an array where each trial has a set of duration periods, and to keep it all the same size,
    #each one has the length of the one with the most periods of silence with -np.inf as remainder values
    return epoch_array

def create_baseline_events(events, start_time, end_time):
    '''
    Match recall events to matching baseline periods of failure to recall.
    Baseline events all begin at least 1000 ms after a vocalization, and end at least 1000 ms before a vocalization.
    Each recall event is matched, wherever possible, to a valid baseline period from a different list within 3 seconds
     relative to the onset of the recall period.
    Parameters:
    -----------
    events: The event structure in which to incorporate these baseline periods
    start_time: The amount of time to skip at the beginning of the session (ms)
    end_time: The amount of time within the recall period to consider (ms)
    '''
    #In the end, you get a np.recarray of all events for the subject/subjects you passed in for this experiment
    #complete with periods of silence lasting 1000ms that are located at the mstime that they occured.
    #The periods of silence that appear are only those which have a match, but the match is not evident, you
    #need to go in and create a match key manually. 
    #Start time for NICLS: 1000ms. End time: 90000ms.
    exp = events.experiment[0] #Get the experiment name from the experiment that is in the first row
#     sub = events.subject[0] #Get the subject code from the subject that is in the first row
    subjects= np.unique(events.subject)
    
    all_events = []
    #in case you want to create matched periods for multiple subjects at once, this loop allows it 
    for subject in subjects:
        sub_events = events[events.subject == subject] 
        problem_sessions = {'LTP497':5}
        for session in np.unique(sub_events.session): #go through a list of the sessions
            skip = False
            for key, val in problem_sessions.items(): #pre-defined dictionary of problem sessions
                if (subject == key) and (session == val):
                    skip = True #if this is a problem session, set skip to True
            if skip == True:
                continue #if we want to skip this session, skip this session
            sess_events = sub_events[(sub_events.session == session)] #if this is a good session, grab the events from the og DF
            irts = np.append([0], np.diff(sess_events.mstime)) #create an array of the periods of time between each event
            #create a dataframe of just the correct recalls/repetitions that occur 1000ms after the previous event in that session
            rec_events = sess_events[(sess_events.type == 'REC_WORD') & (sess_events.intrusion == 0) & (irts >= 1000)]
            #create a dataframe of all words & vocalizations during all recall periods in that session
            voc_events = sess_events[((sess_events.type == 'REC_WORD') | (sess_events.type == 'REC_WORD_VV'))]
            #create a dataframe of the recall period starts
            starts = sess_events[(sess_events.type == 'REC_START')]
            #create a dataframe of the recall period stops
            ends = sess_events[(sess_events.type == 'REC_STOP')]
            #create a tuple of the trials in this session
            rec_lists = tuple(np.unique(starts.trial))
            #create a list of lists containing the timing of each vocalization in a trial
            times = [voc_events[(voc_events.trial == lst)].mstime for lst in rec_lists]
            start_times = starts.mstime #list of start times
            end_times = ends.mstime #list of end times
            epochs = free_epochs(times, 1000, 1000, 1000, start=start_times, end=end_times) #see function
            rel_times = [(t - i)[(t - i > start_time) & (t - i < end_time)] for (t, i) in
                         zip([rec_events[rec_events.trial == lst].mstime for lst in rec_lists ], start_times)
                         ]
            #rel_times is a list of arrays that has the time of recall events relative to the start for each trial in a session
            #only for recall events where the inter-event timing is greater than the start time of recall and less than the end 
            #time of retrieval, aka within the retrieval window
            rel_epochs = epochs - start_times[:, None]
            #rel_epochs takes the timing of deliberation periods in each trial and subtracts the start time from them, aka 
            #relative timing of silence
            full_match_accum = np.zeros(epochs.shape, dtype=bool) #initialize an array for deliberation periods
            for (i, rec_times_list) in enumerate(rel_times): #for each trial
                is_match = np.empty(epochs.shape, dtype=bool) #initialize an array 
                is_match[...] = False #set values to false
                for t in rec_times_list: #for each valid recall
                    is_match_tmp = np.abs((rel_epochs - t)) < 3000 #is the timing of recall within 3000ms of silence in any of the trials
                    is_match_tmp[i, ...] = False #exclude the trial that the recall is in
                    good_locs = np.where(is_match_tmp & (~full_match_accum)) 
                    #where is there a valid silence that hasn't been taken (list number first, then event location of silence)
                    if len(good_locs[0]): #if there are valid delibs
                        choice_position = np.argmin(np.mod(good_locs[0] - i, len(good_locs[0])))
                        #what is the minimum remainder of the division of the trial location of good locations minus the current trial
                        #with the number of overall good locations (aka what is the closest list)
                        choice_inds = (good_locs[0][choice_position], good_locs[1][choice_position])
                        #choose the closest list (indexed by the mod) and its corresponding silence event
                        full_match_accum[choice_inds] = True #make sure that silence doesn't get chosen again
            matching_epochs = epochs[full_match_accum] #select timing of silence periods that have matches
            new_events = np.zeros(len(matching_epochs), dtype=sess_events.dtype).view(np.recarray) #initialize 
            for i, _ in enumerate(new_events): #make df rows and fill in
                new_events[i].mstime = matching_epochs[i] 
                new_events[i].type = 'REC_BASE'
            new_events.recalled = 0
            new_events.session = session
            new_events.subject = subject
            new_events.experiment = exp
            new_events.classifier = 'X'
            new_events.eegfile = ''
            new_events.item = ''
            new_events.phase = ''
            new_events.protocol = 'ltp'
            new_events.store = ''
            new_events.intrusion = 0
            merged_events = np.concatenate((sess_events, new_events)).view(np.recarray)
            #merge session events with matched periods of silence, order by time
            merged_events.sort(order='mstime')
            for (i, event) in enumerate(merged_events):
                if event.type == 'REC_BASE': #fill in the below vairables
                    merged_events[i].session = merged_events[i - 1].session
                    merged_events[i].trial = merged_events[i - 1].trial
                    merged_events[i].eegfile = merged_events[i - 1].eegfile
                    merged_events[i].eegoffset = merged_events[i - 1].eegoffset + (
                    merged_events[i].mstime - merged_events[i - 1].mstime) + 1000
                    merged_events[i].mstime += 1000
            all_events.append(merged_events)
    return np.concatenate(all_events).view(np.recarray)

def compute_scalp_features(subject, settings_path='/home1/joycerose14/DougEtal24/NiclsReadOnly_retrieval.pkl', save_path='/home1/joycerose14/DougEtal24/'):
    """
    Compute log-transformed powers, averaged over time and stacked as (frequency, channel) to create features
    These can later be normalized along the event axis.
    """
    settings = da.Settings.Load(settings_path)
    data = cml.get_data_index(kind = 'ltp')
    data = data[(data['experiment']==settings.experiment)&(data['subject']==subject)].sort_values('session').reset_index()
    problem_sessions = [('LTP497',5),('LTP448',0)]
    feats = []
    for i, row in data.iterrows():
        print(f"Reading session {i} data")
        skip = False
        for tup in problem_sessions: #pre-defined dictionary of problem sessions
            if (subject == tup[0]) and (i == tup[1]):
                skip = True
        if skip == True:
            continue #if we want to skip this session, skip this session
        # intialize data reader, load words events and buffered eeg epochs
        r = cml.CMLReader(subject=subject, experiment=row['experiment'], session=row['session'])
        evs = r.load('task_events')
        if settings.type == "encoding":
            word_evs = evs[(evs.type=='WORD')&(evs.eegoffset!=-1)]
            if len(word_evs)==0:
                continue # sync pulses not recorded
            eeg = r.load_eeg(word_evs, rel_start=settings.rel_start, 
                             rel_stop=settings.rel_stop, clean=settings.clean
                            ).to_ptsa()
        elif settings.type == "retrieval":
            rec_evs = evs.query('type in ["REC_START", "REC_STOP", "REC_WORD", "REC_WORD_VV"] & phase != "practice" & eegoffset != -1').reset_index(drop = True)
            if len(rec_evs)==0:
                continue # sync pulses not recorded
            rec_evs['irt'] = np.append([0], np.diff(rec_evs.mstime))
            for i, row in rec_evs.iterrows(): 
                if (rec_evs.type.iloc[i - 1] == "REC_START"):
                    row['irt'] = 9999
#             rec_evs = rec_evs.query('(irt >= 1999) | (type in ["REC_START", "REC_STOP"])') #should I only be selecting vocalizations with 2s in between? 
            rec_evs = pd.DataFrame.from_records(
                create_baseline_events(rec_evs.to_records(), start_time=3000, end_time=90000)
            ).reset_index(drop = True)
            rec_evs = rec_evs.query("type in ['REC_WORD', 'REC_BASE']").reset_index(drop = True)
            current_trial = 0
            prev_trial = 0
            rec_words = []
            rec_evs['repetition'] = 0
            for ev_index, ev_row in rec_evs.iterrows():
                current_trial = ev_row['trial']
                if ev_row['type'] == "REC_BASE":
#                     rec_evs.loc[rec_evs.index == ev_index, 'eegoffset'] += 1000
#                     rec_evs.loc[rec_evs.index == ev_index, 'mstime'] += 1000
                    prev_trial = ev_row['trial']
                    continue
                if current_trial != prev_trial:
                    rec_words = []
                if ev_row['item'] in rec_words:
                    rec_evs.loc[rec_evs.index == ev_index, 'repetition'] = 1
                rec_words.append(ev_row['item'])
                prev_trial = ev_row['trial']
            rec_evs = rec_evs.query('type in ["REC_WORD","REC_BASE"] & repetition != 1 & intrusion == 0')
            eeg = r.load_eeg(rec_evs, rel_start=settings.rel_start, 
                             rel_stop=settings.rel_stop, clean=settings.clean).to_ptsa()
            print(type(eeg))
        # select relevant channels
        eeg = eeg.add_mirror_buffer(settings.buffer_time/1000)
        if settings.reference == 'average':
            eeg = eeg[:, :128]
            if eeg.channel[0].str.startswith('E') and not settings.clean: # EGI system
                eeg.drop_sel({'channel':['E8', 'E25', 'E126', 'E127']})
            eeg -= eeg.mean('channel')
        elif settings.reference == 'bipolar':
            bipolar_pairs = np.loadtxt("/home1/jrudoler/biosemi_cap_bipolar_pairs.txt", dtype=str)
            mapper = MonopolarToBipolarMapper(bipolar_pairs, channels_dim='channel')
            eeg = eeg.filter_with(mapper)
            eeg = eeg.assign_coords({"channel":np.array(["-".join(pair) for pair in eeg.channel.values])})
        else:
            raise ValueError("reference setting unknown")
        # filter out line noise at 60 and 120Hz
        eeg = ButterworthFilter(filt_type='stop', freq_range=[58, 62], order=4).filter(eeg)
        eeg = ButterworthFilter(filt_type='stop', freq_range=[118, 122], order=4).filter(eeg)
        # highpass filter to account for drift
        eeg = ButterworthFilter(filt_type='highpass', freq_range=1).filter(eeg)
        pows = MorletWaveletFilter(settings.freqs,
                                   width=settings.width,
                                   output='power',
                                   cpus=25).filter(eeg)
        del eeg
        pows = pows.remove_buffer(settings.buffer_time / 1000) + np.finfo(float).eps/2.
        pows = pows.reduce(np.log10)
        # swap order of events and frequencies --> result is events x frequencies x channels x time
        # next, average over time
        pows = pows.transpose('event', 'frequency', 'channel', 'time').mean('time')
        # reshape as events x features
        pows = pows.stack(features=("frequency", "channel"))
        pows = pows.reduce(func=zscore, dim='event', keep_attrs=True, ddof=1)
        feats.append(pows)
        del pows
    feats = concat(feats, dim='event')
    feats = feats.assign_attrs(settings.__dict__)
    if settings.save:
        if settings.type == "encoding":
            feats.to_hdf(save_path+f'{subject}_encoding_feats.h5')
        elif settings.type == "retrieval":
            feats.to_hdf(save_path+f'{subject}_retrieval_feats.h5')
    return feats

if __name__=="__main__":
    compute_scalp_features(sys.argv[1], experiment="NiclsCourierClosedLoop", save=True, save_path='/scratch/nicls_intermediate/closed_loop/encoding_powers/')
