" A function for filtering BAREfoot's prediction"
import pandas as pd
import numpy as np

def find_consecutive_repeats(vector):
    #  finds consecutive repeats in a vector, returns their value,
    #  how many times they repeated in each sequence, and their index in the vector
    vector = np.array(vector)
    repeats = []
    i = 0
    while i < len(vector) - 1:
        count = 1
        while i < len(vector) - 1 and vector[i] == vector[i + 1]:
            count += 1
            i += 1
        if count >= 1:
            repeats.append((vector[i], count, i - count + 1))
        i += 1
    return np.array(repeats, dtype=object)

def vector_polish(vector, cat_to_change=1, change_to=0, min_bout=3, min_after_bout=1, repeat=2):
#replace a repeated catagory (cat_to_change) with a different one (change_to) if the size is smaller than min_repeat
# [0 0 0 0 1 1 1 0 0 0 0 ],if the min_repeat is 5, then they are replace [0 0 0 0 0 0 0 0 0 0 0 ]
    print("polishing...")
    for r in range(0,repeat):
        consec_repeat_mat=pd.DataFrame(find_consecutive_repeats(vector))
        vector=pd.DataFrame(vector+0)
        for i in range(0,len(consec_repeat_mat)-1):
            if consec_repeat_mat.iloc[i,0]==cat_to_change and consec_repeat_mat.iloc[i,1]<min_bout and consec_repeat_mat.iloc[i + 1,1]>=min_after_bout:
               vector.iloc[consec_repeat_mat.iloc[i,2]:(consec_repeat_mat.iloc[i,2]+consec_repeat_mat.iloc[i,1]),0] = change_to
    return pd.DataFrame(vector+0)

def vector_bridge(vector, cat_to_change=0, change_to=1, min_bout=3, max_gap=2, min_after_gap=1):
    print("bridging...")
    consec_repeat_mat=pd.DataFrame(find_consecutive_repeats(vector))
    padded_vector=pd.DataFrame(vector+0)
    # print(padded_vector)
    if np.sum(padded_vector)[0]>0 and len(consec_repeat_mat)>1: #Check if there was any posotive and if the
        for i in range(0, np.max(np.where(consec_repeat_mat.iloc[:,0]==1))): #len(consec_repeat_mat)-2):
            if consec_repeat_mat.iloc[i,0]==change_to and consec_repeat_mat.iloc[i,1]>=min_bout and consec_repeat_mat.iloc[i + 1,1]<=max_gap and consec_repeat_mat.iloc[i+2,1]>=min_after_gap:
               padded_vector.iloc[consec_repeat_mat.iloc[i,2]:(consec_repeat_mat.iloc[i,2]+consec_repeat_mat.iloc[i,1]+consec_repeat_mat.iloc[i+1,1]),0] = change_to
    return pd.DataFrame(padded_vector+0)

def BAREfoot_Filter(y_to_filt, polish_repeat=2, min_bout=3, min_after_bout=1, max_gap=2, min_after_gap=1):
    y_filt = vector_polish(y_to_filt, cat_to_change=1, change_to=0, min_bout=min_bout, min_after_bout=min_after_bout, repeat=polish_repeat)
    y_filt = vector_bridge(y_filt, cat_to_change=0, change_to=1, min_bout=min_bout, max_gap=max_gap, min_after_gap=min_after_gap)
    return y_filt
