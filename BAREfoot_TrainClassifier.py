"""
Behavior Classifier Training Script

This script trains a machine learning model to classify specific animal behaviors
(e.g., rearing, flicking, scratching) based on features extracted from pose estimation
and pixel brightness analysis of video data. The pipeline includes feature extraction,
dataset preprocessing, model training, and evaluation.

Usage:
1. Set up a project folder with the following subfolders:
   - [project_folder]/Videos: Contains video files and corresponding DeepLabCut (DLC) data (.h5 files).
   - [project_folder]/Targets: Contains target behavior labels (.csv files).
2. Configure the `project_folder` and other parameters in the USER DEFINED FIELDS section.
3. Run the script to train the classifier and validate its performance.

Requirements:
- Python environment with required dependencies (e.g., XGBoost, sklearn, matplotlib).
- Ensure DeepLabCut files and target label files are correctly paired.

Output:
- Trained classifier saved in the specified output path.
- Performance metrics and evaluation plots for the test dataset.

"""

import matplotlib.pyplot as plt
from AniML_utils_GeneralFunctions import *
import matplotlib
matplotlib.use('Qt5Agg')
plt.ion()
import pandas as pd
from AniML_utils_PoseFeatureExtraction import *
from AniML_utils_PixBrightnessFeatureExtraction import *
from AniML_utils_PreprocessingDataset import *
from AniML_utils_LearningCurve import *
from AniML_VideoLabel import *
from BAREfoot_utils_Predict import *
from AniML_utils_Publishing import *
import glob
from AniML_utils_LearningCurve import *
from BAREfoot_utils_Filter import *

import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, r2_score, confusion_matrix, ConfusionMatrixDisplay
import shap
import matplotlib.colors as mcolors


#%% USER DEFINED FIELDS
#######################################################################################################

#### New classifier training specs
#### Acute pain (used in Barkai et al.)
# Behavior_type = ['Flinching']; min_bout, min_after_bout, max_gap = 4,1,2; thresh_tuned=0.5
# Behavior_type = ['LickingBiting']; min_bout, min_after_bout, max_gap = 5,1,2; thresh_tuned=0.5
# Behavior_type = ['Grooming']; min_bout, min_after_bout, max_gap = 1,1,5; thresh_tuned=0.4

Behavior_type = ['NewBehavior'];
min_bout, min_after_bout, max_gap = 1,1,5;
test_set = [9, 11, 13, 20] # files to leave out as test set from the video folder.

# Parameters
bp_include_list = None # Choose which keypoint body parts to include in pose features(None=All)
bp_pixbrt_list = ['hrpaw', 'hlpaw', 'snout'] # # Choose which keypoint body parts to include in pixelBrightness features
pix_threshold = 0.3 # Threshold of birghtness: <-1 is by precentage (e.g 0.3 for 30%); >1 for 1-to256 pixel intensity
square_size = [40, 40, 40] # square sizes for Brightness analysis

Rig='BlackBox'
project_folder = r'Classifier Training Sets/Rearing'


# USER OPTIONAL FIELDS
###################################################################################################
classifier_library_path = rf'BAREfoot_Classifiers\{Rig}/'
os.makedirs(classifier_library_path, exist_ok=True)
# Choose train/test dataset source folders:
train_pose_folder = project_folder+r'\Videos'
train_video_folder = project_folder+r'\Videos'
train_target_folder = project_folder+r'\Targets'

test_video_folder = train_video_folder
test_pose_folder = train_video_folder
test_target_folder = train_target_folder

# Choose video output folder:
video_output_folder = project_folder+r'\VideosScored/'

Behavior_join= ''.join(Behavior_type[:]) # Single string in case of multi behvaiors
classifier_name = 'BAREfoot_' + Behavior_join

# Display setting
Beh_color = 'darkred'
Beh_cmap = custom_cmap(f'{Beh_color}s_map', from_color='white', to_color=Beh_color)

# Which classifiers do you want to train (names should match header is target files)
# Behavior and filtering parameters from Barkai et al. (Uncomment the behavior wanted)

#RUN! Train Classifier
########################################################################################################################
#%% 1.1 Load training files and extract features: DLCs, videos
#load train file list
train_file_list = glob.glob(train_pose_folder + '/*.h5')
train_file_list = [os.path.basename(file) for file in train_file_list]  # for list of sessionfiles without folder
#leave out files for test
data_set=range(0,len(train_file_list))
train_set = np.setdiff1d(data_set, test_set)

X,y = pd.DataFrame(), pd.DataFrame()
PoseFeatures = pd.DataFrame()
PixBrightFeatures = pd.DataFrame()
temp_X = pd.DataFrame()
X_prob = pd.DataFrame()

for i_count,i_file in enumerate(train_set):  # len(train_file_list)): worked well with - train:(0,7) test:(8,10)
    print(f"Train file {i_count + 1} / {len(train_set)}")
    target_file=glob.glob(f"{train_target_folder}/{train_file_list[i_file].split('DLC')[0]+ '.csv'}")[0]
    train_pose_file = train_pose_folder + '/' + train_file_list[i_file]
    vid_file = os.path.basename(train_pose_file).split("DLC", 1)[0] + '.avi'
    temp_X =BAREfoot_ExtractFeatures(pose_data_file=train_pose_file,
                          video_file_path=train_video_folder + '/' + vid_file,
                          bp_pixbrt_list=bp_pixbrt_list,
                          square_size=square_size,
                          pix_threshold=pix_threshold)
    X_headers = temp_X.columns
    # Drop NaN rows from feature extraction
    rows_with_nan = temp_X[temp_X.isna().any(axis=1)].index
    temp_X = temp_X.drop(index=rows_with_nan)
    temp_y = pd.read_csv(target_file)
    temp_y = temp_y.drop(index=rows_with_nan)

    X = pd.concat([X, temp_X])
    y = pd.concat([y, pd.DataFrame(temp_y.astype(int))])
    print(train_pose_file)
    print('Done')
X = X.reset_index(drop=True)  # drop=True for dropping the 'index' column created with reset_index
X_copy = X
y_copy = y

#%% 1.2 Set classifier parameters and train model
#Preprocess - balance data
X = X_copy.reset_index(drop=True)
y = y_copy.reset_index(drop=True)
y = np.array(y[Behavior_type].sum(axis=1)>0).ravel()
zeros_to_ones = (np.sum(y == 0) / len(y) + 0.5) / (np.sum(y == 1) / len(y) + 0.5) #Chicco et al. BioDataMining, BMC 2017
X, y = BalanceXy(X, y,'Downsample',zeros_to_ones)

#Model Parameters
clf_parameters = {
    'n_estimators': 1700,
    'objective': 'reg:squaredlogerror',
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,  # how many rows are used to train each tree
    'colsample_bytree': 0.2,  # how many features are used to train each tree
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'seed': 42,
    'alpha': 1,  # 2.75, #2.2,
    'lambda': 0.1,  # 0.2, #2
}
clf_model = xgb.XGBClassifier(**clf_parameters, n_jobs=-42, scale_pos_weight=(len(y) - np.sum(y)) / np.sum(y))

#Build model
print(f"Building {Behavior_type} model...")
tic()
clf_model.fit(X, y)
toc()

# 1.3 Set threshold
if 'thresh_tuned' in globals():
    best_thresh=thresh_tuned
    print (f'Threshold was tuned to : {thresh_tuned}')
    # del thresh_tuned
else:
    print("Finding prediction threshold with cross-validation. ")
    best_thresh = np.mean(AniML_FindThresh(X, y, clf_model, k=5, min_thr=0.3, max_thr=0.7,coarse_increment=0.02,
                                           fine_increment=0.025, search_radius=0.025, n_jobs=42))
    print(f'Discrimination threshold set to:{best_thresh}.'
          f'\nTo change, manually set best_thresh value (e.g. best_thresh=0.5)')


########################################################################################################################
### Validate Classifier
#%% 2.1 Load test file and extract features: DLCs, videos
test_file_list = glob.glob(test_pose_folder + '/*.h5')
test_file_list = [os.path.basename(file) for file in test_file_list]  # for list of sessionfiles without folder
X_test, y_test = pd.DataFrame(), pd.DataFrame()
y_test_starts=[0]
for i_count, i_file in enumerate(test_set):
    print(f"Test file {i_count + 1} / {len(test_set)}")
    test_pose_file = test_pose_folder + '/' + test_file_list[i_file]
    vid_file = os.path.basename(test_pose_file).split("DLC", 1)[0] + '.avi'
    print(test_pose_file)
    temp_X =BAREfoot_ExtractFeatures(pose_data_file=test_pose_file,
                          video_file_path = test_video_folder + '/' + vid_file,
                          bp_pixbrt_list=bp_pixbrt_list,
                          square_size=square_size,
                          pix_threshold=pix_threshold)
    traget_file=glob.glob(f"{test_target_folder}/{train_file_list[i_file][0:train_file_list[i_file].find('body') + 4] + '*.csv'}")[0]
    temp_y = pd.read_csv(traget_file)
    y_test_starts.append(len(temp_X) + y_test_starts[-1]) #Keep record of beginnings of each test video

    X_test = pd.concat([X_test, temp_X])
    y_test = pd.concat([y_test, pd.DataFrame(temp_y)])
    y_test = y_test[Behavior_type].astype(int)
y_test_starts=[0] + y_test_starts[1:-1]
y_test_ends = y_test_starts[1:] + [len(X_test)]
X_test = X_test.reset_index(drop=True)  # drop=True for dropping the 'index' column created with reset_index
X_test_copy=X_test
y_test_copy=y_test

X_test = X_test_copy.reset_index(drop=True)
y_test = y_test_copy.reset_index(drop=True)
y_test = np.array(y_test[Behavior_type].sum(axis=1) > 0).ravel()

#%% 2.2 plot F1-precision-recall vs. threshold for Test
y_test_all, y_filt_all = pd.DataFrame(), pd.DataFrame()
f1_scores_all, precision_scores_all, recall_scores_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
thresholds = np.concatenate([
                                 np.arange(0.00, 0.35, 0.05),
                                 np.arange(0.35, 0.725, 0.025),
                                 np.arange(0.7, 1.03, 0.05),
                                 [best_thresh]])
f1_scores = np.zeros(shape=np.shape(thresholds))
precision_scores = np.zeros(shape=np.shape(thresholds))
recall_scores = np.zeros(shape=np.shape(thresholds))

for i, th in enumerate(thresholds):
    y_filt_segments = []
    # Process thresholds for this segment
    for start, end in zip(y_test_starts, y_test_ends):
        X_test_segment = X_test.iloc[start:end]
        # Predict probabilities and apply threshold for the current segment
        y_pred_thr = clf_model.predict_proba(X_test_segment[clf_model.feature_names_in_])[:, 1] >= th
        # Filter (polish and bridge)
        y_filt_temp = BAREfoot_Filter(y_pred_thr, polish_repeat=2, min_bout=min_bout, min_after_bout=min_after_bout, max_gap=max_gap, min_after_gap=1)
        # Append the processed segment predictions to the list
        y_filt_segments.append(y_filt_temp)
    y_filt_th = np.concatenate(y_filt_segments).ravel()
    f1_scores[i], precision_scores[i], recall_scores[i] = round(f1_score(y_test, y_filt_th), 10), round(precision_score(y_test, y_filt_th), 5), round(recall_score(y_test, y_filt_th), 5)
    y_filt=y_filt_th #the last one was best threshold
y_filt_copy=y_filt
f1_scores_all=pd.concat([f1_scores_all, pd.Series(f1_scores)],axis=1)
recall_scores_all = pd.concat([recall_scores_all, pd.Series(recall_scores)],axis=1)
precision_scores_all = pd.concat([precision_scores_all, pd.Series(precision_scores)],axis=1)

print('Validation summary-----------------------------------------------------------------------')
print(f"{Behavior_type[:len(Behavior_type)]} |F1: {round(f1_score(y_test, y_filt), 3)} |Recall: {round(recall_score(y_test, y_filt), 3)} |Precision: {round(precision_score(y_test, y_filt), 3)}")
print(f'min_bout: {min_bout} | min_after_bout: {min_after_bout} | max_gap:{max_gap} | Threshold: {best_thresh} ')

plt.figure(figsize=(2, 2.7))
num_colors = 3
cmap = plt.get_cmap(Beh_cmap)
colors = cmap(np.linspace(.25, 1, num_colors))
plt.plot(thresholds[:-1], recall_scores[:-1], color=colors[0], label="Recall", linewidth=1)
plt.plot(thresholds[:-1], precision_scores[:-1], color=colors[1], label="Precision", linewidth=1)
plt.plot(thresholds[:-1], f1_scores[:-1], color=colors[2], label="F1 Score", linewidth=1)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.text(0,0.9,Behavior_join)
plt.xticks(ticks=np.arange(0, 1.05, 0.25))
plt.yticks(ticks=np.arange(0, 1.01, 0.1))
plt.xticks(rotation=45, ha='right')
plt.ylim(-0.02, 1.002)
plt.axvline(x=best_thresh, color='k', linestyle=':', alpha=0.5)
plt.legend(fontsize=8)
plt.tight_layout()
print(f'Threshold set to:{best_thresh}.\n'
      f'To change, manually set best_thresh value (e.g. best_thresh=0.5)')
plt.savefig(classifier_library_path +classifier_name + '_PerformanceThreshold' +  '.png', format='png', dpi=300)

#%% 2.3 Plot learning curve
learning_curve=True #set to false to avoid running automatically
if learning_curve:
    # %% Create learning curve
    CV = 5
    y_test = y_test_copy
    y_test = np.ravel(y_test[Behavior_type].sum(axis=1) > 0).astype(int)
    X_test = X_test_copy
    pstv_bouts_sizes = np.concatenate(([50, 500], np.arange(1500, np.sum(y), 1500)))
    train_sizes = np.array(pstv_bouts_sizes) * (zeros_to_ones + 1)
    train_sizes = train_sizes.astype(int)
    train_sizes_true, test_scores = AniML_learning_curve(clf_model, X, y, method='f1', train_sizes=train_sizes,
        best_thresh=best_thresh, cv=CV,
        X_val=X_test[clf_model.feature_names_in_], y_val=y_test)
    #Calculate mean+SEM of scores and plot
    test_scores_mean = np.mean(np.round(test_scores, 2), axis=1)
    test_scores_SEM = np.std(np.round(test_scores, 2), axis=1) / np.sqrt(len(test_scores))
    plt.figure(figsize=(2, 2.85))
    plt.plot(pstv_bouts_sizes, test_scores_mean, marker='.', linewidth=0.25, color=Beh_color, markersize=2)
    plt.fill_between(pstv_bouts_sizes, test_scores_mean - test_scores_SEM,
                     test_scores_mean + test_scores_SEM, alpha=0.25, color=Beh_color)
    plt.xlabel('Bout-positive frames')
    plt.ylabel(f'F1-score ({CV}-fold CV)')  # You can change to 'F1 Score' if needed
    plt.yticks(ticks=np.arange(0.00, 1.1, 0.1))
    plt.ylim(-0.02, 1.001)
    plt.legend([Behavior_join])
    plt.xticks(ticks=pstv_bouts_sizes, rotation=45, ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(classifier_library_path + classifier_name +'_LearningCurve'+  '.png', format='png', dpi=300, transparent=False)

#######################################################################################################################
#%% 3. Save classifier and SHAP values'''
save_model=True
if save_model:
    # %% Save clf_model to file
    import pickle
    model_data = {
        'clf_model': clf_model,
        'best_thresh': best_thresh,
        'Behavior_type': Behavior_type,
        'min_bout': min_bout,
        'min_after_bout': min_after_bout,
        'max_gap': max_gap,
        'bp_include_list': bp_include_list,
        'bp_pixbrt_list': bp_pixbrt_list,
        'pix_threshold': pix_threshold,
    }
    with open(classifier_library_path + classifier_name + '.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    '''4. Save Importances '''
    # SHAP importance
    explainer = shap.Explainer(clf_model)
    X_forSHAP = pd.concat([X[y == 1].sample(n=1000, random_state=42), X[y == 0].sample(n=1000, random_state=42)])
    y_forSHAP = [1] * 1000 + [0] * 1000
    shap_values = explainer(X_forSHAP)

    ## SHAP ##

    N_display = 10
    plt.figure()
    plt.subplot(1, 2, 1)
    shap_cmap = custom_cmap('somename', from_color='white', to_color='darkgoldenrod')
    # Create the SHAP summary plot without the default color bar
    shap.summary_plot(shap_values, X_forSHAP, cmap=truncate_colormap(shap_cmap, min_val=0.25, max_val=1),
                      max_display=N_display, plot_size=[6.5, 3], color_bar=False)
    # Add a horizontal color bar above the plot
    sm = plt.cm.ScalarMappable(cmap=truncate_colormap(shap_cmap, min_val=0.25, max_val=1))
    cbar = plt.colorbar(sm, orientation='vertical', shrink=0.3, anchor=(0.0,0.0))
    cbar.set_label('Feature Value')  # Optional: Remove label if not needed
    # Set custom tick labels for the color bar
    cbar.set_ticks([0.0, 1.0])  # Positions for 'Low' and 'High'
    cbar.set_ticklabels(['Low', 'High'])  # Custom labels
    # Format other plot elements
    plt.xlabel('Feature impact (SHAP value)')
    plt.gca().tick_params(labelsize=8)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    Imp_cmap = custom_cmap('somename', from_color='white', to_color='darkgoldenrod')
    # plt.figure(figsize=(3.5, 2.67))
    sorted_idx = np.argsort(np.abs(shap_values.values).mean(0))[::-1]
    top_N_idx = sorted_idx[:N_display]  # Select indices of the top 10 features
    shap_values_topN = shap.Explanation(
        values=shap_values.values[:, top_N_idx],
        base_values=shap_values.base_values,
        data=shap_values.data[:, top_N_idx],
        feature_names=[shap_values.feature_names[i] for i in top_N_idx]
    )
    plt.xticks(rotation = 0, ha='right')
    for i, (category, value) in enumerate(zip(shap_values_topN.feature_names[::-1], np.mean(np.abs(shap_values_topN.values), axis=0)[::-1])): # Plot horizontal lines with varying shades of blue
        if category in [col for col in X.columns if 'Pix' in col]:
            # plt.hlines(category, xmin=0, xmax=value, color=Beh_cmap(np.linspace(0.2, 0.8, N_display))[i], linewidth=6)
            plt.hlines(category, xmin=0, xmax=value, color=plt.cm.Greys(np.linspace(0.5, 1, N_display))[i], linewidth=6)
            plt.plot(-0.0015, category, '<', color='lightcoral', markersize=5)
        if category in [col for col in X.columns if 'Pix' not in col]:
            # plt.hlines(category, xmin=0, xmax=value, color=Beh_cmap(np.linspace(0.2, 0.8, N_display))[i], linewidth=6)
            plt.hlines(category, xmin=0, xmax=value, color=plt.cm.Greys(np.linspace(0.2, 0.8, N_display))[i], linewidth=6)
    plt.gca().tick_params(labelsize=10, size=3)
    plt.xticks(rotation=0, ha='center')  # Center the x-axis labels
    plt.xlabel('Importance (mean(|SHAP value|))')
    plt.ylim([-1, N_display])
    plt.tick_params(axis='y', labelleft=False)
    plt.legend([Behavior_join])
    plt.tight_layout()

    plt.savefig(classifier_library_path  + classifier_name+'_SHAP_Importance' + '.png', format='png', dpi=300, transparent=False)


# %% 4. Save training and test data to a pickle file
    import pickle

    with open(f'{classifier_library_path}/{Behavior_type[0]}_train_test_set.pkl', 'wb') as f:
        train_test_data = {
            'X': X_copy, 'y': y_copy, 'X_test': X_test_copy, 'y_test': y_test_copy, 'test_file_list': test_file_list,
            'train_file_list': train_file_list
        }
        pickle.dump(train_test_data, f)

# %% 5. Save training and test data to a pickle file
# Creat video
        file_to_test = 0
        test_pose_file = train_file_list[test_set[file_to_test]]
        VideoName = test_pose_file.split('DLC')[0]
        print("Writing video...")
        LabelVideo(VideoName=VideoName,
                   Folder=test_video_folder,
                   OutputFolder=video_output_folder,
                   BehaviorLabels=y_test_copy[y_test_starts[file_to_test]:y_test_ends[file_to_test]],
                   InputFileType='.avi',
                   OutputFileType='.avi',
                   FrameCount=True,
                   Resolution_factor=0.33,
                   fromFrame=0,
                   # toFrame=1000,
                   inIncrement=1,
                   pix_threshold=pix_threshold * 256,
                   only_pix_threshold=False,
                   colormap='coolwarm',
                   LabelType='Text',  # 'Number'
                   plot=False
                   )
