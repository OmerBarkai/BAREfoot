"""
BAREfoot Automatic Behavior Scoring Script

This script is designed for automatic behavior scoring using the BAREfoot framework.
It integrates pose estimation, pixel brightness feature extraction, and behavior
classification through machine learning models.

Components Used:
- `AniML_VideoLabel`: Handles video labeling and visualization of behaviors.
- `BAREfoot_utils_Predict`: Runs the classification models to predict behaviors from extracted features.
- `AniML_utils_PoseFeatureExtraction`: Extracts pose-based features for classification.
- `AniML_utils_PixBrightnessFeatureExtraction`: Extracts pixel brightness features.

User-Defined Parameters:
- `Project`: Path to the main folder containing experimental data.
- `Experiments`: List of experimental folders to process. An experiment folder should contain all videos.
- `Behavior_classifiers`: List of pre-trained classifier files for behavior prediction.
- Feature extraction parameters such as `pix_threshold`, `bp_pixbrt_list`, and `square_size`.
- Video creation options such as resolution, FPS, and input/output formats.

Workflow:
1. Loop through experimental folders and identify videos for analysis.
2. Extract pose and pixel brightness features from video data.
3. Apply classifiers to predict behaviors and save results.
4. Compile and save summary statistics for each experiment.
5. (Optional) generate labeled videos for the detected behaviors.

Note:
- Ensure the required classifier files are available in the specified `ClassifierLibraryFolder`.
- Running DeepLabCut on the video is optional; uncomment and configure DLC lines if analysis is needed.

"""


from AniML_utils_GeneralFunctions import *
from AniML_VideoLabel import *
from BAREfoot_utils_Predict import *
from AniML_utils_PoseFeatureExtraction import *
from AniML_utils_PixBrightnessFeatureExtraction import *
import glob
tic()
#%% USER DEFINED FIELDS
#######################

create_videos=False # Slower;
Resolution_factor=0.4 # Users can adjust Resolution_factor parameter (0-1) to change speed.
runDLC=False # If videos were not DeepLabCutted then true

fps = 25
PoseDataFileType='h5'
#Crop video scoring (from frame to frame)
From,To=(0, None)
skip_processed_folders=True #redo analysis on already processed folder?

''' 1. Define behavior video folder '''
# Form: Experiment  -> Experiments (mouse) -> i_Trial(Before, drug, after, timepoints, etc.)
# Example: Project (DARPA) > Experiment (Capsaicin 0.1%, morphine, Date) > Subject Trial files (*.H5, *.avi)

Rig='DarkBottomUp' #Enter rig name

Project = r' ' #Enter project folder
Experiments =['   ','   '] #Enter experiemtns

pix_threshold=0.3
bp_pixbrt_list=['hrpaw', 'hlpaw','snout']
square_size=[40,40,40]


'''2. Define classifiers'''
ClassifierLibraryFolder = rf'\BAREfoot_Classifiers\{Rig}/'
Behavior_classifiers = [
                       'BAREfoot_Flinching.pkl',
                       'BAREfoot_LickingBiting.pkl',
                       'BAREfoot_Grooming.pkl',
                    ]

#%% RUN! Automatic Recognition of Behavior Enhance with Light
#############################################################
timestamp=datetime.now().strftime("%H%M")
'''Run by the list of experiments (Folders)'''
for Experiment in Experiments:
    closeall()
    Experiment_path = Project + f'{Experiment}/'
    OutputFolder = 'BAREfoot_output_filtered'
    OutputFolderPath = Experiment_path + OutputFolder
    videos_folder = Experiment_path + f'/videos/'

    # Check if videos exist in folder at all
    if os.path.exists(videos_folder)==False:
        print(f"{Experiment} - videos folder not found.")
        continue
    else:
        print(f"{Experiment} - analyzing video folder.")

    #Check if it was already analyzed
    if os.path.exists(OutputFolderPath) and skip_processed_folders:
        print(f"Skipping {Experiment} - already processed and skip_processed_folders is True.")
        continue
    else:
        print(f"{Experiment} - creating BAREfoot behavior folder")
    os.makedirs(OutputFolderPath, exist_ok=True)

    # DLC pose estimation labeling - DLC doesnt re-analyze if it recognized DLC files in the folder
    if runDLC:
        print(f'DeepLabCut will run only on files that were not analyzed by DLC....')
        # Insert DeepLabCut config file to 'config_path'
        import deeplabcut
        dlc_config_path=r'\config.yaml'
        deeplabcut.analyze_videos(dlc_config_path,videos_folder, save_as_csv=False, gputouse=0)

    # Experiment in Folders
    pose_file_list = glob.glob(videos_folder + '/*DLC*filtered.' + PoseDataFileType)
    pose_file_list = sorted([os.path.basename(file) for file in pose_file_list])  # for list of sessionfiles without folder
    pose_ID = sorted([os.path.basename(file).split('DLC')[0] for file in pose_file_list])
    data_summary=pd.DataFrame()
    data_subjects = pd.DataFrame(pose_ID, columns=[f'{Experiment}_Subject'])

    print(f'Experiment: {Experiment} - with {Behavior_classifiers}')
    for i_file in range(0, len(pose_file_list)):
        print(f'Preparing data and extracting features (X) for classification: {pose_file_list[i_file]}')
        pose_data_file = pose_file_list[i_file]
        vid_file = pose_data_file.split('DLC')[0] + '.avi'
        DataID = pose_data_file[0:pose_data_file.find('.')]
        if pose_data_file.find('DLC') > -1:
            DataID = DataID[:pose_data_file.find('DLC')]
        data_path = videos_folder + pose_data_file
        X = BAREfoot_ExtractFeatures(pose_data_file=data_path,
                                  video_file_path=videos_folder + '/' + vid_file,
                                  bp_pixbrt_list=bp_pixbrt_list,
                                  square_size=square_size,
                                  pix_threshold=pix_threshold)
        subject_summary = pd.DataFrame()

        '''Run the data with all the selected classifiers'''
        subject_y_pred_filts=pd.DataFrame()
        for Behavior_classifier in Behavior_classifiers:
            clf_model_path = ClassifierLibraryFolder + Behavior_classifier

            with open(clf_model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            Behavior_type = loaded_model['Behavior_type']
            Behavior_join= ''.join(Behavior_type[:])

            y_pred_filt, y_pred = BAREfoot_Predict(clf_model_path, X)

            y_pred=y_pred.iloc[From:To,:]
            y_pred_filt=y_pred_filt.iloc[From:To,:]

            ##### Save prediction to CSV file
            os.makedirs(OutputFolderPath + f'/{DataID}_Behavior', exist_ok=True)
            y_pred_filt.columns=[''.join(Behavior_join)]
            y_pred_filt.to_csv(OutputFolderPath + f'/{DataID}_Behavior/' + DataID + '_' + ''.join(Behavior_join) + '_ML.csv', index=False)
            subject_y_pred_filts = pd.concat([subject_y_pred_filts,y_pred_filt],axis=1)
            ##### 2.2 Save summary file
            total_behavior_time = pd.DataFrame([round(np.sum(np.array(y_pred_filt))/fps,2)], columns=[f'{Behavior_join} time (s)'])
            # mean_bout_duration = float(np.mean(np.transpose(find_consecutive_repeats(y_pred_filt)[np.where(find_consecutive_repeats(y_pred_filt)[:, 0] == True), 1])/fps))
            # new_col=pd.DataFrame([[total_behavior_time,mean_bout_duration]],
            #                      columns=[f'{Behavior_join} time (s)', f'{Behavior_join} bout mean (s)'])
            subject_summary =pd.concat([subject_summary, total_behavior_time],axis=1)

        ##### Create video (optional if 'create_videos==1')
        if create_videos:
                print("Writing video...")
                LabelVideo(VideoName=DataID,
                           Folder=videos_folder,
                           OutputFolder=OutputFolderPath + f'/{DataID}_Behavior/',
                           BehaviorLabels=subject_y_pred_filts,
                           InputFileType='.avi',
                           OutputFileType='.mp4',
                           FrameCount=True,
                           Resolution_factor=0.7,
                           pix_threshold=0.3 * 256,
                           )
        data_summary = pd.concat([data_summary, subject_summary], axis=0)
    data_summary = pd.concat([data_subjects, data_summary.reset_index(drop=True)], axis=1)
    data_summary.to_csv(OutputFolderPath + '/Behavior_summary_' + Experiment + '_' + timestamp + '.csv')
print('Done.')
toc()
# Done() #Play sounds when done.

