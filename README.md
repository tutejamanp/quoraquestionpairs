Quora Question Pair


Steps to be followed-
(Considering generating outputs using the pickle files)

1) Download the PickleFiles, InputFiles, LemmatizedFiles, FeatureEngineeringFiles folders from "https://drive.google.com/drive/folders/1p2aU2ThajuBGdB63sR19mvdzVUrBSmsN?usp=sharing" and replace it from their corresponding empty folders in the zip you extracted.
2) requirements.txt : The requirements.txt has all the necessary libraries and the version you may need to run the project.
3) TestingPickle.py : Run the file called PythonFiles/TestingPickle.py. This would created a pickled-output.csv in the OutputFiles Folder. 
4) PostProcessing.py : Following this run the file called PythonFiles/PostProcessing.py. This creates the final file to be considered, submission.csv in the OutputFiles Folder. 
5) submission.csv : Final output file to be considered is submission.csv in the OutputFiles Folder.


In case you feel like running the entire project (PS. might take couple of hours or even more)
Steps to be followed-

0) Download the PickleFiles, InputFiles, LemmatizedFiles, FeatureEngineeringFiles folders from "https://drive.google.com/drive/folders/1p2aU2ThajuBGdB63sR19mvdzVUrBSmsN?usp=sharing" and replace it from their corresponding empty folders in the zip you extracted.
1) Lemmatization.py : We begin by lemmatizing all the questions in the given input files. Thus, run the PythonFiles/Lemmatization.py file. This usually takes a three hours but what we get is 2 files called trainlem.csv and testlem.csv in the LemmatizedFiles folder. These files contain all the corresponding lemmarized questions of the original questions.
2) IntersectionCount.py : Now we begin with the feature engineering, we run a file called IntersectionCount.py that adds a graph based feature in both trainlem and testlem.
3) CalculateTFIDF.py : CalculateTFIDF.py helps us add tf-idf values as a feature in both trainlem and testlem. 
4) QuestionFrequencies.py : Then comes other graph based feature that is incorporated via the file called Questionfrequencies.py it adds a couple of features called q1_freq and q2_freq to trainlem and testlem.
5) FeatureEngineering.py : All the other features are added by running this file. It creates two files called featured_train.csv and featured_test.csv in the folder called FeatureEngineeringFiles. These files are then used for all the further calculations.
6) DataExploration.py : Once all the features are ready, we deduce importance and co relation between them using plots and visualizations that are made possible by runniog the file called DataExploration.py
7) ModelExpeditions.py : After deducing the best models using various empirical experimentations(all the trial subjects can be found in InputFiles/check6.csv), we ran the best models using the file ModelExpeditions.csv. This create pickle files for all the models used which are stored in the folder called PickleFiles.
8) TestingPickle.py : The models stored in the pickle files then predict the pickled results stored in the file called pickled_output.csv.
9) PostProcessing.py : This takes pickled-output as an input and generates the final result to be considered using the called PostProcessing.py called submission.csv.
10) submission.csv : Final output file to be considered is submission.csv in the OutputFiles Folder.