In this multiclass sentiment analysis project, I used pre-trained BERT for it with 2 epochs, and it got to 90& accuracy only on 2nd epochs.

Files Guidance:
1. requirements.txt (package and library version that needed to be use for this project, remember to add streamlit)
2. Twitter_Tweets_BERT.ipynb (Whole coding part of the project is inside of this file, from preprocessing to evaluation)
3. twitter-prediction-check.py (Just to confirm whether running .py from laptop side is working or not while importing model's file)
4. app.py (Host the model on a website with GUI, let others to make prediction on it)

Extra NOTES:
There is a '!pip freeze > requirements.txt' at the end of my .ipynb file, that is to get all of the package/ library version from the 'Google Colab' I using for this project. Reason is that I am going to directly get the necessary and same version of the package/ library to run my prediction and app.py on my own local laptop. To do this, the step is at below:
1. Run the '!pip freeze > requirements.txt', it will generate all the package/ library that is needed from the current env that u are using.
2. Run 'twitter-tweets-package.ipynb' to get the ONLY necessary package for the project (since 'Google Colab' have a lot library/package). Then run another code will compare the 'requirements.txt' with the ONLY requirements.txt, then generate a final 'requirements.txt' which is included the ONLY necessary package with version inside it.
3. Create a new env in conda (since im using anaconda vs code) by typing 'conda create -n twitter-multiclass-sentiment python=3.10 -y', this will generate a specified name env with python 3.10 version
4. Activate the new env, and tytpe 'pip install -r final_requirements.txt' to install all ONLY necessary package with specified version into the new env.
5. Since the final requirements.txt did not include pytorch etc, remember to type 'pip install torch torchvision torchaudio', 'pip install scikit-learn', 'pip install streamlit' into your new env.
6. Verify your installation by typing 'python -c "import transformers; import torch; import seaborn; import pandas; import sklearn; print('Environment is ready!')" (If all is done, then it will print out 'Environment is ready!', else check which package/library you are missing and manually install it)
7. All is done, run 'twitter-prediction-check.py' and 'app.py' to test the model out.

