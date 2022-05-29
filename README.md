# MLHeroku
Machine Learning Heroku Repo

## Steps to Seting up your app: 

- Create a github repo
- Clone your repo onto your local machine
- Build / Train your model
- Pickle the model / save it to a file
- Create a new anaconda environment
  - conda create --name **new environment name**
  - conda activate **new environment name**
- pip install your requirements the new environment
- build your flask application
  - build the html form
  - build your inputs 
  - read inputs into your route 
  - run the inputs through your machine learning model
- Set up for Heroku
  - create requirements.txt
    - pip freeze > requirements.txt
  - create Procfile
    - web gunicorn app:app
- Build Heroku site
  -  heroku create
  -  git add / commit 
  -  git push heroku main 
  -  heroku ps:scale web=1
  
