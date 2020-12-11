# AMI Group 15 Machine Learning Model Pipeline and User Interface for Web Application

## Creating python environment & dependency installation
The codebase uses Python 3.8 and pip 20.0.2.

### Python virtual environment 


    pip3 install virtualenv
    virtualenv ami_project_group15
    source ami_project_group15/bin/activate    

### Installing libraries

    pip3 install -r requirements.txt
    pip3 install -e .

### Running the application
There are two pipelines currently. One for the time-series prediction of the individual features and another for the time-series prediction of the main model that would predict total carbon-dioxide emission of countries.

User can select the option to train or evaluate (on basis of saved model checkpoints) with the --train command line parameter with True or False. Secondly, user can select whether they want to execute the main model or the features as a whole with the --main_model parameter set to True or False.

Below is a sample command of how to run our pipeline application

    python3 src/Models/run.py --train False --main_model True 

### Saving model checkpoints and metrics

The pre-trained models are kept in /src/Models/model_checkpoints folder where there is a folder for each of the type of model used. There is also an excel sheet (contains separate sheets for main model and features model) within each model type which contains training and evaluation metrics (RMSE and r2 score).

# Web Application

### Installation
This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 10.0.6. which requires node.js 12.18.3. To setup node.js and angular execute the following commands.

    sudo apt update 

    sudo apt install nodejs 

    sudo npm install -g @angular/cli 

### Installing dependencies
Navigate to the folder predictor-app and execute

    npm install
    
### Build

Navigate to the predictor-app folder and run 

    ng build
    
to build the project. The build artifacts will be stored in the `dist/` directory. Use the `--prod` flag for a 
production build.


## Run the Web Application

Navigate to the predictor-app folder and run 

    ng serve
    
to start the client. Navigate to `http://localhost:4200/`


To start the python server run

    python3 src/server.py
    
