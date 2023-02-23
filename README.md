# Completed test assignment for Fulhaus
Image classification dockerized model

## Description 
Description here 

## Instructions and progress
1. "Build a classification model (Deep learning preferred)."

I check the data and train the model in "Train_and_save_classification_CNN_model" notebook. To develop the model I  used
PyTorch framework and CNN neural network architecture. I save trained model into "train_model_state_dict.pt" file.

2. "Build an API to access the model (Preferred frameworks based on Python)."

I choose Streamlit library for API development. The application was developed and saved into "app.py" file. In this file
I load pre-trained model architecture from "nn_arch.py" and load model parameters from "train_model_state_dict.pt" file 
(this is necessary step when working with PyTorch).

3. "Create a Docker image of your code by following docker best practices."
I save all necessary requirements into "requirements.txt" file and create a "Dockerfile" where I put base docker image
name, copy and run commands.
   
4. "Implement CI/CD pipeline on Github Actions."
This step is in progress
   
5. "Add a clear README file with instructions."
README file is ready
   

## How to Install and Run the Project
To run web page with image classification you need: 
1. Change directory to folder with the project
2. Start Docker
3. Run 2 commands in terminal:
   
    3.1 To build the image: 
    
     ```docker build -t app .``` 
   
    3.2 To run the image: 
   
   ```docker run -p 8501:8501 app```

## How to Use the Project
If the image is running successfully the next message will appear:

```You can now view your Streamlit app in your browser.```
```URL: http://0.0.0.0:8501```

You need to open your browser and past the URL in it.

Then you need: 
1. Wait for the website to load
2. Drag and drop or load the picture of bed/chair/sofa 
3. Press "Detect image" button 
4. Check the result 


## ToDo
1. Apply augmentation techniques 
2. Apply transfer learning techniques
3. Write tests 
4. Wrap code into functions/classes/modules