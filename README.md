# FACE-RFCN-Project
Face-RFCN implementation, on top of RFCN code by  https://github.com/parap1uie-s/Keras-RFCN

##Steps to execute the project:
1)  Change the path of WIDER FACE dataset files in the .py files
    1.1) set the variable name  (job_dir) in the FaceR_FCN_Train.py file
    1.2) set the variable names ( file, file1, file2 and file3) to point to correct folder names
2) To Run and Train the model, execute the following command:
    python3 FaceR_FCN_Train.py

3) To Test the model, Change the path stored in the variable named 'modelPath', and then execute this command
  python3 FaceR_FCN_Test.py
