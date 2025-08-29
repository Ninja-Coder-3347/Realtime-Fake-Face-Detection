import os#provide interaaaction with file system
import random
import shutil#shutil used here to delete file from all over
from itertools import islice#divide data in val, train and test.

outputFolderPath = "Dataset/SplitData"#path where data is going to store
inputFolderPath = "Dataset/all"#location of wheere input is stored
#Defines the ratio for splitting the dataset into train, val, and test sets.
splitRatio = {"train":0.7,"val":0.2,"test":0.1}
classes = ["fake","real"]
#Randomly Sampling and delete previous sampling.
try:
    #The try block attempts to delete the output folder (Dataset/SplitData) if it exists,
    shutil.rmtree(outputFolderPath)#shutil used here to delete file from all over
#and the except block creates the folder if it doesn’t exist.
except OSError as e:
    os.mkdir(outputFolderPath)

# ---------Directories to create -----
#These lines create directories for storing the train, validation,
# and test images and their corresponding labels.
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
#The exist_ok=True flag ensures that the code doesn't raise
# an error if the directories already exist.
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)


#-------------get the name -------
# it list the all dataset in the inputfolder
listNames = os.listdir(inputFolderPath)
print(listNames)
uniqueNames = []
for name in listNames:
    #o remove the file extensions (e.g., .jpg or .txt).
    uniqueNames.append(name.split('.')[0])
uniqueNames=list(set(uniqueNames))
print(uniqueNames)

#----shuffle-----------
random.shuffle(uniqueNames)



#----------find the number of images for each folder------
lenData = len(uniqueNames)
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])
#length of the data which is divided into train,val,test


#-------------Put remaining images in Training ------
#This block ensures that the total number of images is exactly
# divided between the three sets. If there are any remaining images---->
# (due to rounding), they are added to the train set.
if lenData != lenTrain+lenTest+lenVal:
    remaining =lenData-(lenTrain+lenTest+lenVal)
    lenTrain += remaining



#--------split the list -------
#real splittiing is done here
lenthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lenthToSplit]
print(f'Total Images : {lenData}\nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')


#-----------copy the files -------
#inputfolder madhun data copy kela jato outputfolder madhye
#images madhye .jpg and labels madhye .txt
sequence = ['train','val','test']
for i,out in enumerate(Output):
    for filename in out:
        shutil.copy(f'{inputFolderPath}/{filename}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{filename}.jpg')
        shutil.copy(f'{inputFolderPath}/{filename}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{filename}.txt')

print("Split Process Completed........")

#-------- Creating Data.yaml file ---------
#file, which is used to store information
# about the dataset and paths for the training, validation,
# and test sets.
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

#data.yaml file will be created here
f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created")

