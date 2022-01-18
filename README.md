# NEWBEE data 

This is the repository to the article<br/>
V.Losing & M. Hasenjäger, _NEWBEE: A Multi-Modal Gait Database of Natural Everyday-Walk in an Urban Environment_, 2022. We provide some scripts to generate custom pandas data frames from the raw data to streamline the processing and machine learning. Additionally, there is also a visualization / labeling tool enabling data inspection and the modification of the labels.


## Setup
- Create a new conda environment<br/>
`conda create -n env_name python==3.9`

- Activate the environment<br/>
`conda activate env_name`

- Install required packages<br/>
`pip install -r requirements.txt`

- Install pyqt package using conda (using pip instead may lead to runtime errors)<br/>
`conda install -c anaconda pyqt `

## Generate pandas data frame from .csv files
We provide a script that generates one pandas data frame stored as pickle file from the single recording .csv files. This is quite handy for further processing or analysis.<br/>
`PYTHONPATH=$(pwd) python src/data_frame_extractor.py -s 1,2 -c a,b -p xxxx/NEWBEE_dataset/data_set/ -d destination_data_frame_path`
- -s 1,2 for subjects 1 and 2 (default is all subjects) 
- -c a,b for course A and B (default is all courses A,B,C)
- -p path to the data set
- -d path to the data frame stored as pickle file
(more parameters are available see `python src/data_frame_extractor.py --help`)

## Labeling tool
The tool allows to inspect the data but also to change the labels or use even custom labels.

It can be started by:<br/>
`PYTHONPATH=$(pwd) python src/labeling_tool.py -s 1 -c a -p xxxx/NEWBEE_dataset/data_set/`<br/>

- -s 1 for subject 1
- -c a for course A
- -p path to the data set


## Example calls and tests
- To run the test in src.test_extractor.py set the class variable SOURCE_PATH to the data set path
- To run the example calls set the variables DATA_SET_PATH, DESTINATION_PATH in the shell scripts according to your local directory structure