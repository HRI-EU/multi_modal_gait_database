# Multi Modal Gait Database

This is the repository to the article<br/>
V.Losing & M. Hasenjäger, _A Multi-Modal Gait Database of Natural Everyday-Walk in an Urban Environment_, 2022. We provide some scripts to generate custom pandas data frames from the raw data to streamline the processing and machine learning. Additionally, there is also a visualization / labeling tool enabling data inspection and the modification of the labels.

## Example videos
These videos provide an impression of the available data that is based on recordings using the XSens motion suit, insoles pressure sensors as well as an eye tracker.
- [course A](https://github.com/HRI-EU/multi_modal_gait_database/blob/master/videos/courseA_2.mp4)
- [course B](https://github.com/HRI-EU/multi_modal_gait_database/blob/master/videos/courseB_2.mp4)
- [course C](https://github.com/HRI-EU/multi_modal_gait_database/blob/master/videos/courseC_2.mp4)

## Setup
- Download the data set
- Create a new virtual environment (we name it `my_venv`) and activate it
```shell
python3 -m venv ./my_venv
source ./my_venv/bin/activate
```

- Install requirements
```shell
pip install -r requirements.txt
```

## Generate pandas data frame from .csv files
We provide a script that generates one pandas data frame stored as pickle file from the single recording .csv files. This is quite handy for further processing or analysis.<br/>
`PYTHONPATH=$(pwd) python src/data_frame_extractor.py -s 1,2 -c a,b -p xxxx/NEWBEE_dataset/data_set/ -d destination_file_path`
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


