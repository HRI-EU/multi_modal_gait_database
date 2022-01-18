DATA_SET_PATH=/home/vlosing/storage/phume/gait_data_set
DESTINATION_PATH=/home/vlosing/storage/phume/gait_data_set/test.pkl
PYTHONPATH=$(pwd) python src/data_frame_extractor.py -s 1,2 -c a,b -p $DATA_SET_PATH -d $DESTINATION_PATH