cd /aida/ta2-pipeline/graphdb
# create repos
./start.sh
./create.sh ta2-test-ta1
sleep 5s
./create.sh ta2-test-ta2
sleep 5s
./stop.sh

# import data
./import.sh /input/for_ta2_pipeline_test.zip ta2-test-ta1
sleep 5s
./import.sh /input/for_ta2_pipeline_test.zip ta2-test-ta2
sleep 5s

# run ta2 algorithm
./start.sh
cd ..
python ta2_runner.py docker.params
cd graphdb
./stop.sh

# export data
./export.sh /output/export.ttl ta2-test-ta2
