# python init.py

# check if kb is available
if [ ! -f "${REPO_KB}/data/entities.tab" ] || [ ! -f "${REPO_KB}/data/alternate_names.tab" ]; then
	echo "KB is not complete."
	exit 1
fi

INPUT_NIST="${INPUT}/${RUN_NAME}/NIST"
INPUT_INTER_TA="${INPUT}/${RUN_NAME}/INTER-TA"
TMP_DIR="${OUTPUT}/WORKING"
OUTPUT_NIST="${OUTPUT}/NIST"
OUTPUT_INTER_TA="${OUTPUT}/INTER-TA"
REPO_TA1_NIST="ta1-nist"
REPO_TA1_INTER_TA="ta1-inter-ta"
REPO_TA2_NIST="ta2-nist"
REPO_TA2_INTER_TA="ta2-inter-ta"

# =========================================
# NIST
rm -rf "$TMP_DIR" && mkdir "$TMP_DIR"

cd /aida/ta2-pipeline/graphdb
# create repos
./start.sh
./create.sh "${REPO_TA1_NIST}"
sleep 5s
./create.sh "${REPO_TA2_NIST}"
sleep 5s
./stop.sh

# import data
./import.sh "${INPUT_NIST}" "${REPO_TA1_NIST}"
sleep 5s
./import.sh "${INPUT_NIST}" "${REPO_TA2_NIST}"
sleep 5s

# run ta2 algorithm
./start.sh
cd ..

PARAMS_FILE=./docker-nist.params
cat <<EOT > $PARAMS_FILE
[DEFAULT]
endpoint=http://localhost:7200/repositories
wikidata_sparql_endpoint=https://dsbox02.isi.edu:8888/bigdata/namespace/wdq/sparql
kg_tab_dir_path=${REPO_KB}/data/
repo_src=${REPO_TA1_NIST}
repo_dst=${REPO_TA2_NIST}
graph=http://www.isi.edu/001
version=001
delete_existing_clusters=False
outdir=${TMP_DIR}
cluster_nb=er-rpi.ipynb
kernel_name=python3
EOT

python ta2_runner.py $PARAMS_FILE
cd graphdb
./stop.sh
rm $PARAMS_FILE

# export data
./export.sh "${OUTPUT_NIST}/export.ttl" "${REPO_TA2_NIST}"


# =========================================
# INTER-TA
rm -rf "$TMP_DIR" && mkdir "$TMP_DIR"

cd /aida/ta2-pipeline/graphdb
# create repos
./start.sh
./create.sh "${REPO_TA1_INTER_TA}"
sleep 5s
./create.sh "${REPO_TA2_INTER_TA}"
sleep 5s
./stop.sh

# import data
./import.sh "${INPUT_INTER_TA}" "${REPO_TA1_INTER_TA}"
sleep 5s
./import.sh "${INPUT_INTER_TA}" "${REPO_TA2_INTER_TA}"
sleep 5s

# run ta2 algorithm
./start.sh
cd ..

PARAMS_FILE=./docker-inter-ta.params
cat <<EOT > $PARAMS_FILE
[DEFAULT]
endpoint=http://localhost:7200/repositories
wikidata_sparql_endpoint=https://dsbox02.isi.edu:8888/bigdata/namespace/wdq/sparql
kg_tab_dir_path=${REPO_KB}/data/
repo_src=${REPO_TA1_INTER_TA}
repo_dst=${REPO_TA2_INTER_TA}
graph=http://www.isi.edu/001
version=001
delete_existing_clusters=False
outdir=${TMP_DIR}
cluster_nb=er-rpi.ipynb
kernel_name=python3
EOT

python ta2_runner.py $PARAMS_FILE
cd graphdb
./stop.sh
rm $PARAMS_FILE

# export data
./export.sh "${OUTPUT_INTER_TA}/export.ttl" "${REPO_TA2_INTER_TA}"
