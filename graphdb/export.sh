source env.sh

FILE="$0"
REPO_NAME="$1"

PARAMS_FILE=./export.params
cat <<EOT > $PARAMS_FILE
outputFile: ${FILE}
graphDbBaseDir: ${GRAPHDB_BASE}/data/
repositoryId: ${REPO_NAME}
EOT

${GRAPHDB_EXPORT_UTIL_PATH} ${PARAMS_FILE}

rm ${PARAMS_FILE}