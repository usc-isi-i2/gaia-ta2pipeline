source env.sh

FILE="$1"
REPO_NAME="$2"

PARAMS_FILE=./export.params
cat <<EOT > $PARAMS_FILE
outputFile: ${FILE}
graphDbBaseDir: ${GRAPHDB_BASE}/data/
repositoryId: ${REPO_NAME}
EOT

${GRAPHDB_EXPORT_UTIL_PATH} ${PARAMS_FILE}

rm ${PARAMS_FILE}