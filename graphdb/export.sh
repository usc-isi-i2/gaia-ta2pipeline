source env.sh

PARAMS_FILE=./export.params
cat <<EOT > $PARAMS_FILE
outputFile: ./${REPO_NAME}_exported.ttl
graphDbBaseDir: ${GRAPHDB_BASE}/data/
repositoryId: ${REPO_NAME}
EOT

${GRAPHDB_EXPORT_UTIL_PATH} ${PARAMS_FILE}

rm ${PARAMS_FILE}