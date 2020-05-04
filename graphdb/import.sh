source env.sh

FILE="$1"
REPO_NAME="$2"

# cp $FILE ${GRAPHDB_BASE}/data/
${GRAPHDB_BASE}/bin/preload -f -i $REPO_NAME $FILE

