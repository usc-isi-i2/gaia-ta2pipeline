source env.sh

FILE="$0"
REPO_NAME="$1"

# cp $FILE ${GRAPHDB_BASE}/data/
${GRAPHDB_BASE}/bin/preload -f -i $REPO_NAME $FILE

