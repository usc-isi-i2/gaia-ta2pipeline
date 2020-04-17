source env.sh

${GRAPHDB_BASE}/bin/graphdb -d -p ${GRAPHDB_PID_FILE}

# wait until endpoint accessible
# graphdb returns 404 on homepage if request is from curl
while [[ "$(curl -s -o /dev/null -w ''%{http_code}'' ${ENDPOINT})" != "404" ]]; do sleep 5; done
