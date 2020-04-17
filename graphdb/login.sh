source env.sh

curl -X POST -I "${ENDPOINT}/rest/login/admin" -H "X-GraphDB-Password: ${GRAPHDB_ADMIN_PASSWD}" | grep Authorization > admin.auth
