source env.sh

REPO_NAME="$0"

# create repo
CONFIG_FILE=./config.ttl
cat <<EOT > $CONFIG_FILE
#
# Configuration template for a GraphDB-Free repository
#
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>.
@prefix rep: <http://www.openrdf.org/config/repository#>.
@prefix sr: <http://www.openrdf.org/config/repository/sail#>.
@prefix sail: <http://www.openrdf.org/config/sail#>.
@prefix owlim: <http://www.ontotext.com/trree/owlim#>.

[] a rep:Repository ;
    rep:repositoryID "$REPO_NAME" ;
    rdfs:label "Created for GAIA" ;
    rep:repositoryImpl [
        rep:repositoryType "graphdb:FreeSailRepository" ;
        sr:sailImpl [
            sail:sailType "graphdb:FreeSail" ;

                # ruleset to use
                owlim:ruleset "empty" ;

                # disable context index(because my data do not uses contexts)
                owlim:enable-context-index "false" ;

                # indexes to speed up the read queries
                owlim:enablePredicateList "true" ;
                owlim:enable-literal-index "true" ;
                owlim:in-memory-literal-properties "true" ;
        ]
    ].
EOT
curl -H "${AUTH}" -X POST --header "Content-Type:multipart/form-data" -F "config=@./config.ttl" "${ENDPOINT}/rest/repositories"
rm -f $CONFIG_FILE

# grant access
PAYLOAD=$(curl -H "${AUTH}" -s -X GET "${ENDPOINT}/rest/security/freeaccess" | jq '.authorities += ["WRITE_REPO_'${REPO_NAME}'","READ_REPO_'${REPO_NAME}'"]')
curl -H "${AUTH}" -H "Content-Type: application/json" -X POST -d "$PAYLOAD" "${ENDPOINT}/rest/security/freeaccess"
