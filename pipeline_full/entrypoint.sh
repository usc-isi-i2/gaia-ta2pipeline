echo "start resolving named entities coref..."
/aida/ta2-pipeline/entrypoint.sh

echo "start resolving uke coref..."
python -m uke_coref --input $INPUT --output $OUTPUT --params /aida/uke-coref/params/$PARAMS --log DEBUG
