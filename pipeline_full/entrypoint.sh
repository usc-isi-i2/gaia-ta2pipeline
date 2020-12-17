echo "start resolving named entities coref..."
cd /aida/ta2-pipeline
./entrypoint.sh

echo "start resolving uke coref..."
cd /aida/uke-coref
python -m uke_coref --input "$INPUT/$RUN_NAME" --output "$OUTPUT/$RUN_NAME" --params /aida/uke-coref/params/$PARAMS --log DEBUG
