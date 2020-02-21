source env.sh

pid=$(cat ${GRAPHDB_PID_FILE})

while ps -p $pid > /dev/null
do
    kill $pid
    sleep 5
done

rm ${GRAPHDB_PID_FILE}