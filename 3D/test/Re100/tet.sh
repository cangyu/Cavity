RUN_DIR=`date +%F | sed 's/-//g'`-`date +%T | sed 's/://g'`

$CAVITY_DIR/bin/CAVITY --mesh 75470_translated.txt --Re 100 --time-span 10 --write-interval 100 --tag $RUN_DIR

cd $RUN_DIR

mkdir raw

mv *.txt raw/

TEC_DATA_CONVERTER --composition tet --input-dir raw --output-dir tec

