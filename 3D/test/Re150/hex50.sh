RUN_DIR=`date +%F | sed 's/-//g'`-`date +%T | sed 's/://g'`

$CAVITY_DIR/bin/CAVITY --mesh $MESH_DIR/CUBE/Fluent/hex/50_translated.txt --Re 150 --time-span 15 --write-interval 200 --tag $RUN_DIR

cd $RUN_DIR

mkdir raw

mv *.txt raw/

TEC_DATA_CONVERTER --composition hex --input-dir raw --output-dir tec

