RUN_DIR=`date +%F | sed 's/-//g'`-`date +%T | sed 's/://g'`

$CAVITY_DIR/bin/CAVITY --mesh $MESH_DIR/CUBE/Fluent/hex/80_translated.txt --Re 300 --time-span 25 --write-interval 200 --tag $RUN_DIR

cd $RUN_DIR

mkdir raw

mv *.txt raw/

TEC_DATA_CONVERTER --composition hex --input-dir raw --output-dir tec

