RUN_DIR=`date +%F | sed 's/-//g'`-`date +%T | sed 's/://g'`

$CAVITY_DIR/bin/CAVITY --mesh $MESH_DIR/CUBE/Fluent/hex/90_translated.txt --Re 400 --time-span 40 --write-interval 200 --tag $RUN_DIR

cd $RUN_DIR

mkdir raw

mv *.txt raw/

TEC_DATA_CONVERTER --composition hex --input-dir raw --output-dir tec

