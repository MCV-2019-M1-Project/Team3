#!/bin/bash

workdir=$PWD
echo $workdir

cd ..
mkdir -p dataset

cd dataset

datadir=$PWD
echo $datadir
wget --header="Host: e-aules.uab.cat" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" --header="Accept-Language: en-GB,en;q=0.9" --header="Referer: https://e-aules.uab.cat/2019-20/mod/page/view.php?id=101861" --header="Cookie: MoodleSessioneaulesprod201920=614v717heinb0l63lc75plf853; messageCookie=true; BALANCEIDEA19=balancer.lbd1906; UqZBpD3n3iPIDwJU9DqQmVKiaLc0wapuQ+ac6w@@=v1mHtDFwSD8SU" --header="Connection: keep-alive" "https://e-aules.uab.cat/2019-20/pluginfile.php/368994/mod_page/content/126/bbdd.zip" -O "bbdd.zip" -c; unzip bbdd.zip -d train;
wget --header="Host: e-aules.uab.cat" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" --header="Accept-Language: en-GB,en;q=0.9" --header="Referer: https://e-aules.uab.cat/2019-20/mod/page/view.php?id=101861" --header="Cookie: MoodleSessioneaulesprod201920=614v717heinb0l63lc75plf853; messageCookie=true; BALANCEIDEA19=balancer.lbd1906; UqZBpD3n3iPIDwJU9DqQmVKiaLc0wapuQ+ac6w@@=v1mHtDFwSD8SU" --header="Connection: keep-alive" "https://e-aules.uab.cat/2019-20/pluginfile.php/368994/mod_page/content/126/qsd1_w2.zip" -O "qsd1_w2.zip" -c; unzip qsd1_w2.zip; mv qsd1_w2 val1
wget --header="Host: e-aules.uab.cat" --header="User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3" --header="Accept-Language: en-GB,en;q=0.9" --header="Referer: https://e-aules.uab.cat/2019-20/mod/page/view.php?id=101861" --header="Cookie: MoodleSessioneaulesprod201920=614v717heinb0l63lc75plf853; messageCookie=true; BALANCEIDEA19=balancer.lbd1906; UqZBpD3n3iPIDwJU9DqQmVKiaLc0wapuQ+ac6w@@=v1mHtDFwSD8SU" --header="Connection: keep-alive" "https://e-aules.uab.cat/2019-20/pluginfile.php/368994/mod_page/content/126/qsd2_w2.zip" -O "qsd2_w2.zip" -c; unzip qsd2_w2.zip; mv qsd2_w2 val2

rm *.zip

cd $workdir
mkdir -p data/dataset

ln -s $datadir/train data/dataset
ln -s $datadir/val1 data/dataset
ln -s $datadir/val2 data/dataset
