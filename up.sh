#
# up.sh
# 
cd ..

a="2019-ncov-japan covid-19 covid19 covid-19-data covid-19-data"
for x in $a ; do
  cd $x
  git pull
  cd ..
done

b=fukuoka/covid19
for x in $b ; do
  cd $x
  git pull
  cd ../..
done

# curl -O https://dl.dropboxusercontent.com/s/6mztoeb6xf78g5w/COVID-19.csv
curl.exe -o jag-covid.csv https://dl.dropboxusercontent.com/s/6mztoeb6xf78g5w/COVID-19.csv

