#
# jag1.sh
# 
cd data
mv COVID-19.CSV COVID-19.CSV.3
curl -o jag-covid.csv https://dl.dropboxusercontent.com/s/6mztoeb6xf78g5w/COVID-19.csv
cd ..

