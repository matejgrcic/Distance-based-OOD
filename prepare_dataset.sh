mkdir Lost_and_found
cd Lost_and_found

echo '> Downloading files..'
wget http://www.dhbw-stuttgart.de/%7Esgehrig/lostAndFoundDataset/gtCoarse.zip
wget http://www.dhbw-stuttgart.de/%7Esgehrig/lostAndFoundDataset/leftImg8bit.zip
wget http://www.dhbw-stuttgart.de/%7Esgehrig/lostAndFoundDataset/disparity.zip
wget http://www.dhbw-stuttgart.de/%7Esgehrig/lostAndFoundDataset/camera.zip

echo '> Extracting data..'
unzip gtCoarse.zip
unzip leftImg8bit.zip
unzip disparity.zip
unzip camera.zip

rm gtCoarse.zip
rm leftImg8bit.zip
rm disparity.zip
rm camera.zip

echo '> Dataset prepared.'
