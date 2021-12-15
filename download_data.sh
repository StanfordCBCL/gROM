#!/bash/bin

curl -L https://www.dropbox.com/s/6o8i37kprxu24yb/vtps.zip?dl=1 > vtps.zip
unzip -o vtps.zip
rm vtps.zip
rm -r __MACOSX
mv vtps graphs/
