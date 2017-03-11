# deepcolor: Automatic coloring and shading of manga-style lineart, using Tensorflow + cGANs

![]()

![]()

Setup:
```
0. Have tensorflow + OpenCV installed.
1. make a folder called "results"
2. make a folder called "imgs"
3. Fill the "imgs" folder with your own .jpg images, or run "download_images.py" to download from Safebooru.
4. Run "python main.py train". I trained for ~20 epochs, taking about 16 hours on one GPU.
5. To sample, run "python main.py sample"
6. To start the server, run "python server.py". It will host on port 8000.
```

Read the writeup:
http://kvfrans.com/coloring-and-shading-line-art-automatically-through-conditional-gans/

Try the demo:
http://color.kvfrans.com

Code based off [this pix2pix implementation](https://github.com/yenchenlin/pix2pix-tensorflow).


------------
Reference from the [deep color](https://github.com/kvfrans/deepcolor)
