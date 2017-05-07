## how to build a dl model to solve the zhihu verification recognition

**issue 1**

How to gen the random flip text image, it seems that `draw.text(pos, txt, font=self.font, fill=fill)` and the `rotate` are not useful.
Google whether there is a solution to gen the random flip text in a image.
Or I can concat the images (random flip text images) to gen a new Image.