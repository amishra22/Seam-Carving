Points to note:

1) You should have the mentioned libraries installed viz. matplotlib, numpy, and skimage
2) Initially I am removing one seam. 
3) In line number 143, I gave a input parameter as 50 which means that I am removing a total of 50 seams. We can change the parameter depending on the seams one might have to remove
4) In this algorithm we are only removing the vertical seams. Based on the same approach we can remove the horizontal seam. [A simple approach might be to tilt the image by 90 degree clockwise and then remove vertical seam and then again titl 90 degree anticlockwise.
5) I gave the input image as "test.png" whose output iamge is "carved.png". 

NOTE: The input image file should be in the same directory of the python file.