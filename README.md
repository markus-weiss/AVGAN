# AVGAN
## Deep neural audiovisual network
 
 The ﬁnal AVGAN-architecture is shown in ﬁgure 12. To combine the sound and image input streams the concatenation was used,after the two transformed soundimages were concatenated,as well. After the sound and image concatenation the GAN network starts the training while the generator tries to generate better images and the discriminator judges the output. After the training the network has to do the reverse way to display deconvoluted images. The last step is de deconcatenation of the sound images and the retransformation into a .wav ﬁle, so sound can be played.

# 

## The presentation

https://github.com/markus-weiss/AVGAN/blob/master/Generative%20Modelle%20f%C3%BCr%20audio-visuelle%20Formen.pptx.pdf

## A GIF from the frist trainng with the dcgan

https://github.com/markus-weiss/AVGAN/blob/master/firstTrainWithDCGAN.gif

##
For more information read the paper: https://github.com/markus-weiss/AVGAN/blob/master/Forschungsprojekt%20(2).pdf
