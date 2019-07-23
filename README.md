# AVGAN
 Deep neural audiovisual network
 
 The ﬁnal AVGAN-architecture is shown in ﬁgure 12. To combine the sound and image input streamstheconcatenationwasused,afterthetwo transformedsoundimageswereconcatenated,as well. After the sound and image concatenation the GAN network starts the training while the generatortriestogeneratebetterimagesandthediscriminator judges the output. After the training the network has to do the reverse way to display deconvoluted images. The last step is de deconcatenation of the sound images and the retransformation into a .wav ﬁle, so sound can be played.

# 

For more information read the paper: https://github.com/markus-weiss/AVGAN/blob/master/Forschungsprojekt%20(2).pdf
