"# naivebayes"
naive bayes trained on MAGIC gamma telescope data 2004

#there is a telescope there are particles that are hitting the telescope and there is detector that records certain patterns how this light hit the camera and from the pattern we predict weither it is gamma or hadron

#pattern that is collected in the camera
#Attribute information:

1.  fLength: continuous # major axis of ellipse [mm]
2.  fWidth: continuous # minor axis of ellipse [mm]
3.  fSize: continuous # 10-log of sum of content of all pixels [in #phot]
4.  fConc: continuous # ratio of sum of two highest pixels over fSize [ratio]
5.  fConc1: continuous # ratio of highest pixel over fSize [ratio]
6.  fAsym: continuous # distance from highest pixel to center, projected onto major axis [mm]
7.  fM3Long: continuous # 3rd root of third moment along major axis [mm]
8.  fM3Trans: continuous # 3rd root of third moment along minor axis [mm]
9.  fAlpha: continuous # angle of major axis with vector to origin [deg]
10. fDist: continuous # distance from origin to center of ellipse [mm]
11. class: g,h # gamma (signal), hadron (background)

#. Class Distribution:
g = gamma (signal): 12332
h = hadron (background): 6688

#the second dataset

#Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

Attribute Information:

#variance of Wavelet Transformed image (continuous)
#skewness of Wavelet Transformed image (continuous)
#curtosis of Wavelet Transformed image (continuous)
#entropy of image (continuous)
#class (integer)
