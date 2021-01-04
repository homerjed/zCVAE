# zCVAE

This is a Conditional Variational AutoEncoder (CVAE) for generating samples of cosmological structure. 

The model can interpolate a sample continuously through the minimum and maximum redshifts it is exposed to in training.

The encoder and decoder utilise a conditioning that translates the montonous redshift variable as is. This avoids the complex conversion of non-linear lookback redshift to categorical vectors.

<p align="center">
  <img src="https://github.com/homerjed/zCVAE/blob/main/imgs/z_anim.gif" />
</p>

<!--![z_interpolation](https://github.com/homerjed/zCVAE/blob/main/imgs/z_anim.gif)-->
