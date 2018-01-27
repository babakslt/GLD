Number of Lensing Parameters: #6
- l_amp = 1.5   # Einstein radius
- l_xcen = 0.0  # x position of center
- l_ycen = 0.0  # y position of center
- l_axrat = 1.0 # minor-to-major axis ratio
- l_pa = 0.0    # major-axis position angle (degrees) c.c.w. from x axis
lpar = n.asarray([l_amp, l_xcen, l_ycen, l_axrat, l_pa])

Range: [-2.5,2.5]

Number of Gaussian blob Parameters: #7
- g_amp = 1.0   # peak brightness value
- g_sig = 0.05  # Gaussian "sigma" (i.e., size)
- g_xcen = 0.0  # x position of center
- g_ycen = 0.0  # y position of center
- g_axrat = 1.0 # minor-to-major axis ratio
- g_pa = 0.0    # major-axis position angle (degrees) c.c.w. from x axis
gpar = n.asarray([g_amp, g_sig, g_xcen, g_ycen, g_axrat, g_pa])

