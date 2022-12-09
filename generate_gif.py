import imageio
filenames = ["./figures/capacity_gif/fig_Eve_farther_{}.png".format(i) for i in range(30)]
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('secrecy_beamforming.gif', images)