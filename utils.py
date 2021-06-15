import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def plot_burger(array, color = "#ffe800", color2 = "#ff00ff", name = None):
    fig, ax = plt.subplots(2,1,figsize = (10,5), sharex = True)
    ax[0].imshow(array.reshape(1,1024), cmap="plasma", aspect="auto")
    ax[0].set_yticks([])
    ax[1].plot(array, color = color2)
    for axes in ax:
        [axes.spines[key].set_visible(False) for key in axes.spines]
        axes.tick_params(axis='both', colors=color)
    yt = np.around(ax[1].get_yticks(), 2)
    ax[1].set_yticklabels(yt, color = color, fontsize = 15)
    xt = ax[1].get_xticks()
    ax[1].set_xticklabels(xt, color = color, fontsize = 15)
    ax[1].set_xlabel("Longitud", fontsize = 20, color = color)
    ax[1].set_ylabel("Valor numérico", fontsize = 20, color = color)
    if name:
        fig.savefig(f"figs/{name}.png", transparent = True, bbox_inches = "tight")
    return fig, ax
    
def plot_navierstokes(predicted, true, t, color = "#ffe800", color2 = "#212121ff"):
    fig, ax = plt.subplots(1,3, figsize = (16,4), sharey = True)
    fig.patch.set_facecolor(color2)
    zero = ax[0].imshow(predicted)
    ax[0].set_title("Predichos\n", fontsize = 30, color = color)
    one = ax[1].imshow(true)
    ax[1].set_title("Verdaderos\n", fontsize = 30, color = color)
    two = ax[2].imshow(np.abs(predicted - true))
    ax[2].set_title("Diferencia\nabsoluta", fontsize = 30, color = color)
    ims = [zero, one, two]
    for i in range(3):
        ax[i].set_xticklabels("")
        ax[i].set_yticklabels("")
        ax[i].tick_params(left=False, bottom=False)
    cbar = fig.colorbar(ims[2])
    ax[1].set_xlabel(f"t = {t}", fontsize = 30, color = color)
    # Para cambiar el color de los "ticks" de la barra de color
    cbar.ax.yaxis.set_tick_params(color = color)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color = color)
    return fig, ax    

def plot_many(predicted, true, epochs, color = "#ffe800"):
    for i in range(predicted.shape[-1]):
        if len(str(i)) == 1:
            idx = f"0{i}"
        else:
            idx = i
        fig, ax = plot_navierstokes(predicted[:,:,i], true[:,:,i], idx, color)
        fig.savefig(f"figs/predicciones/ns_{epochs}epoch/img_{idx}.png", bbox_inches = 'tight', transparent=False)

def plot_metrics(loss_hist, mse_hist, c="#ffe800", name = None, epoch = 5):
    trn_loss, tst_loss = zip(*loss_hist)
    trn_mse, tst_mse = zip(*mse_hist)
    fig, ax = plt.subplots(1,2, figsize=(13,5))

    ax[0].plot(trn_loss, 'tab:orange', label='trn loss')
    ax[0].plot(tst_loss, 'tab:red', label='tst loss')
    ax[0].set_title("Pérdidas", fontsize = 30, color = c)
    ax[0].set_ylabel('pérdida', fontsize = 25, color = c)
    ax[1].plot(trn_mse, 'tab:green', label='trn mse')
    ax[1].plot(tst_mse, 'tab:blue', label='tst mse')
    ax[1].set_title("MSE", fontsize = 30, color = c)
    ax[1].set_ylabel('mse', fontsize = 25, color = c)
    for i in range(2):
        ax[i].set_xlabel('época', fontsize = 25, color = c)
        ax[i].tick_params(axis='both', colors=c)
        ax[i].legend(loc='upper left')
        ax[i].spines["left"].set_color(c)
        ax[i].spines["bottom"].set_color(c)
        yt = np.around(ax[i].get_yticks(), 3)
        ax[i].set_yticklabels(yt, color = c, fontsize = 15)
        xt = ax[i].get_xticks()
        ax[i].set_xticklabels([int(x) for x in xt], color = c, fontsize = 15)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
    if name:
        fig.savefig(f"figs//metrics/{name}.png", transparent = True, bbox_inches = "tight")
        
def create_gif(epochs):
    # Create the frames
    frames = []
    imgs = glob.glob(f"figs/predicciones/ns_{epochs}epoch/*.png")
    imgs.sort()
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(f'gifs/{epochs}_epochs.gif', format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=150, loop=0)