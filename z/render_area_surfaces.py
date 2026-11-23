"""Render measurement and completion grids on one signed height scale."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import Patch


def render(out, xy, pitch, baseline, height, model, filled, confirmed):
    x, y = xy[0,:,0], xy[:,0,1]
    limits = x[0]-pitch/2, x[-1]+pitch/2, y[0]-pitch/2, y[-1]+pitch/2
    norm = colors.Normalize(-15, 15)
    cmap = plt.get_cmap("RdYlBu_r").copy()
    cmap.set_bad("#e9edf1")
    accent = "#733f87"
    fig, axes = plt.subplots(1,3,figsize=(15,5.6))
    fig.subplots_adjust(left=.065,right=.96,bottom=.26,top=.81,wspace=.30)
    fig.suptitle("Высоты П-образной детали",fontsize=22,fontweight="bold",x=.065,y=.96,ha="left")
    for ax,z,title in zip(axes,[baseline,height,model],["Все ракурсы","Ближайшие ракурсы","Карта с достройкой"]):
        im=ax.imshow(z,origin="lower",extent=limits,cmap=cmap,norm=norm,interpolation="nearest")
        ax.set(title=title,xlabel="X, мм",ylabel="Y, мм")
    axes[-1].scatter(xy[...,0][filled],xy[...,1][filled],s=3,c=accent,marker=".",linewidths=0)
    fig.legend(handles=[Patch(color=accent,label="Достроено по плоскости"),
                        Patch(color="#e9edf1",label="Нет значения")],loc="lower center",
               bbox_to_anchor=(.5,.010),ncol=2,frameon=False)
    cb=fig.colorbar(im,cax=fig.add_axes([.28,.155,.44,.027]),orientation="horizontal",extend="both")
    cb.set_label("Z относительно стола, мм")
    fig.savefig(out/"height_maps.png",dpi=165)
    plt.close(fig)

    fig=plt.figure(figsize=(10,7))
    ax=fig.add_subplot(projection="3d")
    fig.subplots_adjust(left=.03,right=.95,bottom=.19,top=.89)
    fig.suptitle("3D-вид измеренных и достроенных высот",fontsize=18,fontweight="bold",y=.96)
    measured=confirmed & np.isfinite(height)
    uncertain = np.isfinite(height) & ~measured
    ax.scatter(xy[...,0][uncertain],xy[...,1][uncertain],height[uncertain],c="#8092a4",
               s=2,linewidth=0,alpha=.16,depthshade=False)
    dots=ax.scatter(xy[...,0][measured],xy[...,1][measured],height[measured],c=height[measured],
                    cmap=cmap,norm=norm,s=4,linewidth=0,depthshade=False)
    ax.scatter(xy[...,0][filled],xy[...,1][filled],model[filled],c=accent,s=6,linewidth=0,depthshade=False)
    ax.set(xlabel="X, мм",ylabel="Y, мм",zlabel="Z, мм",xlim=limits[:2],ylim=limits[2:])
    ax.set_box_aspect((np.ptp(x),np.ptp(y),np.ptp(ax.get_zlim())))
    ax.view_init(elev=37,azim=67)
    fig.legend(handles=[Patch(color=accent,label="Достроено"), Patch(color="#8092a4",label="Без согласия карт"),
                        Patch(color="#e88236",label="Согласованные по Z")],
               loc="lower center",bbox_to_anchor=(.5,.008),ncol=3,frameon=False)
    fig.colorbar(dots,cax=fig.add_axes([.25,.126,.50,.024]),orientation="horizontal",
                 extend="both",label="Z относительно стола, мм")
    fig.savefig(out/"height_3d.png",dpi=165)
    plt.close(fig)
