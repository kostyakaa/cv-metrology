"""Render measured signed heights and mutually exclusive evidence classes at their true XY."""
import argparse
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.figure import Figure


def render(folder):
    with np.load(folder/'area.npz') as data:
        z=data['height_mm']; mask=data['confirmed']; pitch=float(data['cell_mm']); origin=data['xy_origin']
        classes=['no_data','no_texture','occluded','conflict','measured_uncertain','measured_confirmed','filled_or_inferred']
        flags=np.stack([data[k] for k in classes])
    if not (flags.sum(axis=0)==1).all() or not np.isfinite(z[mask]).all():
        raise ValueError('Invalid status partition or nonfinite confirmed height')
    labels=['Нет данных','Нет текстуры','Закрыто','Конфликт','Недостаточно подтверждений','Подтверждено','Достроено']
    hist=flags.sum(axis=(1,2))
    ny,nx=z.shape
    extent=[origin[0]-pitch/2,origin[0]+(nx-.5)*pitch,origin[1]-pitch/2,origin[1]+(ny-.5)*pitch]
    fig=Figure(figsize=(12,5.4),layout='constrained')
    FigureCanvasAgg(fig)
    ax,bx=fig.subplots(1,2)
    status=ListedColormap(['#272d33','#9d815e','#644e82','#dc7845','#aab2b8','#49a982','#3698bb'])
    norm=BoundaryNorm(np.arange(-.5,7),status.N)
    ax.imshow(flags.argmax(axis=0),extent=extent,origin='lower',interpolation='nearest',cmap=status,norm=norm)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=status(i),label=f'{labels[i]} ({n:,})') for i,n in enumerate(hist) if n],
              loc='upper center',bbox_to_anchor=(.5,-.14),ncol=2,fontsize=8,frameon=False)
    ax.set_title('Статус ячеек')
    import matplotlib as mpl
    cmap=mpl.colormaps['RdYlBu_r'].copy()
    cmap.set_bad('#272d33')
    col=Normalize(-15,15)
    image=bx.imshow(np.where(mask,z,np.nan),origin='lower',extent=extent,interpolation='nearest',cmap=cmap,norm=col)
    bx.set_title('Измеренная высота')
    fig.colorbar(image,ax=bx,shrink=.78,label='Z от плоскости стола, мм',extend='both')
    for axes in (ax,bx):
        axes.set_xlabel('X станка, мм');axes.set_ylabel('Y станка, мм')
    fig.savefig(folder/'height.png',dpi=155)
    fig=Figure(figsize=(9,6),layout='constrained')
    FigureCanvasAgg(fig)
    ax=fig.add_subplot(projection='3d')
    yy,xx=np.where(mask)
    scatter=ax.scatter(origin[0]+pitch*xx,origin[1]+pitch*yy,z[mask],c=z[mask],s=4,
                       cmap=cmap,norm=col,depthshade=False,linewidths=0)
    ax.set(xlabel='X станка, мм',ylabel='Y станка, мм',zlabel='Z, мм',xlim=extent[:2],ylim=extent[2:])
    low=min(-15,float(z[mask].min()) if mask.any() else -15)
    high=max(15,float(z[mask].max()) if mask.any() else 15)
    ax.set_zlim(low,high); ax.set_box_aspect((nx*pitch,ny*pitch,high-low))
    ax.view_init(elev=36,azim=65)
    ax.set_title(f'Подтверждённые измерения · {mask.sum():,} ячеек')
    fig.colorbar(scatter,ax=ax,shrink=.7,label='Z, мм',extend='both')
    fig.savefig(folder/'height_3d.png',dpi=160)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    render(p.parse_args().folder)
