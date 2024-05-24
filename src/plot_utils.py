import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib import gridspec

#FUNCTIONS FOR PLOTTING GRAPHICS ON AN IMAGE 

_bbox = (154,154)

def make_grid(bounds=_bbox,n=11):
    patchsize = bounds//n
    corner = -bounds//2 + patchsize//2
    coords = list(range(corner,-corner+patchsize,patchsize))

    return coords

def plot_bbox_and_rfs(
    segmap, bbox_color="red", rfs_color="cyan", bbox=_bbox
):
    # makes it easier to work with numpy coords by making the center of the image the origin
    transform = lambda x, y: tuple(tb.transform_coord_system((-y, x)))

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.axis("off")
    rectangle_params = {
        "corner": transform(-bbox[0] // 2, -bbox[1] // 2),
        "height": -bbox[1],
        "width": bbox[0],
    }
    bounding_box = patches.Rectangle(
        rectangle_params["corner"],
        rectangle_params["width"],
        rectangle_params["height"],
        ec=bbox_color,
        fill=False,
        lw=2.5,
    )
    coords = make_grid(bounds=bbox[0],n=11)

    for i in coords:
        for j in coords:
            x, y = transform(i, j)
            rf = patches.Circle(
                (x, y), radius=25, fill=False, lw=1.2, ls=":", ec="cyan"
            )
            ax.add_patch(rf)
            plt.scatter(x, y, marker="^", color=rfs_color, s=75,ec="black")
    plt.imshow(segmap, cmap="gray")
    ax.add_patch(bounding_box)
    fig = plt.gcf()

    #fname = plot_fname(segmap,flag="bbox_and_rfs")
    
    #fig.savefig(fname+".png", format="png",dpi=100)

    plt.show()