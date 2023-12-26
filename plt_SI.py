from case_SI import simInhale
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import itertools

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def triPlot(cA: list[simInhale],cN: list[str],filename,pd=4.3e-6 ,key='4.2999999999999995e-06',sim=True ):
    """
    This function takes as arguments:
         1.an array of clouds 
         2. array of names to give each cloud
         3. Particle diameter
         3. the key corresponding to each particle used for the Total deposition
    and creates a plot composed of the following plots:
    1. Global deposition
    2. Total deposition
    3. Local deposition
    """


    n=10
    m=len(cA)+3

    fontisize=3*n

    matplotlib.rc('font',size=fontisize)

    
    n=n+fontisize//n
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n

    n=n+fontisize//n
    fig = plt.figure(figsize=[m+2*n,n],constrained_layout=True)
    gs = fig.add_gridspec(1,m+2*n)

    gd=fig.add_subplot(gs[0,:m])
    td=fig.add_subplot(gs[0,m:m+n])
    ld=fig.add_subplot(gs[0,m+n:])



    ##########################################################################################################################
    gd.grid(zorder=0)
    y=[100*case.totalStickes[0]/case.parcelsAddedTotal[0] for case in cA]
    #x=[1+0.5*i for i in range(len(y))]
    x=np.arange(1,len(y)+0.1,1,dtype=int)
    if len(x)!=len(y):
        raise ValueError

    x=[x[i]+0.7*i for i in range(len(x))]
    simInhale=np.loadtxt("./simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
    LES1=np.loadtxt("./simInhale/DF_LES1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))

    gd.hlines(np.sum(simInhale["DF"]),0.5,m+0.5,label="Exp. (Lizal et al. 2015)",color="k",linestyles='dashed',zorder=10)
    gd.hlines(np.sum(LES1["DF"]),0.5,m+0.5,label="LES1 (Koullapis et al. 2018)",color="grey",linestyles='dashed',zorder=10)

    for i in range(len(y)):
        gd.bar(x[i],y[i],width=1.5,zorder=i+2)

    gd.set_ylim([0,(max(y)//10 +1)*10])
   
    gd.set_xlabel("Mesh size")
    gd.set_ylabel("Total deposition fraction %")
    gd.set_xticks(x,cN)
    ###################################################################################
    td.grid(which="both",zorder=0)

    for i,case in enumerate(cA):
        td.scatter(case.simInhale[key]["patch"],case.simInhale[key]["DF"],label=cN[i],zorder=i+2)#,zorder=2,alpha=0.7)

    if sim:
        td.scatter(simInhale["patch"],simInhale["DF"],label="Exp. (Lizal et al. 2015)",c="k",zorder=7,alpha=0.8)

    td.set_ylim([0,8])
    td.set_xlabel("Patch")
    td.set_xticks(simInhale["patch"])
    td.set_ylabel("Deposition fraction %")
    
    ############### LEGEND HANDLING
    hand_gd,lables_gd=gd.get_legend_handles_labels()
    hand_td,lables_td=td.get_legend_handles_labels()
    hnd=hand_gd+hand_td
    lbl=lables_gd+lables_td
    ncols=2 #(m+1*sim)//2
    fig.legend(flip(hnd,ncols),flip(lbl,ncols),loc='upper center',
               bbox_to_anchor=((m+ncols+0.45*n)/(m+2*n), 0.95), ncol=ncols, fancybox=True, shadow=True,
               fontsize=1.7*n)
    ###################################################################################
    ld.grid(which="both",zorder=0)
    for i,case in enumerate(cA):
        x_,y_=case.getRelativeDeposition(pd)
        ld.scatter(x_,y_,zorder=i+2)
    ld.set_xlabel("Patch")
    ld.set_xticks(simInhale["patch"])
    ld.set_ylabel("Deposition efficiency %")
    
    _,ymax=ld.get_ylim()
    ld.set_ylim([0,ymax])
    ld.set_yticks(np.arange(0,int(ymax)+1,5,dtype=np.int32,))
    #####################################################################################
    plt.savefig(f"{filename}.png",bbox_inches="tight")


def plotTotalDF(cA: list[simInhale],cN: list[str],filename, pd=4.3e-6, key='4.2999999999999995e-06', sim=True ):
    """
        This program plots the total deposition over the simInhale geometry.
        Takes the following arguments:
        -cA : a list of simInhale cases to be compared
        -cN: a listo of strings being the names of the cases
        -name->str: path to save the plot
        -pd->float : particle's diameter
        -key->string: the particle diameter in key format
        -sim->bool: value that enables or si
    """
    n=10
    m=len(cA)+3

    fontisize=3*n

    matplotlib.rc('font',size=fontisize)

    
    n=n+fontisize//n
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n

    n=n+fontisize//n

    fig=plt.figure(figsize=[m,n],constrained_layout=True)
    
    gs = fig.add_gridspec(1,m)
    gd=fig.add_subplot(gs[0,:m])

    ##########################################################################################################################
    gd.grid(zorder=0)
    y=[100*case.totalStickes[0]/case.parcelsAddedTotal[0] for case in cA]
    #x=[1+0.5*i for i in range(len(y))]
    x=np.arange(1,len(y)+0.1,1,dtype=int)
    if len(x)!=len(y):
        raise ValueError

    x=[x[i]+0.7*i for i in range(len(x))]
    simInhale=np.loadtxt("./simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
    LES1=np.loadtxt("./simInhale/DF_LES1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))

    gd.hlines(np.sum(simInhale["DF"]),0.5,m+0.5,label="Exp. (Lizal et al. 2015)",color="k",linestyles='dashed',zorder=10)
    gd.hlines(np.sum(LES1["DF"]),0.5,m+0.5,label="LES1 (Koullapis et al. 2018)",color="grey",linestyles='dashed',zorder=10)

    for i in range(len(y)):
        gd.bar(x[i],y[i],width=1.5,zorder=i+2)

    gd.set_ylim([0,(max(y)//10 +1)*10])
   
    gd.set_xlabel("Mesh size")
    gd.set_ylabel("Total deposition fraction %")
    gd.set_xticks(x,cN)

    plt.savefig(f"{filename}.png",bbox_inches="tight")


