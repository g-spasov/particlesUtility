
from .case_SI import simInhale
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

import matplotlib
import itertools
import os
import numpy as np


def calculate_text_length(text, fontsize):
    text_path = TextPath((0, 0), text, size=fontsize)
    #bb = text_path.get_extents(Affine2D().scale(fontsize))
    bb=text_path.get_extents()
    # Length of the bounding box in x-direction
    length = bb.width
    return length

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

    bigName=" ".join(cN)



    
    n=10
    fontsize=7*n

    length=calculate_text_length(bigName,fontsize=fontsize)
    length0=calculate_text_length("_",fontsize=fontsize)

    nn= len(cN)+1


    m=np.int32(np.ceil(length/(nn*length0)))  #+len(cN)

    print(length,length0,m,nn)


    #m=len(cA)+3


    matplotlib.rc('font',size=fontsize)

    extras_space=1
    n=n+fontsize//n
    n=int(1.6*n)
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n
    

     #fontsize//n
    fig = plt.figure(figsize=[m+2*n+2*extras_space,n],constrained_layout=True)
    gs = fig.add_gridspec(1,m+2*n+2*extras_space)

    gd=fig.add_subplot(gs[0,:m])
    td=fig.add_subplot(gs[0,extras_space+m:m+n])
    ld=fig.add_subplot(gs[0,extras_space+m+n:])

    plt.setp(gd.spines.values(), linewidth=5)
    plt.setp(td.spines.values(), linewidth=5)
    plt.setp(ld.spines.values(), linewidth=5)


    ##########################################################################################################################
    gd.grid(zorder=0)
    y=[100*case.totalStickes[0]/case.parcelsAddedTotal[0] for case in cA]
    #x=[1+0.5*i for i in range(len(y))]
    
    xmax=length/length0
    x=np.linspace(0,xmax,num=len(cA)+1,endpoint=True)

    x=[0.5*(x[i]+x[i+1]) for i in range(len(x)-1)]
    # x=np.arange(1,length/length0,1,dtype=int)
    if len(x)!=len(y):
        raise ValueError(f"len(x)={len(x)}, len(y)={len(cN)}")

    #x=[x[i]+0.7*i for i in range(len(x))]
    simInhale=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
    LES1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_LES1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))

    gd.hlines(np.sum(simInhale["DF"]),0,xmax,label="Exp. (Lizal et al. 2015)",color="k",linestyles='dashed',zorder=10)
    gd.hlines(np.sum(LES1["DF"]),0,xmax,label="LES1 (Koullapis et al. 2018)",color="grey",linestyles='dashed',zorder=10)

    for i in range(len(y)):
        gd.bar(x[i],y[i],width=0.9*xmax/len(y),zorder=i+2)

    _,ymax=gd.get_ylim()

    print(ymax)
    gd.set_ylim([0,((ymax)//5 +1)*5])
   
    #gd.set_xlabel("Mesh size")
    gd.set_ylabel("Total deposition fraction %")
    gd.set_xticks(x,[])
    #gd.set_xticks(x,cN)
    ###################################################################################
    td.grid(which="both",zorder=0)

    for i,case in enumerate(cA):
        td.scatter(case.simInhale[key]["patch"],case.simInhale[key]["DF"],label=cN[i],zorder=i+2)#,zorder=2,alpha=0.7)

    if sim:
        td.scatter(simInhale["patch"],simInhale["DF"],label="Exp. (Lizal et al. 2015)",c="k",zorder=7,alpha=0.8)

    td.set_ylim([0,8])
    td.set_xlabel("Patch number")
    # td.set_xticks(simInhale["patch"])

    td.set_xticks(simInhale["patch"][0::2])
    td.set_xticks(simInhale["patch"][1::2],minor=True)
    td.set_ylabel("Deposition fraction %")
    
    ############### LEGEND HANDLING
    hand_gd,lables_gd=gd.get_legend_handles_labels()
    hand_td,lables_td=td.get_legend_handles_labels()
    hnd=hand_gd+hand_td
    lbl=lables_gd+lables_td
    ncols=2 #(m+1*sim)//2
    fig.legend(flip(hnd,ncols),flip(lbl,ncols),loc='upper right',
               bbox_to_anchor=((n+m)/(m+2*n), 0.97), ncol=ncols, fancybox=True, shadow=True,
               fontsize=1.7*n)
    ###################################################################################(m+ncols+0.45*n)/(m+2*n)
    ld.grid(which="both",zorder=0)
    for i,case in enumerate(cA):
        x_,y_=case.getRelativeDeposition(pd)
        ld.scatter(x_,y_,zorder=i+2)
    ld.set_xlabel("Patch number")
    ld.set_xticks(simInhale["patch"][0::2])
    ld.set_xticks(simInhale["patch"][1::2],minor=True)
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

    fontsize=3*n

    matplotlib.rc('font',size=fontsize)

    
    n=n+fontsize//n
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n

    n=n+fontsize//n

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
    simInhale=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
    LES1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_LES1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))

    gd.hlines(np.sum(simInhale["DF"]),0.5,m+0.5,label="Exp. (Lizal et al. 2015)",color="k",linestyles='dashed',zorder=10)
    gd.hlines(np.sum(LES1["DF"]),0.5,m+0.5,label="LES1 (Koullapis et al. 2018)",color="grey",linestyles='dashed',zorder=10)

    for i in range(len(y)):
        gd.bar(x[i],y[i],width=1.5,zorder=i+2)

    gd.set_ylim([0,(max(y)//10 +1)*10])
   
    gd.set_xlabel("Patch number")
    gd.set_ylabel("Total deposition fraction %")
    gd.set_xticks(x,cN)

    plt.savefig(f"{filename}.png",bbox_inches="tight")


def plotDiameterTUL(case: simInhale,caseName:str,filename, pd=4.3e-6, key='4.2999999999999995e-06', sim=True ):
    """
        This program plots the total deposition over the simInhale geometry.
        Resolved per diameter and divides it in lower and upper DR
        Takes the following arguments:
        -cA : a list of simInhale cases to be compared
        -cN: a listo of strings being the names of the cases
        -name->str: path to save the plot
        -pd->float : particle's diameter
        -key->string: the particle diameter in key format
        -sim->bool: value that enables or si
    """
    
    n=10
    fontsize=7*n

    m=n  #+len(cN)
  
    matplotlib.rc('font',size=fontsize)

    extras_space=1
    n=n+fontsize//n
    n=int(1.6*n)
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n
    
    m=n
    fig = plt.figure(figsize=[m+2*n+2*extras_space,n],constrained_layout=True)
    gs = fig.add_gridspec(1,m+2*n+2*extras_space)

    gd=fig.add_subplot(gs[0,:m])
    ud=fig.add_subplot(gs[0,extras_space+m:m+n])
    ld=fig.add_subplot(gs[0,extras_space+m+n:])

    plt.setp(gd.spines.values(), linewidth=5)
    plt.setp(ud.spines.values(), linewidth=5)
    plt.setp(ld.spines.values(), linewidth=5)


    ##########################################################################################################################
    ## Global deposition 
    simInhale=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
    LES1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_LES1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
        
    wedel=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/Wedel.csv")
    les1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/LES1-p.csv")
    y=100*case.totalStickes/case.parcelsAddedTotal
    ds=np.unique(case.d)

    gd.grid(zorder=1)
    gd.scatter(ds*1e6,y,c="C0",label="Current")
    curr1,=gd.plot(ds*1e6,y,c="C0")
    gd.set_xlabel(r"$d(\mu m)$")
    gd.set_ylabel(r"Global depostion %")
    gd.set_yticks([10*i for i in range(11)])
    gd.set_xticks([i for i in range(1,11)])

    gd.scatter(wedel[:,0],wedel[:,1],c="C2",label="Wedel")
    curr2,=gd.plot(wedel[:,0],wedel[:,1],c="C2")

    gd.scatter(les1[:,0],les1[:,1],c="C1",label="LES1")
    curr3,=gd.plot(les1[:,0],les1[:,1],c="C1")
        # gd.scatter(4.3,np.sum(LES1["DF"]),c="C2")
        # print(np.sum(LES1["DF"]))
    gd.scatter(4.3,np.sum(simInhale["DF"]),c="k",label="simInhale-exp")
                
    gd.legend()


        ## Upper deposition
    wedel=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/Wedel_upper.csv")
    les1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/LES1-p2.csv")
    d_les=[0.5,1,2,2.5,4,4.3,5,6,8,10]
    y_les=[np.sum(les1[:2,i+1]) for i in range(len(d_les))]
    ud.grid(which="both",zorder=0)

    y=[]
    for key in case.simInhale.keys():
        x=case.simInhale[key]["patch"]
        y.append(np.sum(case.simInhale[key]["DF"][0:2]))

    ud.scatter(ds*1e6,y,c="C0",label="Current")
    ud.plot(ds*1e6,y,c="C0")

    ud.scatter(d_les,y_les,c="C1",label="LES1")
    ud.plot(d_les,y_les,c="C1")

    ud.scatter(wedel[:,0],wedel[:,1],c="C2",label="Wedel")
    ud.plot(wedel[:,0],wedel[:,1],c="C2")

    ud.scatter(4.3,np.sum(simInhale["DF"][:2]),c="k",label="simInhale-exp")

    ud.set_xlabel(r"$d(\mu m)$")
    ud.set_ylabel(r"Upper global depostion %")
    ud.set_yticks([10*i for i in range(11)])
    ud.set_xticks([i for i in range(1,11)])
    
    ud.legend()
    ## Lower deposition
    ld.grid(which="both",zorder=0)
    wedel=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/Wedel_lower.csv")

    les1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/LES1-p2.csv")
    d_les=[0.5,1,2,2.5,4,4.3,5,6,8,10]
    y_les=[np.sum(les1[2:,i+1]) for i in range(len(d_les))]

    y=[]
    for key in case.simInhale.keys():
        x=case.simInhale[key]["patch"]
        y.append(np.sum(case.simInhale[key]["DF"][2:]))

    ld.scatter(d_les,y_les,c="C1",label="LES1")
    ld.plot(d_les,y_les,c="C1")

    ld.scatter(wedel[:,0],wedel[:,1],c="C2",label="Wedel")
    ld.plot(wedel[:,0],wedel[:,1],c="C2")

    ld.scatter(4.3,np.sum(simInhale["DF"][2:]),c="k",label="simInhale-exp")
    ld.scatter(ds*1e6,y,c="C0",label="Currend")
    ld.plot(ds*1e6,y,c="C0")
    ld.set_xlabel(r"$d(\mu m)$")
    ld.set_ylabel(r"Lower global depostion %")
    ld.set_yticks([10*i for i in range(11)])
    ld.set_xticks([i for i in range(1,11)])
    ld.legend()
    plt.savefig(f"./{filename}.png",bbox_inches="tight")


def dfVariability(cases: list[simInhale],filename: str, pd=4.3e-6, key='4.3e-06', sim="LES" ):
    """"
        This fuction plots the  Deposition Fraction variability of the presented case againts the LES/RANS data.
        
        Input:
            -cases : a list of simInhale data of which the data will be shown there is no naming of the cases
            -filename: the name of the plot can be a path+name, withouth the extension, the file will be a .png
            - pd: particle diameter
            - key: particle diameter key
            - sim: "LES" or "RANS" option denoting which dataset to compare againts
    """
    n=10
    m=n

    fontsize=3*n

    matplotlib.rc('font',size=fontsize)

    
    n=n+fontsize//n
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n

    n=n+fontsize//n

    m=n

    fig=plt.figure(figsize=[n,n],constrained_layout=True)
    
    gs = fig.add_gridspec(1,n)
    gd=fig.add_subplot(gs[0,:n])

    gd.grid(zorder=0)
    
    #Reas the simInhale-experimental data
    siexp=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
    x=siexp["patch"]

    if sim=="LES":
        LES1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_LES1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
        LES2=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_LES2_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
        LES3=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_LES3_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))

        y=[]
        for i in range(len(LES1["patch"])):
            y.append([np.min([LES1["DF"][i],LES2["DF"][i],LES3["DF"][i]]),np.max([LES1["DF"][i],LES2["DF"][i],LES3["DF"][i]])  ])
    if sim=="RANS":
        RANS1=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_RANS1_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
        RANS2=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_RANS2_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
        RANS3=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/DF_RANS3_Koullapis.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))

        y=[]
        for i in range(len(RANS1["patch"])):
            y.append([np.min([RANS1["DF"][i],RANS2["DF"][i],RANS3["DF"][i]]),np.max([RANS1["DF"][i],RANS2["DF"][i],RANS3["DF"][i]])])


    ydata=[]

    for i in range(len(siexp["patch"])):
        #case.simInhale[key]["patch"],case.simInhale[key]["DF"]
        tmp=[case.simInhale[key]["DF"][i] for case in cases]
        tmp2=[np.min(tmp),np.max(tmp)]
        ydata.append(tmp2)

    for i,x_ in enumerate(siexp["patch"]):
        if i==0:
            gd.vlines(x_,ydata[i][0],ydata[i][1],colors="C0",label="Current")
            gd.vlines(x_+0.1,y[i][0],y[i][1],colors="C1",label=sim+" (Koullapis et al. 2018)")
        else:
            gd.vlines(x_,ydata[i][0],ydata[i][1],colors="C0")
            gd.vlines(x_+0.1,y[i][0],y[i][1],colors="C1")


    gd.scatter(x,siexp["DF"],c="k",label="Exp. (Lizal et al. 2015)")

    gd.set_xlabel("Patch number")
    gd.set_ylabel("Deposition fraction %")
    gd.set_xticks(x)

    gd.set_ylim([0,7.5])
    fig.legend()

    plt.savefig(f"./{filename}.png",bbox_inches="tight")

def segmentDFperGeneration(cases: list[simInhale],cases_names: list[str],filename: str, pd=4.3e-6, key='4.3e-06',savenames=None, **kwargs):
    """
    This function generates the plot of the Deposition fraction per segment of a given generation generation, i.e.
        DF vs. Generation number

        Inputs:
            -cases: list of simINhale cases
            -cases_names: The names to be printed inside the legend
            -filename: The name of the figure to be saved
            -pd: particle's diameter
            -key: particle's diameter key

            -savenames=None #List of names to give to the case which will be appended to the relative files used in the case.getLineSticked and case.loadLineSticked() methods
            -**kwargs refer to optional arguments used in the case.getLineSticked and case.loadLineSticked methods i.e.:
                -savedata=False #Name if the data are not preset will be calculated using the case.getLineSticked() method, and if one wants to save the data, this should be True
                -savefolder="./generationDepositionData" : The folder in which the data will be saved, this is the default value
    """

    n=15
    fontsize=4*n
    matplotlib.rc('font',size=fontsize)

    n=n+fontsize//n
    matplotlib.rcParams['lines.linewidth'] = 0.35*n
    matplotlib.rcParams['lines.markersize']=1.4*n
    matplotlib.rcParams['grid.linewidth']=0.2*n

    n=n+fontsize//n

    fig=plt.figure(figsize=[n,n],constrained_layout=True)
    
    gs = fig.add_gridspec(1,n)
    gd=fig.add_subplot(gs[0,:n])
    gd.grid(zorder=0)

    for i,case in enumerate(cases):
        kwargs["savename"]=savenames[i]
        try:
            case.loadLineSticked(**kwargs)
        except:
            print(f"Generation data not preset for case {savenames[i]}, going to calculate them...")
            case.getLineSticked(**kwargs)
        gens=case.gens

        ymean=case.generationSticked[key]

        gd.scatter(gens,ymean,label=cases_names[i])
        


    gd.set_xlabel("Generation number")
    gd.set_ylabel("Deposition fraction %")
    gd.set_xticks(gens)
    gd.set_ylim([0,gd.get_ylim()[1]])
    fig.legend(loc='upper left',bbox_to_anchor=(0.1,0.95),fancybox=True, shadow=True)
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #       ncol=3, fancybox=True, shadow=True)

    plt.savefig(f"./{filename}.png",bbox_inches="tight")