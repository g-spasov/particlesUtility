import numpy as np
import os
import re
import matplotlib
import matplotlib.pyplot as plt
import typing
import itertools

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def convert(str):
    if str.startswith(b"("):
        return float(str[1:])
    elif str.endswith(b")"):
        return float(str[:-1])
    else:
        return float(str)
    

class cloud:
    def __init__(self,timeStep,casePath=".") -> None:
        self.timeStep=timeStep
        self.casePath=casePath
        self.getParticleActive()
        self.getParticlesD()
        self.getParticlesPosition()
        self.getCurrentRegion()
        
        self.getPatchAndStickedParticles()
        self.getTotalInjected()
        self.getParticleZoneInfo()
        self.calculateDepositionFraction()

        

        self.simInhalePatchID={"1":[f"Patch{i}" for i in range(1,6)],"2":["Patch6","Patch7"],"3":["P3"],
                                "4":["P4"],"5":["P5"],"6":["P6"],"7":["P7"],"8":["P8"],"9":["P9"],"10":["P10"],
                                "11":["P11"],"12":["P12"],"13":["P13_1"],"14":["P14_1"],"15":["P15_1"],"16":["P16_1"],
                                "17":["P17_1"],"18":["P18_1"],"19":["P19_1"],"20":["P20_1"],"21":["P21_1"],"22":["P22_1"]}
        
        self.comparisionSimInhale()
        pass

    def getParticlesID(self):
        id=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/origId",dtype=np.str_,delimiter="\n")
        start=np.where(id=="(")[0][0]
        end=np.where(id==")")[0][0]

        self.id=np.array(id[start+1:end],dtype=np.int32)
        del(id,start,end)

    def getParticlesD(self):
        d=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/d",dtype=np.str_,delimiter="\n")
        try:
            start=np.where(d=="(")[0][0]
            end=np.where(d==")")[0][0]
            self.d=np.array(d[start+1:end],dtype=np.float32)
            del(d,start,end)
        except:
            pattern=re.compile(r"\d+\{\d+\.\d+[eE]\-\d+\}")

            for string in d:
                if(pattern.match(string)):
                    self.d=np.float32(string.split("{")[1].split("}")[0])*np.ones(len(self.active),dtype=np.float32)
                
                
        

    def getParticleActive(self):
        a=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/active",dtype=np.str_,delimiter="\n")
        start=np.where(a=="(")[0][0]
        end=np.where(a==")")[0][0]

        self.active=np.array(a[start+1:end],dtype=np.bool_)
        del(a,start,end)

    def getParticlesPosition(self):
        d1=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/positions",dtype=np.str_,delimiter="\n")
        start=np.where(d1=="(")[0][0]
        end=np.where(d1==")")[0][0]

        self.positions=np.loadtxt(d1[start+1:end],converters=convert,usecols=(0,1,2))
        del(d1,start,end)
        
    def getCurrentRegion(self):
        """Does not work, this is the cell and not the region as a Patch"""
        cls=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/coordinates",dtype=np.str_,delimiter="\n")
        start=np.where(cls=="(")[0][0]
        end=np.where(cls==")")[0][0]

        self.cellId=np.loadtxt(cls[start+1:end],dtype=np.int32,converters=convert,usecols=(4))
        del(cls,start,end)

    def getTotalInjected(self):
        tmp = np.genfromtxt(f"{self.casePath}/{self.timeStep}/uniform/lagrangian/kinematicCloud/kinematicCloudOutputProperties",dtype=np.str_,delimiter="\n")
        self.diameters=np.array([np.float64(j.split("model")[1]) for j in tmp if "model" in j ])
        self.parcelsAddedTotal=[np.int32(j.split()[1].split(";")[0]) for j in tmp if "parcelsAddedTotal" in j]

        #Let's save the particles that are inside the pipe and are active
        R=10e-3
        pos=self.positions[self.active][:,[0,2]]
        self.insidePipe=np.where(np.linalg.norm(pos,axis=1)<=R)[0]

        self.sticPatchPerParticle={}
        for key in self.patchSticked.keys():
            self.sticPatchPerParticle[key]=np.empty(len(self.diameters),dtype=np.int32)
            for i,d in enumerate(self.diameters):
                self.sticPatchPerParticle[key][i]=np.sum(np.isclose(self.patchSticked[key]["d"],d))
        #Let us eliminate the particles that are inside the pipe
        for i,d in enumerate(self.diameters):
            self.parcelsAddedTotal[i]-=np.sum(self.d[self.insidePipe]==d)
                

    def calculateDepositionFraction(self):
        self.depositionFraction={}

        tmp=np.zeros(len(self.diameters),dtype=np.int32)
        for key in self.patchSticked.keys():
            tmp+=self.sticPatchPerParticle[key]
            

        self.totalStickes=tmp
        print(tmp,self.parcelsAddedTotal)
        for key in self.patchSticked.keys():
            self.depositionFraction[key]=100* self.sticPatchPerParticle[key]/self.parcelsAddedTotal

                   

    def getPatchAndStickedParticles(self):
        pP=self.casePath+"/postProcessing/lagrangian/kinematicCloud"
        if(not os.path.isdir(pP)):
            raise EnvironmentError("The case path is not valid or there is no kinematicCloud postProcessing!")
        
        self.patchSticked={}
        dt=np.dtype([('sticking time', np.float32), ('Position', np.float64, (3,)),("d",np.float64)])
        for dir in os.listdir(pP):
            if "particleZoneInfo" in dir:
                continue
            if not os.path.isfile(pP+f"/{dir}/collectedData.txt"):
                fld=pP+f"/{dir}"
                os.system(f"bash MergeLagrangianPost.sh {fld}")
            tmp=np.loadtxt(pP+f"/{dir}/collectedData.txt",dtype=dt,converters=convert,usecols=(0,2,3,4,7))
            self.patchSticked[dir]=tmp
    
    def getParticleZoneInfo(self):
        self.outletMap={}
        for i in range(1,11):
            self.outletMap[f"{i}"]=f"{i+12}"
        
        pP=self.casePath+"/postProcessing/lagrangian/kinematicCloud"
        if(not os.path.isdir(pP)):
            raise EnvironmentError("The case path is not valid or there is no kinematicCloud postProcessing!")
        
        self.particleZoneInfo={}
        # origID      	origProc	(x y z)	time0	age	d0	d	mass0	mass
        for dir in os.listdir(pP):
            if "particleZoneInfo" not in dir:
                continue

            dirs=os.listdir(pP+f"/{dir}")
            f_dirs=[float(time) for time in dirs]
            i=np.argmax(f_dirs)
            tmp=np.loadtxt(pP+f"/{dir}/{dirs[i]}/particles.dat",dtype=np.float64,usecols=(7),converters=convert)
            val=dir.split("particleZoneInfo")[-1]
            self.particleZoneInfo[self.outletMap[val]]=tmp
            del(tmp,dirs,f_dirs,i)

    def calculatePassingParticles(self,d):
        # self.simInhalePatchID={"1":[f"Patch{i}" for i in range(1,6)],"2":["Patch6","Patch7"],"3":["P3"],
        #                         "4":["P4"],"5":["P5"],"6":["P6"],"7":["P7"],"8":["P8"],"9":["P9"],"10":["P10"],
        #                         "11":["P11"],"12":["P12"],"13":["P13_1"],"14":["P14_1"],"15":["P15_1"],"16":["P16_1"],
        #                         "17":["P17_1"],"18":["P18_1"],"19":["P19_1"],"20":["P20_1"],"21":["P21_1"],"22":["P22_1"]}
        if(str(d) not in self.simInhale.keys() and (not np.sum(np.abs(self.diameters-d)/d < 1e-3))):
            print("Available diameters are:",np.unique(self.diameters))
            raise ValueError(f"The diameter {d} is not present in the simulation")
        
        self.passingParticles={}
        ##
        ##LET'S do the LEFT part
        self.passingParticles["13"]=np.sum(self.particleZoneInfo["13"]==d)+np.sum(self.patchSticked["P13_1"]["d"]==d)
        ##
        self.passingParticles["14"]=np.sum(self.particleZoneInfo["14"]==d)+np.sum(self.patchSticked["P14_1"]["d"]==d)
        ##
        self.passingParticles["15"]=np.sum(self.particleZoneInfo["15"]==d)+np.sum(self.patchSticked["P15_1"]["d"]==d)
        ##
        self.passingParticles["16"]=np.sum(self.particleZoneInfo["16"]==d)+np.sum(self.patchSticked["P16_1"]["d"]==d)
        ##
        self.passingParticles["17"]=np.sum(self.particleZoneInfo["17"]==d)+np.sum(self.patchSticked["P17_1"]["d"]==d)
        ##
        self.passingParticles["18"]=np.sum(self.particleZoneInfo["18"]==d)+np.sum(self.patchSticked["P18_1"]["d"]==d)
        ##
        self.passingParticles["19"]=np.sum(self.particleZoneInfo["19"]==d)+np.sum(self.patchSticked["P19_1"]["d"]==d)
        ##
        self.passingParticles["20"]=np.sum(self.particleZoneInfo["20"]==d)+np.sum(self.patchSticked["P20_1"]["d"]==d)
        ##
        self.passingParticles["21"]=np.sum(self.particleZoneInfo["21"]==d)+np.sum(self.patchSticked["P21_1"]["d"]==d)
        ##
        self.passingParticles["22"]=np.sum(self.particleZoneInfo["22"]==d)+np.sum(self.patchSticked["P22_1"]["d"]==d)
        ##
        self.passingParticles["7"]=self.passingParticles["13"]+self.passingParticles["16"]+np.sum(self.patchSticked["P7"]["d"]==d)
        ##
        self.passingParticles["6"]=self.passingParticles["14"]+self.passingParticles["15"]+np.sum(self.patchSticked["P6"]["d"]==d)
        ##
        self.passingParticles["5"]=self.passingParticles["6"]+self.passingParticles["7"]+np.sum(self.patchSticked["P5"]["d"]==d)
        ##
        self.passingParticles["4"]=self.passingParticles["5"]+np.sum(self.patchSticked["P4"]["d"]==d)
        ##
        ##LET'S do the RIGHT part
        self.passingParticles["21"]=np.sum(self.particleZoneInfo["21"]==d)+np.sum(self.patchSticked["P21_1"]["d"]==d)
        ##
        self.passingParticles["19"]=np.sum(self.particleZoneInfo["19"]==d)+np.sum(self.patchSticked["P19_1"]["d"]==d)
        ##
        self.passingParticles["18"]=np.sum(self.particleZoneInfo["18"]==d)+np.sum(self.patchSticked["P18_1"]["d"]==d)
        ##
        self.passingParticles["17"]=np.sum(self.particleZoneInfo["17"]==d)+np.sum(self.patchSticked["P17_1"]["d"]==d)
        ##
        self.passingParticles["20"]=np.sum(self.particleZoneInfo["20"]==d)+np.sum(self.patchSticked["P20_1"]["d"]==d)
        ##
        self.passingParticles["22"]=np.sum(self.particleZoneInfo["22"]==d)+np.sum(self.patchSticked["P22_1"]["d"]==d)
        ##
        self.passingParticles["9"]=self.passingParticles["21"]+self.passingParticles["19"]+np.sum(self.patchSticked["P9"]["d"]==d)
        ##
        self.passingParticles["11"]=self.passingParticles["20"]+self.passingParticles["17"]+np.sum(self.patchSticked["P11"]["d"]==d)
        ##
        self.passingParticles["12"]=self.passingParticles["22"]+self.passingParticles["18"]+np.sum(self.patchSticked["P12"]["d"]==d)
        ##
        self.passingParticles["10"]=self.passingParticles["11"]+self.passingParticles["12"]+np.sum(self.patchSticked["P10"]["d"]==d)
        ##
        self.passingParticles["8"]=self.passingParticles["9"]+self.passingParticles["10"]+np.sum(self.patchSticked["P8"]["d"]==d)
        ##
        self.passingParticles["3"]=self.passingParticles["4"]+self.passingParticles["8"]+np.sum(self.patchSticked["P3"]["d"]==d)
        ##
        self.passingParticles["2"]=self.passingParticles["3"]+np.sum(self.patchSticked["Patch6"]["d"]==d)+np.sum(self.patchSticked["Patch7"]["d"]==d)
        ##
        self.passingParticles["1"]=self.passingParticles["2"]
        for i in self.simInhalePatchID["1"]:
            self.passingParticles["1"]+=np.sum(self.patchSticked[i]["d"]==d)


    def passingFraction(self,d):
        self.calculatePassingParticles(d)
        self.passingFrac={}

        self.passingFrac["13"]=self.passingParticles["13"]/(self.passingParticles["13"]+self.passingParticles["16"])
        self.passingFrac["16"]=self.passingParticles["16"]/(self.passingParticles["13"]+self.passingParticles["16"])

        self.passingFrac["14"]=self.passingParticles["14"]/(self.passingParticles["14"]+self.passingParticles["15"])
        self.passingFrac["15"]=self.passingParticles["15"]/(self.passingParticles["15"]+self.passingParticles["14"])

        self.passingFrac["7"]=self.passingParticles["7"]/(self.passingParticles["7"]+self.passingParticles["6"])
        self.passingFrac["6"]=self.passingParticles["6"]/(self.passingParticles["7"]+self.passingParticles["6"])

        self.passingFrac["18"]=self.passingParticles["18"]/(self.passingParticles["18"]+self.passingParticles["22"])
        self.passingFrac["22"]=self.passingParticles["22"]/(self.passingParticles["18"]+self.passingParticles["22"])

        self.passingFrac["20"]=self.passingParticles["20"]/(self.passingParticles["20"]+self.passingParticles["17"])
        self.passingFrac["17"]=self.passingParticles["17"]/(self.passingParticles["20"]+self.passingParticles["17"])

        self.passingFrac["21"]=self.passingParticles["21"]/(self.passingParticles["21"]+self.passingParticles["19"])
        self.passingFrac["19"]=self.passingParticles["19"]/(self.passingParticles["21"]+self.passingParticles["19"])

        self.passingFrac["11"]=self.passingParticles["11"]/(self.passingParticles["11"]+self.passingParticles["12"])
        self.passingFrac["12"]=self.passingParticles["12"]/(self.passingParticles["11"]+self.passingParticles["12"])

        self.passingFrac["10"]=self.passingParticles["10"]/(self.passingParticles["10"]+self.passingParticles["9"])
        self.passingFrac["9"]=self.passingParticles["9"]/(self.passingParticles["10"]+self.passingParticles["9"])

        self.passingFrac["8"]=self.passingParticles["8"]/(self.passingParticles["8"]+self.passingParticles["4"])
        self.passingFrac["4"]=self.passingParticles["4"]/(self.passingParticles["8"]+self.passingParticles["4"])


    def getRelativeDeposition(self,d):
        if(str(d) not in self.simInhale.keys() and (not np.sum(np.abs(self.diameters-d)/d < 1e-3))):
            print("Available diameters are:",np.unique(self.diameters))
            raise ValueError(f"The diameter {d} is not present in the simulation")
        
        self.calculatePassingParticles(d)
        self.relativeDepositionFraction={}

        x=np.arange(1,23)
        y=np.zeros(len(x))
        

        for j,key in enumerate(self.simInhalePatchID.keys()):
            
            self.relativeDepositionFraction[key]=0
            for i in self.simInhalePatchID[key]:
                self.relativeDepositionFraction[key]+=np.sum(self.patchSticked[i]["d"]==d)
            
            self.relativeDepositionFraction[key]*=100/self.passingParticles[key]  
            y[j]=self.relativeDepositionFraction[key]
        
        return x,y





    def plotDepositionFraction(self):
        #Lets plot the deposition fraction of for each diameter
        #y-axis is the deposition fraction
        #x-axis is the patch
        for i,d in enumerate(self.diameters):
            plt.figure()
            plt.title(f"Deposition fraction for diameter {d}")
            plt.xlabel("Patch")
            plt.ylabel("Deposition fraction")

            y=np.empty(len(self.patchSticked.keys()))
            for j,key in enumerate(self.patchSticked.keys()):
                y[j]=self.depositionFraction[key][i]
            
            plt.plot(y)
            plt.xticks(np.arange(len(self.patchSticked.keys())),self.patchSticked.keys(),rotation=45)

            #Let's create the folder if it does not exist
            if(not os.path.isdir("./plots")):
                os.mkdir("./plots")
            
            #Save with tight layou
            plt.savefig(f"./plots/depositionFraction_{d}.png",bbox_inches="tight")
            plt.close()

    def comparisionSimInhale(self):
        simInhale=np.loadtxt("simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]),converters=float)

        #Let's construct the correspective simInhale vectors
        self.simInhale={}
        for i,d in enumerate(self.diameters):
            #self.simInhale[str(d)]=np.empty_like(simInhale)
            tmp=np.empty_like(simInhale)
            tmp["patch"]=simInhale["patch"]
            for p,key in enumerate(self.simInhalePatchID.keys()):
                tmp["DF"][p]=np.sum([self.depositionFraction[patch][i] for patch in self.simInhalePatchID[key]])
            
            self.simInhale[str(d)]=tmp
            del(tmp)

    def plotComparisonSimInhale(self,d):
        """Plots the present deposition fraction and the simInhale deposition fraction
            The simInhale deposition fraction contains particles of 4.6mum"""
        if(str(d) not in self.simInhale.keys() and (not np.sum(np.abs(self.diameters-d)/d < 1e-3))):
            print("Available diameters are:",np.unique(self.diameters))
            raise ValueError(f"The diameter {d} is not present in the simulation")
        
        #Create the plot folder if not present
        if(not os.path.isdir("./plots")):
            os.mkdir("./plots")
        
        d=self.diameters[np.argmin(np.abs(self.diameters-d))]
        


        simInhale=np.loadtxt("simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]))
        plt.figure()
        plt.title(f"Deposition fraction for diameter {d} comparision with simInhale")
        plt.xlabel("Patch")
        plt.ylabel("Deposition fraction")

        plt.scatter(simInhale["patch"],simInhale["DF"],label="simInhale")
        plt.scatter(self.simInhale[str(d)]["patch"],self.simInhale[str(d)]["DF"],label="present")

        plt.xticks(simInhale["patch"])
        plt.legend()
        plt.yscale("log")
        plt.ylim([0.01,100])
        plt.savefig(f"./plots/comparisionSimInhale_{d}.png",bbox_inches="tight")
        plt.close()



class boundary:
    def __init__(self,path) -> None:
        self.path=path
        if(not os.path.isfile(path+"/constant/polyMesh/boundary")):
            raise EnvironmentError("The case path does not have a constant directory or polyMesh/boundary")
        
        self.boundary=path+"/constant/polyMesh/boundary"

    def readCaseBoundary(self):
        bfile=np.genfromtxt(self.boundary,dtype=np.str_,delimiter="\n")
        start=np.where(bfile=="(")[0][0]
        end=np.where(bfile==")")[0][0]
        
        self.boundaryNames=np.empty(np.int32(bfile[start-1]),dtype=np.str_)
        self.startFace


def createPlots(cA: list[cloud],cN: list[str],pd=4.3e-6 ,key='4.2999999999999995e-06',sim=True ):
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
    gd.set_ylabel("Global deposition fraction %")
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
    td.set_ylabel("Total deposition fraction %")
    
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
    ld.set_ylabel("Local deposition fraction %")
    
    _,ymax=ld.get_ylim()
    ld.set_ylim([0,ymax])
    ld.set_yticks(np.arange(0,int(ymax)+1,5,dtype=np.int32,))
    #####################################################################################
    plt.savefig(f"Deposition_fractions.png",bbox_inches="tight")

