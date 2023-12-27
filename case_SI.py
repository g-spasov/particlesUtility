"""
This class contains all the adaptations of the cloud class to tread the simInhale simulation
"""

from .cloud import cloud,convert
import numpy as np
import os
import re
import pickle

class simInhale(cloud):

    def __init__(self, timeStep, casePath=".") -> None:
        super().__init__(timeStep, casePath)

        #This maps the simInale subdivision to your subdivision
        self.simInhalePatchID={"1":[f"Patch{i}" for i in range(1,6)],"2":["Patch6","Patch7"],"3":["P3"],
                                "4":["P4"],"5":["P5"],"6":["P6"],"7":["P7"],"8":["P8"],"9":["P9"],"10":["P10"],
                                "11":["P11"],"12":["P12"],"13":["P13_1"],"14":["P14_1"],"15":["P15_1"],"16":["P16_1"],
                                "17":["P17_1"],"18":["P18_1"],"19":["P19_1"],"20":["P20_1"],"21":["P21_1"],"22":["P22_1"]}

        self.getTotalInjected()
        self.getParticleZoneInfo()
        self.comparisionSimInhale()

        try:
            with open(f"{os.path.dirname(__file__)}/simInhale/lines.pkl","rb") as file:
                self.lines=pickle.load(file)
        except:
            self.lines=None
            raise Warning("The lines.pk does not exists so no generation analysis will be done.")

        


    def getTotalInjected(self,filter=True):
        """This method reads the file inside timeStep/uniform/lagrangian/kinematicCloud/kinematicCloudOutputProperties
            which contains the injection data for each particles and filters the ones that are inside the inlet
            [WANING: The filtering is specific for this geometry, if one does not want to filter modify the mothod argomenti filter=False ]"""
        tmp = np.genfromtxt(f"{self.casePath}/{self.timeStep}/uniform/lagrangian/kinematicCloud/kinematicCloudOutputProperties",dtype=np.str_,delimiter="\n")
        self.diameters=np.array([np.float64(j.split("model")[1]) for j in tmp if "model" in j ])
        self.parcelsAddedTotal=[np.int32(j.split()[1].split(";")[0]) for j in tmp if "parcelsAddedTotal" in j]

        #Let's save the particles that are inside the pipe and are active
        if filter:
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


    def getParticleZoneInfo(self):
        """
        This method calculated the "particleZoneInfor" in the postProcessing/lagrangian/kinematicCloud.
        This gets us which particle has passed from each defined zone. Thanks to this, one is able to
        reconstruct which particle has exited from which site.
        """
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

    def comparisionSimInhale(self):
        simInhale=np.loadtxt(f"{os.path.dirname(__file__)}/simInhale/simInhale_data.csv",dtype=np.dtype([("patch",np.uint8),("DF",np.float64)]),converters=float)
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

    def calculatePassingParticles(self,d):
        """
        This method uses the particleZoneInfo and defines how many particle has entered each geometry piece.
        """
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
        """
        This method defines how many particles go on the left or on the right at a given bifurcation
        with respect to the all the particles that has entered a given bifurcation.
        """
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
        """
        This method defines the relative deposition, i.e. how many have sticked with respect the one that has entered
        """


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
    
    def getlineSticked(self,savedata=True):
        """
        This method defines the following quantities:
            -self.linesSticked : The number of particles that have sticked in each line of the self.lines varaible.
                            It is a dictionary of dictionaries where each key is the particles' diameter and then the inner
                            dictionary are the line id-keys.
            -self.gensSticked: The number of particles that have sticked in each generation, this time it's a dictionary with the keys being
                                again the diameters, but the values are arrays ordered per generation [0th, 1st, ..., 7th]

            -self.linesStickedMean: The average number of particle sticked per generation, averaged over total number of injected and total
                                    number of segments present per generation.
            -self.ngens : Number of segmenets per generation
            -self.gens  : Array containg the generations [0,1,2,3,4,..,7]
        """

        ps=self.positions[np.logical_not(self.active)]
        ds=self.d[np.logical_not(self.active)]
        self.linesSticked={str(d):{key:0 for key in self.lines.keys()} for d in self.diameters}

        for ip,p in enumerate(ps):
            dm=np.zeros(len(self.lines.keys()))
            ks=[key for key in self.lines.keys()]
            for i,key in enumerate(ks):
                lp=self.lines[key]
                dm[i]=np.min(np.linalg.norm(p*1e3 -lp,axis=1))
            self.linesSticked[str(ds[ip])][ks[np.argmin(dm)]]+=1

    #####################################################################
        gens=[]
        for key in self.lines.keys():
            gens.append(int(key.split("_")[1]))

        gens=np.sort(np.unique(gens))

        self.gensSticked={str(d):{i:[] for i in gens} for d in self.diameters}
        ngens=np.zeros(len(gens))
        for key1 in self.linesSticked.keys():
            for key in self.lines.keys():
                i=int(key.split("_")[1])
                self.gensSticked[key1][i].append(self.linesSticked[key1][key])
                ngens[i]+=1


        self.ngens=ngens

        self.gens=gens
        self.linesStickedMean={str(d):[100*np.mean(self.gensSticked[str(d)][j])/self.parcelsAddedTotal[i] for j in gens ] for i,d in enumerate(self.diameters)}
        self.stdgenStocked={str(d):[100*np.std(self.gensSticked[str(d)][j])/self.parcelsAddedTotal[i] for j in gens ] for i,d in enumerate(self.diameters)}