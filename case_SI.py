"""
This class contains all the adaptations of the cloud class to tread the simInhale simulation
"""

from .cloud import cloud

class simInhale(cloud):

    def __init__(self, timeStep, casePath=".") -> None:
        super().__init__(timeStep, casePath)
        self.simInhalePatchID={"1":[f"Patch{i}" for i in range(1,6)],"2":["Patch6","Patch7"],"3":["P3"],
                                "4":["P4"],"5":["P5"],"6":["P6"],"7":["P7"],"8":["P8"],"9":["P9"],"10":["P10"],
                                "11":["P11"],"12":["P12"],"13":["P13_1"],"14":["P14_1"],"15":["P15_1"],"16":["P16_1"],
                                "17":["P17_1"],"18":["P18_1"],"19":["P19_1"],"20":["P20_1"],"21":["P21_1"],"22":["P22_1"]}


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