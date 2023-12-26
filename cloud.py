import numpy as np
import os
import re




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

        self.id=None #Particles id
        self.active=None #Particles which are active
        self.d=None #Particles diameter
        self.positions=None #Particles position
        self.parcelsAddedTotal=None #Total number of injectd particles for wach diameter
        self.patchSticked=None #The number of sticked particles for each patch. Dictionary where the key is the patch
        self.depositionFraction=None #For each patch the deposition fraction is evaluated

        self.getParticleActive()
        self.getParticlesD()
        self.getParticlesPosition()
        
        self.getPatchAndStickedParticles()
        self.getTotalInjected()
        self.getParticleZoneInfo()
        self.calculateDepositionFraction()

        pass

    def getParticlesID(self):
        """This method gets particles' id.
            [WARNING: The method works perfectly buy the openFoam case does not make the particleID univocous.
        """

        id=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/origId",dtype=np.str_,delimiter="\n")
        start=np.where(id=="(")[0][0]
        end=np.where(id==")")[0][0]

        self.id=np.array(id[start+1:end],dtype=np.int32)
        del(id,start,end)

    def getParticleActive(self):
        """This method reads the active particles from the timeStep file"""
        a=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/active",dtype=np.str_,delimiter="\n")
        start=np.where(a=="(")[0][0]
        end=np.where(a==")")[0][0]

        self.active=np.array(a[start+1:end],dtype=np.bool_)
        del(a,start,end)

    def getParticlesD(self,user_rp=[],default_rp=[re.compile(r"\d+\{\d+\.\d+[eE]\-\d+\}"),re.compile(r"\d+\{\d+[eE]\-\d+\}")]):
        """This method gets the particles' diameter:
            Reads the timeStep folder and gets all the particles that has NOT EXITED their diameter
            In case of single diameter:
            [NOTE: In the case of a SINGLE diameter, the writte format may change from case to case, in order to manage the various cases the following
                    regular expression pattern have been defined:
                    1. re.compile(r"\d+\{\d+\.\d+[eE]\-\d+\}")
                    2. re.compile(r"\d+\{\d+[eE]\-\d+\}")
                In order to define our own regular expression patterns, provide them as a python list in the user_rp]
            
            The default_rp, which are the two previously mensioned regular expression patterns are executed by default, if not emptied.
            """
        d=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/d",dtype=np.str_,delimiter="\n")
        try:
            start=np.where(d=="(")[0][0]
            end=np.where(d==")")[0][0]
            self.d=np.array(d[start+1:end],dtype=np.float32)
            del(d,start,end)
        except:
            rp=default_rp+user_rp
            #pattern=re.compile(r"\d+\{\d+\.\d+[eE]\-\d+\}")
            #pattern2=re.compile(r"\d+\{\d+[eE]\-\d+\}")
            for string in d:
                tmp=[pattern.match(string) for pattern in rp]
                if np.sum([True if mtch else False for mtch in tmp])>0 :
                    self.d=np.float32(string.split("{")[1].split("}")[0])*np.ones(len(self.active),dtype=np.float32)
                
                
    def getParticlesPosition(self):
        """This method gets the particles' position from the timeStep file"""
        d1=np.genfromtxt(f"{self.casePath}/{self.timeStep}/lagrangian/kinematicCloud/positions",dtype=np.str_,delimiter="\n")
        start=np.where(d1=="(")[0][0]
        end=np.where(d1==")")[0][0]

        self.positions=np.loadtxt(d1[start+1:end],converters=convert,usecols=(0,1,2))
        del(d1,start,end)
        

    def getTotalInjected(self):
        """This method reads the file inside timeStep/uniform/lagrangian/kinematicCloud/kinematicCloudOutputProperties
            which contains the injection data for each particles and filters the ones that are inside the inlet
            [WANING: The filtering is specific for this geometry, if one does not want to filter modify the mothod argomenti filter=False ]"""
        tmp = np.genfromtxt(f"{self.casePath}/{self.timeStep}/uniform/lagrangian/kinematicCloud/kinematicCloudOutputProperties",dtype=np.str_,delimiter="\n")
        self.diameters=np.array([np.float64(j.split("model")[1]) for j in tmp if "model" in j ])
        self.parcelsAddedTotal=[np.int32(j.split()[1].split(";")[0]) for j in tmp if "parcelsAddedTotal" in j]

                

    def getPatchAndStickedParticles(self):
        """This method collects all the partciles that have sticked inside the geometry.
            It's done trough the script MeshLagrangianPost.sh which collects the data from
            postProcessing/lagrangian/kinematicCloud/Patch (NOT particleZoneInfo)
            and collects them in a single file which allows a faster identification since looping over all the data takes a while
        """
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
                os.system(f"bash {os.path.dirname(__file__)}/MergeLagrangianPost.sh {fld}")
            tmp=np.loadtxt(pP+f"/{dir}/collectedData.txt",dtype=dt,converters=convert,usecols=(0,2,3,4,7))
            self.patchSticked[dir]=tmp


    def calculateDepositionFraction(self):
        """
            This uses the data obtain from the self.getPatchAndStickedParticles() and defines how many and which particles have sticked
            on a given patch.
        """
        self.depositionFraction={}

        tmp=np.zeros(len(self.diameters),dtype=np.int32)
        for key in self.patchSticked.keys():
            tmp+=self.sticPatchPerParticle[key]
            

        self.totalStickes=tmp
        for key in self.patchSticked.keys():
            self.depositionFraction[key]=100* self.sticPatchPerParticle[key]/self.parcelsAddedTotal

    
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