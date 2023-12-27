from ast import Dict
from particleUtility import *
import pickle

#Read the lines from the pickle file
with open("lines.pkl","rb") as file:
    lines=pickle.load(file)



def getSticked(case:cloud,lines):
    ps=case.positions[np.logical_not(case.active)] #Select the sticked particles
    sticked={key:0 for key in lines.keys()}

    for p in ps:
        dm=np.zeros(len(lines.keys()))
        ks=[key for key in lines.keys()]
        for i,key in enumerate(ks):
            lp=lines[key]
            dm[i]=np.min(np.linalg.norm(p*1e3 -lp,axis=1))

        sticked[ks[np.argmin(dm)]]+=1

    #####################################################################
    gens=[]
    for key in lines.keys():
        gens.append(int(key.split("_")[1]))

    gens=np.sort(np.unique(gens))
    ngens=np.zeros(len(gens))
    dep_gen=np.zeros(len(gens))
    for key in lines.keys():
        i=int(key.split("_")[1])
        dep_gen[i]+=sticked[key]
        ngens[i]+=1

    return gens,100*(dep_gen/ngens)/case.parcelsAddedTotal



fld="/g100_scratch/usera06chi/a06chi00/gspasov/FullA_project/runs/timeDependent/DEM/drivingTest"
inlet=cloud("2",f"{fld}/inletBC")
mixed=cloud("2",f"{fld}/mixedBC")

x1,y1=getSticked(inlet,lines)
x2,y2=getSticked(mixed,lines)

np.save("./save/inlet",np.array([x1,y1]))
np.save("./save/mixed",np.array([x2,y2]))

plt.scatter(x1,y1,label="Inlet",zorder=1)
plt.scatter(x2,y2,label="Mixed",zorder=2)


plt.grid(which="both",zorder=0)
plt.xlabel("Generation")
plt.ylabel("Mean total deposition %")

#plt.yscale("log")


plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("plots/DrivingStrategy.png")
