from ast import Dict
from particleUtility import *
import pickle

#Read the lines from the pickle file
with open("lines.pkl","rb") as file:
    lines=pickle.load(file)

#Count the generations
gens=[]
for key in lines.keys():
    gens.append(int(key.split("_")[1]))

numb=[] #=#np.zeros(len(np.unique(gens)))
gen=[]
for i in np.unique(gens):
    gen.append(i)
    numb.append(np.sum(gens==i))

numb=np.array(numb)

#####################################
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

    data={i:[] for i in gens}
    

    for key in lines.keys():
        i=int(key.split("_")[1])
        data[i].append(sticked[key])

    means=np.zeros(len(gens))
    stdv=np.zeros(len(gens))
    for i in gens:
        means[i]=np.mean(np.array(data[i])/case.parcelsAddedTotal)
        stdv[i]=np.std(np.array(data[i])/case.parcelsAddedTotal)


    return gens,100*means,stdv

# fld="/g100_scratch/usera06chi/a06chi00/gspasov/FullA_project/runs/timeDependent/DEM/MeshValidation"
# c3M=cloud("2",f"{fld}/3.6M")
# c5M=cloud("2",f"{fld}/5.3M")
# c8M=cloud("2",f"{fld}/8.1M")
# c11M=cloud("2",f"{fld}/11.7M")
# c14M=cloud("2",f"{fld}/14.6M")

# print("Getting sticked 1")
# x1,y1,z1=getSticked(c3M,lines)

# print("Getting sticked 2")
# x2,y2,z2=getSticked(c5M,lines)

# print("Getting sticked 3")
# x3,y3,z3=getSticked(c8M,lines)

# print("Getting sticked 4")
# x4,y4,z4=getSticked(c11M,lines)

# print("Getting sticked 5")
# x5,y5,z5=getSticked(c14M,lines)


# np.save("./save/3M_std",np.array([x1,y1,z1]))
# np.save("./save/5M_std",np.array([x2,y2,z1]))
# np.save("./save/8M_std",np.array([x3,y3,z1]))
# np.save("./save/11M_std",np.array([x4,y4,z1]))
# np.save("./save/14M_std",np.array([x5,y5,z1]))
x1,y1,z1=np.load("save/3M_std.npy")
x2,y2,z2=np.load("save/5M_std.npy")
x3,y3,z3=np.load("save/8M_std.npy")
x4,y4,z4=np.load("save/11M_std.npy")
x5,y5,z5=np.load("save/14M_std.npy")

numb=np.array([2**i for i in range(len(numb))])

plt.errorbar(x1,y1*numb,yerr=3*z1*numb,label="3.6M",zorder=1,fmt="o")
plt.errorbar(x2,y2*numb,yerr=3*z2*numb,label="5.3M",zorder=2,fmt="o")
plt.errorbar(x3,y3*numb,yerr=3*z3*numb,label="8.1M",zorder=3,fmt="o")
plt.errorbar(x4,y4*numb,yerr=3*z4*numb,label="11.7M",zorder=4,fmt="o")
plt.errorbar(x5,y5*numb,yerr=3*z5*numb,label="14.6M",zorder=5,fmt="o")

plt.grid(which="both",zorder=0)
plt.xlabel("Generation")
plt.ylabel("Total deposition fraction %")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=5, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig("plots/Total/Mesh.png")
plt.close()



##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#Time
fld="/g100_scratch/usera06chi/a06chi00/gspasov/FullA_project/runs/timeDependent/DEM/timeEvolution"
fld_steady="/g100_scratch/usera06chi/a06chi00/gspasov/FullA_project/runs/timeDependent/DEM/MeshValidation"

steady=cloud("2",f"{fld_steady}/8.1M")
time=cloud("2",f"{fld}/8.1M")
time_noDRW=cloud("1.89",f"{fld}/8.1M_noDRW")

# print("Getting steady")
# x1,y1,z1=getSticked(steady,lines)

# print("Getting time")
# x2,y2,z2=getSticked(time,lines)

# print("Getting time2")
# x3,y3,z3=getSticked(time_noDRW,lines)


# np.save("./save/steady_std",np.array([x1,y1,z1]))
# np.save("./save/time_std",np.array([x2,y2,z3]))
# np.save("./save/time_noDRW_std",np.array([x3,y3,z3]))

x1,y1,z1=np.load("./save/steady_std.npy")
x2,y2,z2=np.load("./save/time_std.npy")
x3,y3,z3=np.load("./save/time_noDRW_std.npy")


plt.errorbar(x1,y1*numb,yerr=3*z1*numb,label="Steady",zorder=1,fmt="o")
plt.errorbar(x2,y2*numb,yerr=3*z2*numb,label="Time",zorder=2,fmt="o")
#plt.errorbar(x3,y3*numb,yerr=3*z3*numb,label="Time_noDRW",zorder=3,fmt="o")

plt.grid(which="both",zorder=0)
plt.xlabel("Generation")
plt.ylabel("Total deposition %")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=5, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig("plots/Total/Time_contribution.png")
plt.close()

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
# fld="/g100_scratch/usera06chi/a06chi00/gspasov/FullA_project/runs/timeDependent/DEM/drivingTest"
# inlet=cloud("2",f"{fld}/inletBC")
# mixed=cloud("2",f"{fld}/mixedBC")

# x1,y1,z1=getSticked(inlet,lines)
# x2,y2,z2=getSticked(mixed,lines)

# np.save("./save/inlet_std",np.array([x1,y1,z1]))
# np.save("./save/mixed_std",np.array([x2,y2,z1]))
x1,y1,z1=np.load("./save/inlet_std.npy")
x2,y2,z2=np.load("./save/mixed_std.npy")


plt.errorbar(x1,y1*numb,yerr=3*z1*numb,label="Inlet",zorder=1,fmt="o")
plt.errorbar(x2,y2*numb,yerr=3*z2*numb,label="Mixed",zorder=2,fmt="o")


plt.grid(which="both",zorder=0)
plt.xlabel("Generation")
plt.ylabel("Mean total deposition %")

#plt.yscale("log")


plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=5, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig("plots/Total/Driving.png")
