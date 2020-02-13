import numpy as np 
#Observation space order of motor angles is :
#FLH,FLK,FRH,FRK,BLH,BLK,BRH,BRK,FLA,FRA,BLA,BRA
raw_data = np.load("raw_angle_data_0.4R.npy")
print(raw_data.shape)

cd = raw_data[0]
def clean_half_step(data, stance = 0):
    clean_data= np.zeros([50,13])
    val = data.shape[0]-1
    indexes = np.arange(0,val-0.001,val/49)
    indexes = np.concatenate([indexes, [val]])
    indexes = np.array([int(round(x)) for x in indexes])
    ii= 0 
    for index in indexes:
        clean_data[ii] = data[index]
        clean_data[ii,0] = int(ii) + 50*stance
        ii= ii+1
    return clean_data

def clean_total_data(data, swing_start =True):
    stance = 0
    if(not swing_start):
        stance = 1  
    final_data = []
    for current_data in data:
        cleaned_current_data = clean_half_step(current_data, stance)
        final_data.append(cleaned_current_data)
        if(stance == 0):
            stance = 1
        else:
            stance = 0
    pass
    final_data = np.concatenate(final_data)
    return final_data
dat = clean_total_data(raw_data)
np.save("Simulation_Data/clean_sim_data_0.4R.npy", dat)

