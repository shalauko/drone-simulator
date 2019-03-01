# For testing main.py should be changed for accept filename as parameters
# and past it in appropriate place in path generator.

import os
names = [line.rstrip('\n') for line in open('trajectories/filenames.txt')]

for i in range(0,10):
    print(names[i])
    os.system('python main.py trajectories/' + names[i])


