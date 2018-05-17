# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Vizdoom Execution
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

"""
USAGE:
python plotresults.py -f "logs/lastrun.log" -x "INFO:__main__:Episode reward:"
"""

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', default="log.txt", help="Log file to read from")
    parser.add_argument('-x', '--plotdatax', default="| final timestep |", help="Name of datastring to plot")
    #parser.add_argument('-y', '--plotdatay', default="INFO:__main__:Episode reward:", help="Name of datastring to plot")


    args = parser.parse_args()

    if args.file is not None:
        with open(args.file, 'r') as f:
            content = f.read().splitlines()

        xdata = []
        ydata = []
        for line in content:
            if line.startswith(args.plotdatax):
                xdata.append(float(line[len(args.plotdatax)+1:-1].replace(" ","")))
            # if line.startswith(args.plotdatay):
            #     x
            #     ydata.append(float(line[len(args.plotdata)+1:]))
        #print ("Plotting "+ args.plotdatax)

        #sorting
        occurencecount = Counter(xdata)
        keys = np.asarray([*occurencecount.keys()])
        values = np.asarray([*occurencecount.values()])
        a = np.column_stack((keys,values))
        a = a[np.lexsort(np.fliplr(a).T)]
        for i in range(1, a[:,1].size):
            a[i,1] = a[i,1] + a[i-1, 1]

        plt.plot(a[:,0], a[:,1])

        plt.ylabel("number of totally finished mazes")
        plt.xlabel('Step Count')
        plt.show()





	


if __name__ == '__main__':
    main()